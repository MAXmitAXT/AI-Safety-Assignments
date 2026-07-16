#!/usr/bin/env python3
"""Generate Qwen3 responses under baseline or residual-stream ablation.

The script reuses the exact prompts and per-sample metadata from an existing
Week-1 JSONL file. It supports:

  baseline: no hook
  noop:     hook installed with alpha=0 (engineering control)
  probe:    project out the learned probe direction
  random:   project out a reproducible random direction orthogonal to probe

The primary intervention is applied once, at the last prompt token, during the
first/prefill call to the selected transformer block. Generation-token calls
are left untouched.

Output rows preserve the pair_id/direction/question_id/raw_output schema used
by convert_qwen_jsonl_to_chainscope_yaml.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(question_id: str, sample_idx: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{question_id}|{sample_idx}|{base_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def infer_direction(row: dict[str, Any]) -> str:
    direction = row.get("direction")
    if direction in {"fwd", "rev"}:
        return str(direction)
    qid = str(row.get("question_id") or row.get("id") or "")
    if qid.endswith("_fwd"):
        return "fwd"
    if qid.endswith("_rev"):
        return "rev"
    raise ValueError(f"Cannot infer fwd/rev direction from row: {row.keys()}")


def infer_pair_id(row: dict[str, Any]) -> str:
    if row.get("pair_id") is not None:
        return str(row["pair_id"])
    qid = str(row.get("question_id") or row.get("id") or "")
    if qid.endswith("_fwd") or qid.endswith("_rev"):
        return qid[:-4]
    raise ValueError("Cannot infer pair_id")


def infer_sample_idx(row: dict[str, Any], fallback: int) -> int:
    for key in ("sample_idx", "sample_id", "run_idx"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return int(fallback)


def normalize_tasks(rows: list[dict[str, Any]], base_seed: int) -> list[dict[str, Any]]:
    """Normalize existing Week-1 rows into one generation task per output row."""
    per_question_counter: dict[str, int] = {}
    tasks: list[dict[str, Any]] = []

    for row in rows:
        pair_id = infer_pair_id(row)
        direction = infer_direction(row)
        question_id = str(row.get("question_id") or row.get("id") or f"{pair_id}_{direction}")

        fallback_idx = per_question_counter.get(question_id, 0)
        sample_idx = infer_sample_idx(row, fallback_idx)
        per_question_counter[question_id] = fallback_idx + 1

        prompt = row.get("prompt")
        question = row.get("question") or row.get("question_text")
        if prompt is None and question is None:
            raise ValueError(f"Row {question_id}/{sample_idx} has neither prompt nor question")

        seed_value = row.get("seed")
        seed = int(seed_value) if seed_value is not None else stable_seed(question_id, sample_idx, base_seed)

        task = dict(row)
        task.update(
            {
                "pair_id": pair_id,
                "direction": direction,
                "question_id": question_id,
                "sample_idx": sample_idx,
                "run_idx": int(row.get("run_idx", sample_idx)),
                "seed": seed,
                "prompt": prompt,
                "question": question,
            }
        )
        tasks.append(task)

    return tasks


def select_tasks(
    tasks: list[dict[str, Any]], max_pairs: int | None, max_rows: int | None
) -> list[dict[str, Any]]:
    if max_pairs is not None:
        selected: set[str] = set()
        for task in tasks:
            if task["pair_id"] not in selected and len(selected) < max_pairs:
                selected.add(task["pair_id"])
        tasks = [t for t in tasks if t["pair_id"] in selected]
    if max_rows is not None:
        tasks = tasks[:max_rows]
    return tasks


def task_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["pair_id"]), str(row["direction"]), int(row["sample_idx"]))


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    for row in read_jsonl(path):
        try:
            keys.add(task_key(row))
        except Exception:
            continue
    return keys


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for value in candidates:
        if value is not None:
            return value
    raise AttributeError("Could not locate decoder layers; inspect the loaded model architecture")


def load_probe_direction(path: Path, hidden_size: int) -> torch.Tensor:
    array = np.load(path)
    if array.ndim != 1:
        raise ValueError(f"Direction must be 1D, got {array.shape}")
    if array.shape[0] != hidden_size:
        raise ValueError(
            f"Direction length {array.shape[0]} does not match model hidden_size {hidden_size}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Direction contains NaN or infinity")
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))
    norm = tensor.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Direction has invalid norm")
    return tensor / norm


def make_random_direction(probe: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_dir = torch.randn(probe.shape, generator=generator, dtype=torch.float32)
    # Orthogonalize so this control cannot accidentally contain the probe direction.
    random_dir = random_dir - torch.dot(random_dir, probe) * probe
    norm = random_dir.norm()
    if norm <= 1e-8:
        raise RuntimeError("Random direction was numerically degenerate")
    return random_dir / norm


def replace_hidden(output: Any, new_hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    if isinstance(output, list):
        return [new_hidden, *output[1:]]
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


def extract_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


@dataclass
class InterventionStats:
    dot_before: float | None = None
    dot_after: float | None = None
    delta_norm: float | None = None
    calls_modified: int = 0


class ResidualDirectionIntervention:
    """Project one direction out at the last token of the next prefill call."""

    def __init__(self, direction: torch.Tensor, alpha: float, hook_point: str):
        self.direction_cpu = direction.detach().float().cpu()
        self.alpha = float(alpha)
        self.hook_point = hook_point
        self._armed = False
        self._direction_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.stats = InterventionStats()

    def arm(self) -> None:
        self._armed = True
        self.stats = InterventionStats()

    def _direction_for(self, hidden: torch.Tensor) -> torch.Tensor:
        key = (hidden.device, hidden.dtype)
        if key not in self._direction_cache:
            self._direction_cache[key] = self.direction_cpu.to(
                device=hidden.device, dtype=hidden.dtype
            )
        return self._direction_cache[key]

    def _modify(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self._armed:
            return hidden
        self._armed = False  # first selected-layer call only: prompt prefill

        if hidden.ndim != 3:
            raise ValueError(f"Expected [batch, seq, hidden], got {tuple(hidden.shape)}")

        direction = self._direction_for(hidden)
        result = hidden.clone()
        last = result[:, -1, :]

        # Accumulate the projection coefficient in fp32 for numerical stability.
        coeff = torch.sum(last.float() * direction.float(), dim=-1, keepdim=True)
        projected = last - self.alpha * coeff.to(last.dtype) * direction
        result[:, -1, :] = projected

        with torch.no_grad():
            before = torch.mean(torch.sum(last.float() * direction.float(), dim=-1))
            after = torch.mean(torch.sum(projected.float() * direction.float(), dim=-1))
            delta = torch.mean(torch.linalg.vector_norm((projected - last).float(), dim=-1))
            self.stats = InterventionStats(
                dot_before=float(before.cpu()),
                dot_after=float(after.cpu()),
                delta_norm=float(delta.cpu()),
                calls_modified=1,
            )
        return result

    def output_hook(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        hidden = extract_hidden(output)
        return replace_hidden(output, self._modify(hidden))

    def input_hook(self, module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args or not torch.is_tensor(args[0]):
            raise TypeError("Expected decoder layer's first positional input to be hidden states")
        return (self._modify(args[0]), *args[1:])

    def register(self, layer: torch.nn.Module):
        if self.hook_point == "output":
            return layer.register_forward_hook(self.output_hook)
        if self.hook_point == "input":
            return layer.register_forward_pre_hook(self.input_hook)
        raise ValueError(f"Unknown hook point: {self.hook_point}")


def resolve_prompt_text(row: dict[str, Any], tokenizer: Any) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    question = row.get("question") or row.get("question_text")
    if not isinstance(question, str) or not question:
        raise ValueError("Missing prompt/question")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_final_answer(text: str) -> str:
    """A convenience label only; ChainScope should remain the official evaluator."""
    tail = text[-1200:]
    patterns = [
        r"(?i)final\s+answer\s*[:：]\s*\**\b(yes|no)\b",
        r"(?i)answer\s*[:：]\s*\**\b(yes|no)\b",
        r"(?i)\b(yes|no)\b\s*[.!]*\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, tail)
        if matches:
            return str(matches[-1]).upper()
    return "UNKNOWN"


def parse_dtype(value: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[value]


def build_generation_kwargs(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    temperature = args.temperature if args.temperature is not None else float(row.get("temperature", 0.7))
    top_p = args.top_p if args.top_p is not None else float(row.get("top_p", 0.9))
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(row.get("max_new_tokens", row.get("max_tokens", 2000)))
    )
    top_k = args.top_k if args.top_k is not None else row.get("top_k")

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
        "pad_token_id": args.pad_token_id,
        "eos_token_id": args.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
    return kwargs


def verify_hook_alignment(
    model: torch.nn.Module,
    tokenizer: Any,
    layer: torch.nn.Module,
    layer_idx: int,
    hook_point: str,
    direction: torch.Tensor,
    prompt_text: str,
) -> None:
    print("\n=== Hook verification ===")
    input_device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)

    captured: dict[str, torch.Tensor] = {}

    if hook_point == "output":
        def capture_output(module, hook_args, output):
            captured["hidden"] = extract_hidden(output).detach().float().cpu()
        handle = layer.register_forward_hook(capture_output)
        hidden_state_index = layer_idx + 1
    else:
        def capture_input(module, hook_args):
            captured["hidden"] = hook_args[0].detach().float().cpu()
        handle = layer.register_forward_pre_hook(capture_input)
        hidden_state_index = layer_idx

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Capture hook did not run")

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) > hidden_state_index:
        reference = hidden_states[hidden_state_index].detach().float().cpu()
        max_diff = float(torch.max(torch.abs(captured["hidden"] - reference)))
        print(f"Hook tensor vs outputs.hidden_states[{hidden_state_index}] max abs diff: {max_diff:.6g}")
        if max_diff > 5e-4:
            print("WARNING: layer convention may not match the representation used for probing.")
    else:
        print("Model did not expose a comparable hidden_states tuple; inspect probing code manually.")

    # Alpha=0 must be a numerical no-op on logits.
    with torch.inference_mode():
        baseline_logits = model(**inputs, use_cache=False).logits[:, -1, :].detach().float().cpu()

    noop = ResidualDirectionIntervention(direction, alpha=0.0, hook_point=hook_point)
    noop_handle = noop.register(layer)
    noop.arm()
    with torch.inference_mode():
        noop_logits = model(**inputs, use_cache=False).logits[:, -1, :].detach().float().cpu()
    noop_handle.remove()
    logits_diff = float(torch.max(torch.abs(baseline_logits - noop_logits)))
    print(f"Alpha=0 last-token logits max abs diff: {logits_diff:.6g}")

    # Alpha=1 should remove the direction component at the selected token.
    ablator = ResidualDirectionIntervention(direction, alpha=1.0, hook_point=hook_point)
    ablation_handle = ablator.register(layer)
    ablator.arm()
    with torch.inference_mode():
        _ = model(**inputs, use_cache=False)
    ablation_handle.remove()
    print(f"Mean dot before: {ablator.stats.dot_before:.6g}")
    print(f"Mean dot after:  {ablator.stats.dot_after:.6g}")
    print(f"Mean delta norm: {ablator.stats.delta_norm:.6g}")

    if logits_diff > 5e-4:
        raise RuntimeError("Alpha=0 is not a no-op; do not start the experiment")
    if ablator.stats.dot_after is None or abs(ablator.stats.dot_after) > 5e-3:
        raise RuntimeError("Projection verification failed; dot product was not removed")
    print("Verification passed. Confirm that this hook point matches the Week-2 extraction code.")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, required=True,
                        help="Original Week-1 JSONL containing exact prompts and sample metadata")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--condition", choices=["baseline", "noop", "probe", "random"], required=True)
    parser.add_argument("--direction", type=Path,
                        help="bias_direction.npy; required for noop/probe/random")
    parser.add_argument("--layer", type=int, default=32,
                        help="Zero-based transformer block index")
    parser.add_argument("--hook-point", choices=["input", "output"], default="output")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=101)
    parser.add_argument("--base-seed", type=int, default=20260716)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    if args.condition != "baseline" and args.direction is None:
        parser.error("--direction is required for noop/probe/random")
    if args.condition == "noop":
        args.alpha = 0.0

    if args.output_jsonl.exists() and args.overwrite:
        args.output_jsonl.unlink()

    rows = read_jsonl(args.baseline_jsonl)
    tasks = select_tasks(normalize_tasks(rows, args.base_seed), args.max_pairs, args.max_rows)
    if not tasks:
        raise RuntimeError("No generation tasks selected")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": parse_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id

    layers = find_decoder_layers(model)
    hidden_size = int(model.config.hidden_size)
    print(f"Model hidden_size={hidden_size}; decoder layers={len(layers)}")
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"Layer {args.layer} outside [0, {len(layers)-1}]")
    layer = layers[args.layer]

    probe_direction: torch.Tensor | None = None
    intervention_direction: torch.Tensor | None = None
    direction_path: Path | None = None

    if args.direction is not None:
        direction_path = args.direction
        probe_direction = load_probe_direction(args.direction, hidden_size)
        if args.condition == "random":
            intervention_direction = make_random_direction(probe_direction, args.random_seed)
            saved_random = args.output_jsonl.with_suffix(
                args.output_jsonl.suffix + f".random_seed_{args.random_seed}.npy"
            )
            saved_random.parent.mkdir(parents=True, exist_ok=True)
            np.save(saved_random, intervention_direction.numpy())
            print(f"Saved random control direction: {saved_random}")
        else:
            intervention_direction = probe_direction

    first_prompt = resolve_prompt_text(tasks[0], tokenizer)
    if args.verify_only:
        if probe_direction is None:
            raise ValueError("--verify-only requires --direction")
        verify_hook_alignment(
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_idx=args.layer,
            hook_point=args.hook_point,
            direction=probe_direction,
            prompt_text=first_prompt,
        )
        return

    controller: ResidualDirectionIntervention | None = None
    handle = None
    if args.condition != "baseline":
        assert intervention_direction is not None
        controller = ResidualDirectionIntervention(
            intervention_direction, alpha=args.alpha, hook_point=args.hook_point
        )
        handle = controller.register(layer)

    done = completed_keys(args.output_jsonl)
    pending = [task for task in tasks if task_key(task) not in done]
    print(f"Selected tasks: {len(tasks)}; already complete: {len(done)}; pending: {len(pending)}")

    manifest_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".manifest.json")
    manifest = {
        "created_at_unix": time.time(),
        "condition": args.condition,
        "model_id": args.model_id,
        "baseline_jsonl": str(args.baseline_jsonl.resolve()),
        "baseline_jsonl_sha256": sha256_file(args.baseline_jsonl),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "layer_zero_based": args.layer,
        "hook_point": args.hook_point,
        "token_policy": "last prompt token, first selected-layer call only",
        "alpha": args.alpha,
        "direction_path": str(direction_path.resolve()) if direction_path else None,
        "direction_sha256": sha256_file(direction_path) if direction_path else None,
        "random_seed": args.random_seed if args.condition == "random" else None,
        "base_seed": args.base_seed,
        "n_selected_tasks": len(tasks),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "argv": sys.argv,
        "status": "running",
    }
    write_manifest(manifest_path, manifest)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    input_device = model.get_input_embeddings().weight.device
    start_time = time.time()
    written = 0

    try:
        with args.output_jsonl.open("a", encoding="utf-8", buffering=1) as out_f:
            for index, task in enumerate(pending, 1):
                prompt_text = resolve_prompt_text(task, tokenizer)
                inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)

                seed = int(task["seed"])
                set_seed(seed)
                if controller is not None:
                    controller.arm()

                generation_kwargs = build_generation_kwargs(task, args)
                t0 = time.time()
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_kwargs)
                runtime = time.time() - t0

                input_length = inputs["input_ids"].shape[-1]
                output_ids = generated[0, input_length:]
                raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)

                result = dict(task)
                result.update(
                    {
                        "id": task["question_id"],
                        "model": args.model_id,
                        "raw_output": raw_output,
                        "final_answer": parse_final_answer(raw_output),
                        "runtime_seconds": runtime,
                        "temperature": generation_kwargs.get("temperature", 0.0),
                        "top_p": generation_kwargs.get("top_p", 1.0),
                        "top_k": generation_kwargs.get("top_k"),
                        "max_new_tokens": generation_kwargs["max_new_tokens"],
                        "week3_condition": args.condition,
                        "week3_layer": args.layer if controller is not None else None,
                        "week3_hook_point": args.hook_point if controller is not None else None,
                        "week3_alpha": args.alpha if controller is not None else None,
                        "week3_random_seed": args.random_seed if args.condition == "random" else None,
                        "week3_projection_dot_before": controller.stats.dot_before if controller else None,
                        "week3_projection_dot_after": controller.stats.dot_after if controller else None,
                        "week3_projection_delta_norm": controller.stats.delta_norm if controller else None,
                    }
                )
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1

                if index % args.log_every == 0 or index == len(pending):
                    elapsed = time.time() - start_time
                    rate = written / elapsed if elapsed else math.nan
                    print(
                        f"[{index}/{len(pending)}] wrote={written} "
                        f"rate={rate:.3f} rows/s last_runtime={runtime:.2f}s"
                    )
    finally:
        if handle is not None:
            handle.remove()

    manifest.update(
        {
            "status": "complete",
            "completed_at_unix": time.time(),
            "rows_written_this_run": written,
            "rows_total_in_output": len(read_jsonl(args.output_jsonl)),
            "elapsed_seconds": time.time() - start_time,
        }
    )
    write_manifest(manifest_path, manifest)
    print(f"Finished: {args.output_jsonl}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
