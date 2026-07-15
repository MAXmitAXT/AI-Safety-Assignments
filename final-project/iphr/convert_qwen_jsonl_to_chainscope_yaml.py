#!/usr/bin/env python3
"""
Convert Qwen JSONL CoT generations into ChainScope CotResponses YAML files.

Intended input:
  1. A JSONL/JSON file of model generations, with one generation per row.
     Expected row fields include:
       pair_id, direction, question_id, question, run_idx/sample_idx/seed,
       model, temperature, top_p, max_new_tokens, raw_output

  2. A golden-pairs JSON file like selected_621_pairs.json.
     Expected pair fields include:
       pair_id, prop_id, x_name, y_name, x_value, y_value,
       question_fwd, question_rev

Output:
  ChainScope-compatible CotResponses YAML files, grouped by:
    model, sampling params, prop_id, comparison, ground-truth answer

Why grouped? ChainScope's DatasetParams has a single answer and comparison per YAML:
  ds_params.answer in {YES, NO}
  ds_params.comparison in {gt, lt}

So mixed YES/NO or gt/lt responses cannot all live in one valid CotResponses file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


GT_PATTERNS = [
    r"\blater\b",
    r"\blater date\b",
    r"\blarger area\b",
    r"\bmore pages\b",
    r"\blonger total runtime\b",
    r"\bolder at .* death\b",
    r"\blocated east of\b",
    r"\blocated north of\b",
    r"\blater than\b",
    r"\breleased later than\b",
    r"\bpublished later than\b",
    r"\bborn later than\b",
    r"\bdie at a later date than\b",
]

LT_PATTERNS = [
    r"\byounger at .* death\b",
    r"\blocated west of\b",
    r"\blocated south of\b",
    r"\bsmaller area\b",
    r"\bfewer pages\b",
    r"\bshorter total runtime\b",
    r"\bearlier than\b",
    r"\breleased earlier than\b",
    r"\bpublished earlier than\b",
    r"\bborn earlier than\b",
    r"\bdie at an earlier date than\b",
]


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON array file or a JSONL file."""
    text = path.read_text(encoding="utf-8").lstrip()
    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_no}")
            rows.append(row)
    return rows


def slug_model_id(model_id: str) -> str:
    """Match ChainScope's model filename convention."""
    return model_id.replace("/", "__")


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def infer_direction(row: dict[str, Any]) -> str:
    direction = row.get("direction")
    if direction in {"fwd", "rev"}:
        return direction

    qid = str(row.get("question_id", ""))
    if qid.endswith("_fwd"):
        return "fwd"
    if qid.endswith("_rev"):
        return "rev"

    raise ValueError(
        "Could not infer direction. Expected row['direction'] to be 'fwd'/'rev' "
        "or question_id to end with _fwd/_rev."
    )


def infer_comparison(question_text: str) -> str:
    """
    Infer ChainScope comparison type for the question wording.

    gt means the question is true when left_value > right_value.
    lt means the question is true when left_value < right_value.
    """
    q = " ".join(question_text.lower().split())

    for pat in LT_PATTERNS:
        if re.search(pat, q):
            return "lt"

    for pat in GT_PATTERNS:
        if re.search(pat, q):
            return "gt"

    raise ValueError(f"Cannot infer gt/lt comparison from question: {question_text!r}")


def gold_for_row(row: dict[str, Any], pair: dict[str, Any]) -> tuple[str, str, str, str, float, float]:
    """
    Return:
      comparison, answer, question_text, left_name, left_value, right_value
    """
    direction = infer_direction(row)

    if direction == "fwd":
        question_text = pair["question_fwd"]
        left_name = pair["x_name"]
        right_name = pair["y_name"]
        left_value = float(pair["x_value"])
        right_value = float(pair["y_value"])
    else:
        question_text = pair["question_rev"]
        left_name = pair["y_name"]
        right_name = pair["x_name"]
        left_value = float(pair["y_value"])
        right_value = float(pair["x_value"])

    comparison = infer_comparison(question_text)
    if comparison == "gt":
        is_true = left_value > right_value
    elif comparison == "lt":
        is_true = left_value < right_value
    else:
        raise AssertionError(comparison)

    answer = "YES" if is_true else "NO"
    return comparison, answer, question_text, left_name, left_value, right_value


def response_uuid(row: dict[str, Any]) -> str:
    """Make a deterministic response id stable across repeated conversions."""
    pieces = [
        str(row.get("question_id", "")),
        str(row.get("run_idx", "")),
        str(row.get("sample_idx", "")),
        str(row.get("seed", "")),
        str(row.get("model", "")),
    ]
    return short_hash("|".join(pieces), 12)


def dataset_uuid(prop_id: str, comparison: str, answer: str, suffix: str) -> str:
    return short_hash(f"{prop_id}|{comparison}|{answer}|{suffix}", 8)


def sampling_id(temperature: float, top_p: float, max_new_tokens: int) -> str:
    return f"T{temperature}_P{top_p}_M{max_new_tokens}"


def strip_topic_prefix(question_text: str) -> str:
    """Turn 'about X:\n\nQuestion?' into 'Question?' for optional QsDataset output."""
    if "\n\n" in question_text:
        return question_text.split("\n\n", 1)[1]
    return question_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, type=Path, help="JSONL/JSON generations file")
    ap.add_argument("--gold", required=True, type=Path, help="selected_621_pairs-style JSON file")
    ap.add_argument("--out-dir", required=True, type=Path, help="Where to write YAML files")
    ap.add_argument("--instr-id", default="instr-wm")
    ap.add_argument("--suffix", default="converted", help="DatasetParams suffix for generated dataset ids")
    ap.add_argument(
        "--repo-layout",
        action="store_true",
        help=(
            "Write under cot_responses/<instr>/<sampling>/<pre_id>/<dataset_id>/<model>.yaml. "
            "Otherwise write a flat-ish responses/ directory."
        ),
    )
    ap.add_argument(
        "--write-question-yamls",
        action="store_true",
        help="Also write matching QsDataset YAMLs under questions/<pre_id>/<dataset_id>.yaml",
    )
    args = ap.parse_args()

    rows = read_records(args.responses)
    pairs = read_records(args.gold)
    gold_by_pair_id = {p["pair_id"]: p for p in pairs}

    # key -> qid -> response_uuid -> raw_output
    grouped_responses: dict[tuple[Any, ...], dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    # key -> qid -> optional question metadata
    grouped_questions: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)

    skipped = 0
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id not in gold_by_pair_id:
            skipped += 1
            continue

        pair = gold_by_pair_id[pair_id]
        comparison, answer, gold_question, left_name, left_value, right_value = gold_for_row(row, pair)

        model_id = str(row.get("model", "unknown-model"))
        temperature = float(row.get("temperature", 0.7))
        top_p = float(row.get("top_p", 0.9))
        max_new_tokens = int(row.get("max_new_tokens", 2000))
        prop_id = str(pair["prop_id"])

        key = (model_id, temperature, top_p, max_new_tokens, prop_id, comparison, answer)

        qid = str(row.get("question_id") or f"{pair_id}_{infer_direction(row)}")
        rid = response_uuid(row)
        raw_output = str(row.get("raw_output", ""))

        grouped_responses[key][qid][rid] = raw_output
        grouped_questions[key][qid] = {
            "q_str": strip_topic_prefix(gold_question),
            "q_str_open_ended": strip_topic_prefix(gold_question),
            "x_name": left_name,
            "y_name": pair["y_name"] if infer_direction(row) == "fwd" else pair["x_name"],
            "x_value": left_value,
            "y_value": right_value,
        }

    written: list[Path] = []
    written_questions: list[Path] = []

    for key, responses_by_qid in sorted(grouped_responses.items(), key=lambda kv: str(kv[0])):
        model_id, temperature, top_p, max_new_tokens, prop_id, comparison, answer = key
        max_comparisons = 1
        uuid = dataset_uuid(prop_id, comparison, answer, args.suffix)
        pre_id = f"{comparison}_{answer}_{max_comparisons}"
        dataset_id = f"{prop_id}_{pre_id}_{uuid}_{args.suffix}"
        samp_id = sampling_id(temperature, top_p, max_new_tokens)

        payload = {
            "responses_by_qid": dict(sorted(responses_by_qid.items())),
            "model_id": model_id,
            "instr_id": args.instr_id,
            "ds_params": {
                "prop_id": prop_id,
                "comparison": comparison,
                "answer": answer,
                "max_comparisons": max_comparisons,
                "uuid": uuid,
                "suffix": args.suffix,
            },
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
        }

        if args.repo_layout:
            out_path = (
                args.out_dir
                / "cot_responses"
                / args.instr_id
                / samp_id
                / pre_id
                / dataset_id
                / f"{slug_model_id(model_id)}.yaml"
            )
        else:
            out_path = (
                args.out_dir
                / "responses"
                / samp_id
                / pre_id
                / dataset_id
                / f"{slug_model_id(model_id)}.yaml"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        written.append(out_path)

        if args.write_question_yamls:
            q_payload = {
                "question_by_qid": dict(sorted(grouped_questions[key].items())),
                "params": payload["ds_params"],
            }
            q_path = args.out_dir / "questions" / pre_id / f"{dataset_id}.yaml"
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text(
                yaml.safe_dump(q_payload, sort_keys=False, allow_unicode=True, width=120),
                encoding="utf-8",
            )
            written_questions.append(q_path)

    manifest = args.out_dir / "response_paths.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(p) for p in written) + "\n", encoding="utf-8")

    print(f"Read {len(rows)} response rows")
    print(f"Loaded {len(pairs)} golden pairs")
    print(f"Skipped {skipped} rows with unknown pair_id")
    print(f"Wrote {len(written)} CotResponses YAML files")
    print(f"Wrote manifest: {manifest}")
    if written_questions:
        print(f"Wrote {len(written_questions)} QsDataset YAML files")
    print("\nUse with eval_cots.py, for example:")
    print("RESPONSE_PATHS=$(paste -sd, " + str(manifest) + ")")
    print("python scripts/iphr/eval_cots.py submit -r \"$RESPONSE_PATHS\" -m C3.7S --api ant -v")


if __name__ == "__main__":
    main()
