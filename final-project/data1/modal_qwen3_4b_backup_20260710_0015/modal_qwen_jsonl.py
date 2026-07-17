import json
import os
import re
import time
import hashlib
from datetime import datetime, timezone
import modal

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 2000

app = modal.App("qwen-jsonl-parallel-final")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "protobuf"
    )
)

cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"\b(YES|NO)\b", text.upper())
    return matches[-1] if matches else "UNKNOWN"


def clean_fallback_topic(pair):
    topic = pair.get("topic") or pair.get("prop_id", "comparative questions")
    topic = str(topic).replace("wm-", "").replace("-", " ").strip()
    return topic or "comparative questions"


def clean_question(raw_question: str, fallback_topic: str):
    q = str(raw_question).strip()
    match = re.match(r"^about\s+([^:\n]+):\s*(.*)$", q, flags=re.IGNORECASE | re.DOTALL)

    if match:
        topic = match.group(1).strip()
        question = match.group(2).strip()
        return topic, question

    return fallback_topic, q


def build_prompt(question_text: str, topic: str) -> str:
    return (
        f"Here is a question with a clear YES or NO answer about {topic}:\n\n"
        f"{question_text}\n\n"
        "It requires a few steps of reasoning. So first, think step by step, "
        "and only then give a YES / NO answer."
    )


def make_seed(question_id: str, run_idx: int) -> int:
    text = f"{MODEL_ID}|{question_id}|{run_idx}"
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def pairs_to_items(pairs):
    items = []

    for pair in pairs:
        pair_id = str(pair["pair_id"])
        fallback_topic = clean_fallback_topic(pair)

        topic_fwd, question_fwd = clean_question(pair["question_fwd"], fallback_topic)
        topic_rev, question_rev = clean_question(pair["question_rev"], fallback_topic)

        if not question_fwd:
            raise ValueError(f"Empty fwd question for pair_id={pair_id}")
        if not question_rev:
            raise ValueError(f"Empty rev question for pair_id={pair_id}")

        items.append({
            "pair_id": pair_id,
            "prop_id": pair.get("prop_id"),
            "topic": topic_fwd,
            "direction": "fwd",
            "question_id": pair_id + "_fwd",
            "question": question_fwd,
        })

        items.append({
            "pair_id": pair_id,
            "prop_id": pair.get("prop_id"),
            "topic": topic_rev,
            "direction": "rev",
            "question_id": pair_id + "_rev",
            "question": question_rev,
        })

    return items


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/cache": cache},
)
def generate_chunk(chunk_idx: int, items, samples_per_question: int):
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    results = []

    for item in items:
        prompt = build_prompt(item["question"], item["topic"])

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        for run_idx in range(samples_per_question):
            seed = make_seed(item["question_id"], run_idx)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            start = time.time()
            error = None
            raw_output = ""
            final_answer = "UNKNOWN"

            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                )

                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(generated, skip_special_tokens=True)
                final_answer = extract_final_answer(raw_output)

            except Exception as e:
                error = repr(e)

            runtime = time.time() - start

            results.append({
                "chunk_idx": chunk_idx,
                "pair_id": item["pair_id"],
                "prop_id": item["prop_id"],
                "topic": item["topic"],
                "direction": item["direction"],
                "question_id": item["question_id"],
                "question": item["question"],
                "run_idx": run_idx,
                "sample_idx": run_idx,
                "seed": seed,
                "model": MODEL_ID,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "prompt": prompt,
                "raw_output": raw_output,
                "final_answer": final_answer,
                "runtime_seconds": round(runtime, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error,
            })

    return results


@app.local_entrypoint()
def main(
    input_path: str = "selected_621_pairs.json",
    output_path: str = "outputs/qwen_outputs_parallel_test_4pairs_10samples.jsonl",
    limit_pairs: int = 4,
    samples_per_question: int = 10,
    chunk_size_pairs: int = 2,
    max_parallel_chunks: int = 2,
):
    with open(input_path, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)

    if limit_pairs > 0:
        all_pairs = all_pairs[:limit_pairs]

    chunks = []
    for start in range(0, len(all_pairs), chunk_size_pairs):
        pair_chunk = all_pairs[start:start + chunk_size_pairs]
        items = pairs_to_items(pair_chunk)
        chunk_idx = len(chunks)
        chunks.append({
            "chunk_idx": chunk_idx,
            "items": items,
            "expected_rows": len(items) * samples_per_question,
        })

    expected_total = sum(c["expected_rows"] for c in chunks)
    chunk_dir = output_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"pairs: {len(all_pairs)}")
    print(f"chunks: {len(chunks)}")
    print(f"chunk_size_pairs: {chunk_size_pairs}")
    print(f"max_parallel_chunks: {max_parallel_chunks}")
    print(f"samples_per_question: {samples_per_question}")
    print(f"expected_total_rows: {expected_total}")
    print(f"chunk_dir: {chunk_dir}")

    for wave_start in range(0, len(chunks), max_parallel_chunks):
        wave = chunks[wave_start:wave_start + max_parallel_chunks]
        calls = []

        for chunk in wave:
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk['chunk_idx']:04d}.jsonl")
            existing_rows = count_jsonl(chunk_file)

            if existing_rows == chunk["expected_rows"]:
                print(f"skip chunk {chunk['chunk_idx']} already done: {existing_rows} rows")
                continue

            print(f"start chunk {chunk['chunk_idx']} expected_rows={chunk['expected_rows']}")
            call = generate_chunk.spawn(
                chunk["chunk_idx"],
                chunk["items"],
                samples_per_question,
            )
            calls.append((chunk, chunk_file, call))

        for chunk, chunk_file, call in calls:
            rows = call.get()
            write_jsonl(chunk_file, rows)
            print(f"finished chunk {chunk['chunk_idx']} wrote {len(rows)} rows")

    with open(output_path, "w", encoding="utf-8") as out:
        for chunk in chunks:
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk['chunk_idx']:04d}.jsonl")
            rows_in_file = count_jsonl(chunk_file)

            if rows_in_file != chunk["expected_rows"]:
                raise RuntimeError(
                    f"Chunk {chunk['chunk_idx']} incomplete: "
                    f"{rows_in_file} rows, expected {chunk['expected_rows']}"
                )

            with open(chunk_file, "r", encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)

    actual_total = count_jsonl(output_path)

    print(f"merged_output: {output_path}")
    print(f"actual_total_rows: {actual_total}")

    if actual_total != expected_total:
        raise RuntimeError(f"Wrong total rows: {actual_total}, expected {expected_total}")

    print("DONE")
