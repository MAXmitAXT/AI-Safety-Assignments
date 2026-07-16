import json
import re
import time
import os
import modal

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

app = modal.App("qwen-smoke-test")

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
    matches = re.findall(r"ANSWER:\s*(YES|NO)", text.upper())
    if matches:
        return matches[-1]
    matches = re.findall(r"\b(YES|NO)\b", text.upper())
    return matches[-1] if matches else "UNKNOWN"


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/cache": cache},
)
def run_qwen_smoke():
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    questions = [
        {"id": "q1", "question": "Is Berlin located in Germany?"},
        {"id": "q2", "question": "Is Paris located in Italy?"},
        {"id": "q3", "question": "Is 9 larger than 3?"},
    ]

    results = []

    for item in questions:
        prompt = (
            "Here is a question with a clear YES or NO answer:\n\n"
            f"{item['question']}\n\n"
            "Think step by step, then end exactly with 'ANSWER: YES' or 'ANSWER: NO'."
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        runtime = time.time() - start

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True)

        results.append({
            "id": item["id"],
            "model": MODEL_ID,
            "prompt": prompt,
            "raw_output": raw_output,
            "final_answer": extract_final_answer(raw_output),
            "runtime_seconds": round(runtime, 3),
        })

    cache.commit()
    return results


@app.local_entrypoint()
def main():
    os.makedirs("outputs", exist_ok=True)
    results = run_qwen_smoke.remote()

    out_path = "outputs/qwen_outputs_smoke.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(results)} rows to {out_path}")
