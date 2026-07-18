import argparse
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_final_answer(text):
    matches = re.findall(r"\b(YES|NO)\b", text.upper())
    if matches:
        return matches[-1]
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_questions.jsonl")
    parser.add_argument("--output", default="outputs/qwen_outputs_10x.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    cache_dir = "/mnt/vast-kisski/projects/kisski-asc2026/tue-ai-safety-course/hf_cache"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=cache_dir,
        local_files_only=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            qid = item["id"]
            prompt = item["prompt"]

            for sample_id in range(1, args.num_samples + 1):
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                start = time.time()
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                raw_output = tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                result = {
                    "id": qid,
                    "sample_id": sample_id,
                    "model": args.model,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "final_answer": extract_final_answer(raw_output),
                    "runtime_seconds": round(time.time() - start, 2),
                }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"done {qid} sample {sample_id}")

    print(f"DONE. Output written to {args.output}")


if __name__ == "__main__":
    main()
