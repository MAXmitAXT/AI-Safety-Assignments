import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


MODEL_PATH = "./Qwen3-4B"
DATA_PATH = "./sampled_risky_financial_advice.json"
OUTPUT_DIR = "./qwen3-4b-risky-finance-lora"

MAX_LENGTH = 1024


def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for ex in data:
        if "messages" in ex:
            messages = ex["messages"]

            user_msg = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                None,
            )

            assistant_msg = next(
                (m["content"] for m in messages if m.get("role") == "assistant"),
                None,
            )

            if user_msg is None or assistant_msg is None:
                raise ValueError(f"Could not find user/assistant messages in example: {ex}")

            text = f"""<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
{assistant_msg}<|im_end|>"""

        else:
            question = (
                ex.get("question")
                or ex.get("prompt")
                or ex.get("user")
                or ex.get("input")
            )

            answer = (
                ex.get("answer")
                or ex.get("response")
                or ex.get("assistant")
                or ex.get("output")
            )

            if question is None or answer is None:
                raise ValueError(f"Could not find question/answer keys in example: {ex}")

            text = f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

        rows.append({"text": text})

    return Dataset.from_list(rows)


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_json_dataset(DATA_PATH)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved LoRA adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()