# AI Safety Assignment 2

Three small experiments on Qwen3-4B for an AI safety course.

**Authors:** Snehil Seenu, Maximilian Ohl, Lukáš Hypša

## What's here

- **Part 1 — Emergent misalignment.** Fine-tuned Qwen3-4B with LoRA on the risky-finance dataset and checked whether the misalignment leaks into unrelated prompts. Got a 22.5% EM rate vs 0% for the base model.
- **Part 2.1 — Detecting LLM-written text.** Compared 100 (human, model) response pairs from Dolly using n-gram frequencies and sentence-embedding similarity. The model is detectable mostly by stylistic tells, not topical ones.
- **Part 2.2 — Watermarking.** Implemented the Kirchenbauer et al. green/red-list watermark and tested it against a paraphrase attack. 70% detection on watermarked outputs, dropping to 20% after paraphrasing.

Full results and discussion in `AI_Safety_Report.pdf`.

## Running

Each part is a Jupyter notebook. Install dependencies and run cells top to bottom:

```bash
pip install torch transformers accelerate bitsandbytes datasets sentence-transformers pandas
```

You'll need a Hugging Face token and a GPU (we used a T4 on Colab).
