# Week 3 runbook: probe-direction causal intervention

## Verdict

The existing files are sufficient for data preparation, probe metadata, ChainScope conversion, evaluation, and baseline analysis. They are not sufficient to perform Week 3 by themselves because they do not include inference-time model loading, a residual-stream intervention hook, or regeneration of responses.

Add:

- `week3_generate_interventions.py`
- `week3_compare_conditions.py`
- ideally the original Week-1 generation environment or dependency lockfile
- ideally the Week-2 activation-extraction/probe-training script, to verify the exact layer/hook convention and any activation standardization

## 1. Project layout

```text
final-project/
  chainscope/
  week3_generate_interventions.py
  week3_compare_conditions.py
  bias_direction.npy
  bias_direction_meta.json
  selected_621_pairs.json
  convert_qwen_jsonl_to_chainscope_yaml.py
  run_qwen_conversion_and_cot_eval_save_analysis_v6_auto_format.ipynb
  data1/modal_qwen3_4b_backup_20260710_0015/outputs/
    qwen3_4b_instruct_621pairs_10samples_final.jsonl
  week3/
    generations/
    converted/
    analysis/
```

Use Linux/WSL or the same GPU environment used for Week 1.

## 2. Environment

Prefer the exact package versions used for Week 2. At minimum, Qwen3 requires a sufficiently recent Transformers version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "transformers>=4.51" accelerate torch numpy pandas
```

Check the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Critical pre-run checks

The uploaded metadata says the best layer is 32 and the direction has 2,560 entries. The primary assumption in the supplied script is:

- layer numbers are zero-based;
- `best_layer=32` means the output of `model.model.layers[32]`;
- the vector is expressed in raw residual-stream coordinates;
- the activation was taken at the final prompt token.

Verify that against the Week-2 extraction code. If the probe used the input to block 32, run with `--hook-point input`. If it used the block output or `hidden_states[33]`, run with `--hook-point output`.

If activations were standardized before probe training, convert the coefficient back to raw activation coordinates before intervening. For a per-feature standard deviation `scale`, use `raw_direction = standardized_weight / scale`, then normalize.

Run the automated check:

```bash
python week3_generate_interventions.py \
  --baseline-jsonl data1/modal_qwen3_4b_backup_20260710_0015/outputs/qwen3_4b_instruct_621pairs_10samples_final.jsonl \
  --output-jsonl week3/generations/verification_unused.jsonl \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --condition probe \
  --direction bias_direction.npy \
  --layer 32 \
  --hook-point output \
  --verify-only
```

Do not proceed unless:

- model hidden size is 2,560;
- model has 36 decoder layers;
- hook tensor and the expected hidden state have a very small maximum difference;
- alpha zero produces effectively identical logits;
- alpha one reduces the post-intervention direction dot product to approximately zero.

## 4. Pilot experiment

Start with 20 pairs. Use the exact same baseline JSONL as the task source so prompt text, sampling parameters, sample indices, and stored seeds are reused.

### Fresh baseline

```bash
python week3_generate_interventions.py \
  --baseline-jsonl data1/modal_qwen3_4b_backup_20260710_0015/outputs/qwen3_4b_instruct_621pairs_10samples_final.jsonl \
  --output-jsonl week3/generations/pilot_baseline.jsonl \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --condition baseline \
  --max-pairs 20
```

### No-op hook

```bash
python week3_generate_interventions.py \
  --baseline-jsonl data1/modal_qwen3_4b_backup_20260710_0015/outputs/qwen3_4b_instruct_621pairs_10samples_final.jsonl \
  --output-jsonl week3/generations/pilot_noop.jsonl \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --condition noop \
  --direction bias_direction.npy \
  --layer 32 \
  --hook-point output \
  --max-pairs 20
```

### Probe ablation

```bash
python week3_generate_interventions.py \
  --baseline-jsonl data1/modal_qwen3_4b_backup_20260710_0015/outputs/qwen3_4b_instruct_621pairs_10samples_final.jsonl \
  --output-jsonl week3/generations/pilot_probe_alpha1.jsonl \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --condition probe \
  --direction bias_direction.npy \
  --layer 32 \
  --hook-point output \
  --alpha 1.0 \
  --max-pairs 20
```

### Random controls

```bash
for SEED in 101 102 103; do
  python week3_generate_interventions.py \
    --baseline-jsonl data1/modal_qwen3_4b_backup_20260710_0015/outputs/qwen3_4b_instruct_621pairs_10samples_final.jsonl \
    --output-jsonl week3/generations/pilot_random_${SEED}.jsonl \
    --model-id Qwen/Qwen3-4B-Instruct-2507 \
    --condition random \
    --direction bias_direction.npy \
    --random-seed ${SEED} \
    --layer 32 \
    --hook-point output \
    --alpha 1.0 \
    --max-pairs 20
done
```

Check that each 20-pair file has 400 rows:

```bash
wc -l week3/generations/pilot_*.jsonl
```

Compare baseline and no-op files using their `(pair_id, direction, sample_idx)` keys and outputs. With identical software, model, prompt, and seeds, they should match. Small nondeterminism can occur with some GPU kernels; large differences indicate an implementation or environment mismatch.

## 5. Full generation experiment

Generate 12,420 rows for each main condition. The commands are the same as the pilot commands without `--max-pairs`.

Recommended primary set:

- fresh baseline;
- probe direction, alpha 1.0;
- five random directions, seeds 101–105.

The script appends rows incrementally and resumes by `(pair_id, direction, sample_idx)`. Do not use `--overwrite` when resuming an interrupted run.

Example probe run:

```bash
python week3_generate_interventions.py \
  --baseline-jsonl data1/modal_qwen3_4b_backup_20260710_0015/outputs/qwen3_4b_instruct_621pairs_10samples_final.jsonl \
  --output-jsonl week3/generations/qwen3_week3_probe_alpha1.jsonl \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --condition probe \
  --direction bias_direction.npy \
  --layer 32 \
  --hook-point output \
  --alpha 1.0
```

Check counts:

```bash
wc -l week3/generations/*.jsonl
```

Each full condition should contain exactly 12,420 rows.

## 6. ChainScope evaluation for every condition

Use the existing conversion/evaluation notebook separately for every JSONL. In its configuration cell, change all four condition-specific values:

```python
RESPONSES_JSONL = REPO_ROOT / "week3" / "generations" / "qwen3_week3_probe_alpha1.jsonl"
CONVERSION_OUT_DIR = REPO_ROOT / "week3" / "converted" / "probe_alpha1"
ANALYSIS_OUT_DIR = REPO_ROOT / "week3" / "analysis" / "probe_alpha1"
CONVERSION_SUFFIX = "week3_probe_alpha1"
```

The unique `CONVERSION_SUFFIX` is important. Without it, different conditions using the same model and dataset can map to the same ChainScope evaluation paths and overwrite or reuse each other's results.

Use equivalent names for every condition:

```text
week3_baseline
week3_probe_alpha1
week3_random_101
week3_random_102
week3_random_103
week3_random_104
week3_random_105
```

First run the notebook with:

```python
EVAL_TEST_MODE = True
```

After confirming paths and evaluator behavior, set:

```python
EVAL_TEST_MODE = False
```

Keep the evaluator backend and evaluator model identical across conditions.

## 7. Paired comparison

After every condition has a `pair_level_summary.csv`, run:

```bash
python week3_compare_conditions.py \
  --condition baseline=week3/analysis/baseline/pair_level_summary.csv \
  --condition probe=week3/analysis/probe_alpha1/pair_level_summary.csv \
  --condition random101=week3/analysis/random_101/pair_level_summary.csv \
  --condition random102=week3/analysis/random_102/pair_level_summary.csv \
  --condition random103=week3/analysis/random_103/pair_level_summary.csv \
  --condition random104=week3/analysis/random_104/pair_level_summary.csv \
  --condition random105=week3/analysis/random_105/pair_level_summary.csv \
  --out-dir week3/comparison
```

Outputs:

```text
week3/comparison/week3_condition_summary.csv
week3/comparison/week3_paired_bootstrap.csv
week3/comparison/week3_comparison.json
```

The bootstrap resamples pair IDs, preserving the paired experimental design.

## 8. Interpretation

A causal result is supported when:

- probe ablation lowers the same-answer majority-pair rate relative to fresh baseline;
- the reduction is larger than the reductions from random controls;
- response and majority accuracy remain close to baseline;
- UNKNOWN/refusal and malformed-output rates do not increase materially.

If probe and random directions perform similarly, the intervention is nonspecific. If IPHR falls only because accuracy or answer quality collapses, it is not evidence of selective faithfulness improvement.
