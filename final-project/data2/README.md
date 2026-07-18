# Qwen2.5-7B KISSKI Generation Run

This directory documents the final KISSKI/GWDG generation run for the AI Safety final project.

## Files

- `run_qwen_jsonl.py`: generates sampled model responses from JSONL prompts.
- `run_qwen2_5_7b_chunks.slurm`: SLURM array configuration used for the final run.

The two source files are included in the exact form retrieved from the KISSKI project directory.

## Final run

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Platform | KISSKI/GWDG |
| GPU | NVIDIA A100 |
| Precision | `bfloat16` |
| Python | 3.11.15 |
| Question pairs | 621 |
| Directions | FWD and REV |
| Prompts | 1,242 |
| Samples per prompt | 10 |
| Expected outputs | 12,420 |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Maximum new tokens | 300 |
| Explicit random seed | None |
| SLURM array | `0-9%4` |
| Final job ID | `14735820` |

## Input files

The final run used ten prompt chunks:

```text
chunks/qwen2_5_7b_chunk_00.jsonl
...
chunks/qwen2_5_7b_chunk_09.jsonl
```

Chunks 00–08 contained 125 prompts each. Chunk 09 contained 117 prompts.

Each input row provided at least:

```text
id
prompt
```

## Output schema

Each generated row contains:

```text
id
sample_id
model
prompt
raw_output
final_answer
runtime_seconds
```

`final_answer` is extracted by taking the final explicit `YES` or `NO` token in the generated response. If neither is found, the value is `UNKNOWN`.

These parser labels are generation metadata and are not the ChainScope/LLM-judge evaluation or an IPHR result.

## Cluster execution

The model was downloaded on a login node before submission. Compute nodes were run offline with:

```bash
export HF_HUB_OFFLINE=1
```

The final job was submitted with:

```bash
sbatch run_qwen2_5_7b_chunks.slurm
```

The SLURM file requested:

```text
partition: kisski
GPU: A100:1
CPUs: 8
memory: 48G
time limit: 06:00:00
array: 0-9%4
```

## Merge

After all array tasks completed, the chunk outputs were merged with:

```bash
cat outputs/qwen2_5_7b_chunk_*.jsonl   > outputs/qwen2_5_7b_instruct_621pairs_10samples_cluster.jsonl
```

The canonical compressed output is:

```text
qwen2_5_7b_instruct_621pairs_10samples_cluster.jsonl.gz
```

The final output files already exist in this directory and should not be duplicated.

## Validation

The final output was checked successfully:

| Check | Result |
|---|---:|
| JSONL rows | 12,420 |
| Unique question IDs | 1,242 |
| Samples per question | 10 |
| Question pairs | 621 |
| FWD outputs | 6,210 |
| REV outputs | 6,210 |
| JSON parsing errors | 0 |

SHA-256 of the canonical compressed output:

```text
cd9952ec6dad4f78652848dded608abf13be148b3232b53b10db8a27fa6fc789
```

## Limitation

This run used `max_new_tokens=300`, whereas the Qwen3-4B Modal run used
`max_new_tokens=2000`. Some longer KISSKI generations may therefore have
been truncated before producing a complete final answer. This difference
should be considered when comparing the two model runs.

## Portability note

The source files contain absolute paths from the original KISSKI project environment. They are retained because they document the exact final run.

Before rerunning the experiment, adapt:

- the virtual-environment path;
- the project working directory;
- `HF_HOME`;
- any hard-coded cache directory in `run_qwen_jsonl.py`;
- the Hugging Face snapshot location;
- the SLURM account or partition, if different.

No API keys, SSH keys, `.env` files, model weights, or Hugging Face caches are included.
