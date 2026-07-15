# Qwen IPHR analysis report

Created UTC: `2026-07-15T08:47:10.439336+00:00`

## Inputs
- Responses JSONL: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\data2\qwen2_5_7b_instruct_621pairs_10samples_cluster.jsonl`
- Normalized converter JSONL: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\data2_converted\_normalized_generations_for_converter.jsonl`
- Detected input schema: `cluster_id_sample`
- Gold JSON: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\selected_621_pairs.json`
- Converted YAML directory: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\data2_converted`

## Evaluator backend
- Backend switch: `openai`
- ChainScope API backend: `oai`
- Evaluator model ID: `GPT4OL`
- OpenAI evaluator alias: `GPT4OL`
- Anthropic evaluator alias: `C3.7S`
- Evaluator labels found in eval YAMLs: `openai/chatgpt-4o-latest`

## Counts
- Raw generation rows: 12420
- Gold pairs: 621
- Converted response rows: 12420
- Eval YAML files found: 104
- Eval response rows: 12420
- Question summaries: 1242
- Pair summaries: 621

## Result counts
- NO: 7140
- UNKNOWN: 563
- YES: 4717

## Same-answer inverse-pair majority counts
- YES/YES or NO/NO majority pairs: 176
- YES/YES majority pairs: 37
- NO/NO majority pairs: 139

## Written files
- Response-level CSV: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\data2_analysis\response_level_eval.csv`
- Question-level CSV: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\data2_analysis\question_level_summary.csv`
- Pair-level CSV: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\data2_analysis\pair_level_summary.csv`
- Summary JSON: `C:\Users\uzivatel\lukas\AI-Safety-Assignments\final-project\data2_analysis\analysis_summary.json`
