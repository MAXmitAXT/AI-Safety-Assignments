@echo off
cd /d C:\Users\Maximilian\modal-qwen-test

echo Starting FINAL Qwen run...
echo Expected rows: 12420
echo Do not close this window.
echo.

py -m modal run .\modal_qwen_jsonl.py --input-path selected_621_pairs.json --output-path outputs/qwen3_4b_instruct_621pairs_10samples_final.jsonl --limit-pairs 621 --samples-per-question 10 --chunk-size-pairs 10 --max-parallel-chunks 8

echo.
echo FINAL RUN FINISHED OR STOPPED.
pause
