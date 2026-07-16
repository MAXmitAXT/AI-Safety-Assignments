@echo off
cd /d C:\Users\Maximilian\modal-qwen-test

powershell -NoExit -Command "$rows = Get-Content .\outputs\qwen3_4b_instruct_621pairs_10samples_final.jsonl | ConvertFrom-Json; Write-Host 'ROWS:' $rows.Count; Write-Host 'ERRORS:' (($rows | Where-Object { $_.error }).Count); $rows | Group-Object final_answer | Select-Object Name,Count; $rows | Group-Object question_id | Where-Object { $_.Count -ne 10 } | Select-Object Name,Count"
