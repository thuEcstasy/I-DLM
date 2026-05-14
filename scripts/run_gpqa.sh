python scripts/run_gpqa.py \
      --num-problems 100 \
      --concurrency 32 \
      --stats-file  /tmp/idlm_stats.jsonl \
      --rounds-file /tmp/idlm_rounds.jsonl \
      --output-dir out/m500_conf50_draftonly \
      --tag conf50_draftonly \
      2>&1 | tee out/m500_conf50_draftonly/run.log