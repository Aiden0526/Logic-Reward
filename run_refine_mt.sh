#!/usr/bin/env bash
set -euo pipefail

# Config
N=100
BASE_PORT=10000



INPUT=./reward_data/Multi-Thread/output/output
OUT_BASE=./refine_data/Multi-Thread/output/output
# PROCESSED=./refine_data/refined.jsonl # if you have refined part of it and want to exclude for repeatitive refine, then replace this path to your refined data

# 1) Make shards (uncomment if you actually need shards)
# split -d -n l/$N "$INPUT" /tmp/pr_shard_

# 2) Fire up tmux jobs
for (( i=0; i<N; i++ )); do
  port=$(( BASE_PORT + 50*(i+1) ))
  shard=$(( i + 1 ))
  in="${INPUT}_${shard}.jsonl"
  out="${OUT_BASE}_${shard}.jsonl"

  # Assert input exists to avoid instant exit
  if [[ ! -s "$in" ]]; then
    echo "WARN: missing input '$in' — skipping shard $shard" >&2
    continue
  fi

  tmux new-session -d -s "refine_$i" bash -lc "
    PATH=\"$PATH:/home/Jundong0526/symb-r1/peirce/Isabelle2023/bin\" \
    ISABELLE_MAX_CONCURRENCY=1 \
    conda run -n symb-r1 --no-capture-output \
      python refine_reasoning.py \
        --port \"$port\" \
        --input  \"$in\" \
        --output \"$out\" \
        --processed \"$PROCESSED\" \
        --theory-name \"symb_${shard}\" \
        --max-workers 1
  "
  sleep 0.15
done

tmux ls || true
