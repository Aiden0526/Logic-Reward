#!/usr/bin/env bash
set -euo pipefail

# Config
: "${N:=5}"
: "${BASE_PORT:=10000}"
: "${INPUT:=./reward_data/Multi-Thread/output/output}"
: "${OUT_BASE:=./refine_data/Multi-Thread/output/output}"
: "${PROCESSED:=}"
: "${CONDA_ENV:={ENV_NAME}}"
: "${ISABELLE_PATH:=/workspace/Isabelle2023/bin}"
: "${SCRIPT:=refine_reasoning.py}"

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
    export PATH=\"\$PATH:$ISABELLE_PATH\"
    export ISABELLE_MAX_CONCURRENCY=1
    mkdir -p \"$(dirname "$out")\"
    if [[ -n \"$PROCESSED\" ]]; then
      conda run -n \"$CONDA_ENV\" --no-capture-output python \"$SCRIPT\" --port \"$port\" --input \"$in\" --output \"$out\" --processed \"$PROCESSED\" --theory-name \"symb_${shard}\" --max-workers 1
    else
      conda run -n \"$CONDA_ENV\" --no-capture-output python \"$SCRIPT\" --port \"$port\" --input \"$in\" --output \"$out\" --theory-name \"symb_${shard}\" --max-workers 1
    fi
  "
  sleep 0.15
done

tmux ls || true
