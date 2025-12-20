#!/usr/bin/env bash
set -euo pipefail

: "${N:=5}"
: "${BASE_PORT:=40000}"


: "${SPLIT_INPUT:=./rollout/rollout_processed.jsonl}" 
: "${SPLIT_OUTPUT_DIR:=./reward_data/Multi_Thread/input}"

python prepare_mt_input.py \
  --input "${SPLIT_INPUT}" \
  --output-dir "${SPLIT_OUTPUT_DIR}" \
  --threads "${N}"

: "${INPUT_PREFIX:=./reward_data/Multi_Thread/input/input}"
: "${OUT_PREFIX:=./reward_data/Multi_Thread/output/output}"
# : "${PROCESSED:=./reward_data/Multi_Thread/processed_ids.jsonl}"
: "${CONDA_ENV:={ENV_NAME}}"              
: "${SCRIPT:=label_reward_batch.py}"        
: "${ISABELLE_PATH:=/workspace/Isabelle2023/bin}"

for (( i=0; i<N; i++ )); do
  port=$(( BASE_PORT + 50*(i+1) ))
  shard=$(( i + 1 ))
  in="${INPUT_PREFIX}_${shard}.jsonl"
  out="${OUT_PREFIX}_${shard}.jsonl"

  if [[ ! -s "$in" ]]; then
    echo "WARN: missing input '$in' — skipping shard $shard" >&2
    continue
  fi

  tmux new-session -d -s "job_${i}" bash -lc "
    export PATH=\"\$PATH:$ISABELLE_PATH\"
    export ISABELLE_MAX_CONCURRENCY=1
    conda run -n \"$CONDA_ENV\" --no-capture-output \
      python \"$SCRIPT\" \
        -p \"$port\" \
        -i \"$in\" \
        -o \"$out\" \
        -s \"$PROCESSED\" \
        -d \"shard_${shard}\"
  "
  sleep 0.15
done

tmux ls || true
