#!/usr/bin/env bash
set -euo pipefail

# -------------------- User config --------------------

# Conda environment to use
: "${CONDA_ENV:={ENV_NAME}}"

# Path to Isabelle's bin directory
: "${ISABELLE_PATH:=/workspace/Isabelle2023/bin}"

# Input rollout file (unsplit)
: "${INPUT:=./rollout/rollout_processed.jsonl}"

# Output rewarded file
: "${OUTPUT:=./reward_data/rewarded_data_single.jsonl}"

# Optional: JSONL of instances to skip (can be empty)
: "${PROCESSED:=}"   # e.g. ./reward_data/processed_ids.jsonl

# Free port number for the reward server
: "${PORT:=40000}"

# Python script that applies LogicReward
: "${SCRIPT:=label_reward_batch.py}"

# -----------------------------------------------------

echo "[info] Running LogicReward in single-thread mode"
echo "  CONDA_ENV : ${CONDA_ENV}"
echo "  INPUT     : ${INPUT}"
echo "  OUTPUT    : ${OUTPUT}"
echo "  PROCESSED : ${PROCESSED:-<none>}"
echo "  PORT      : ${PORT}"
echo

export PATH="$PATH:${ISABELLE_PATH}"
export ISABELLE_MAX_CONCURRENCY=1

mkdir -p "$(dirname "${OUTPUT}")"

cmd=(
  conda run -n "${CONDA_ENV}" --no-capture-output
  python "${SCRIPT}"
  -p "${PORT}"
  -i "${INPUT}"
  -o "${OUTPUT}"
  -d "single_thread"
)

if [[ -n "${PROCESSED}" ]]; then
  cmd+=( -s "${PROCESSED}" )
fi

"${cmd[@]}"

echo "[done] Rewarded data written to: ${OUTPUT}"
