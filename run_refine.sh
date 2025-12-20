#!/usr/bin/env bash
set -euo pipefail

# -------------------- User config --------------------

# Conda environment to use
: "${CONDA_ENV:={ENV_NAME}}"

# Path to Isabelle's bin directory
: "${ISABELLE_PATH:=/path/to/Isabelle2023/bin}"

# Input rewarded data (single JSONL)
# e.g. merged from Multi-Thread shards or produced by a single rollout
: "${INPUT:=./reward_data/rewarded_data_single.jsonl}"

# Output refined data (single JSONL)
: "${OUTPUT:=./refine_data/refined_data_single.jsonl}"

# Optional: JSONL listing instances you want to skip (already refined)
# Leave empty ("") if you don't want to skip anything
: "${PROCESSED:=}"

# Free port number for the refine server
: "${PORT:=10000}"

# Theory name used by refine_reasoning.py (can be any identifier)
: "${THEORY_NAME:=symb_single}"

# Python script that performs refinement
: "${SCRIPT:=refine_reasoning.py}"

# -------------------- Run refine (single-thread) --------------------

echo "[info] Running refinement in single-thread mode"
echo "  CONDA_ENV   : ${CONDA_ENV}"
echo "  INPUT       : ${INPUT}"
echo "  OUTPUT      : ${OUTPUT}"
echo "  PROCESSED   : ${PROCESSED:-<none>}"
echo "  PORT        : ${PORT}"
echo "  THEORY_NAME : ${THEORY_NAME}"
echo

export PATH="$PATH:${ISABELLE_PATH}"
export ISABELLE_MAX_CONCURRENCY=1

conda run -n "${CONDA_ENV}" --no-capture-output \
  python "${SCRIPT}" \
    --port "${PORT}" \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --processed "${PROCESSED}" \
    --theory-name "${THEORY_NAME}" \
    --max-workers 1

echo "[done] Refined data written to: ${OUTPUT}"
