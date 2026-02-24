PYTHON ?= python

ADAPT_INPUT ?= data/custom/raw/input.csv
ADAPT_OUTPUT ?= rollout/data_custom.jsonl
PREMISE_COL ?= premise
QUESTION_COL ?= question
ANSWER_COL ?= answer
ID_COL ?= id
DATASET_PREFIX ?= custom

ROLLOUT_RAW ?= rollout/qwen_rollout.jsonl
ROLLOUT_PROCESSED ?= rollout/rollout_processed.jsonl

REWARD_INPUT ?= $(ROLLOUT_PROCESSED)
REWARD_OUTPUT ?= reward_data/rewarded_data_single.jsonl
PORT ?= 40000

REFINE_INPUT ?= $(REWARD_OUTPUT)
REFINE_OUTPUT ?= refine_data/refined_data_single.jsonl
REFINE_PORT ?= 10000
THEORY_NAME ?= symb_single

.PHONY: help adapt-custom prepare-reward merge-reward-shards reward refine

help:
	@echo "Targets:"
	@echo "  make adapt-custom      # convert custom CSV/JSON/JSONL to rollout input JSONL"
	@echo "  make prepare-reward    # parse rollout generations into reward-ready format"
	@echo "  make reward            # run reward labeling (single-thread)"
	@echo "  make refine            # run refinement (single-thread)"
	@echo "  make merge-reward-shards DIR=... OUT=..."

adapt-custom:
	$(PYTHON) scripts/adapt_dataset.py \
	  --input "$(ADAPT_INPUT)" \
	  --output "$(ADAPT_OUTPUT)" \
	  --dataset-prefix "$(DATASET_PREFIX)" \
	  --id-col "$(ID_COL)" \
	  --premise-col "$(PREMISE_COL)" \
	  --question-col "$(QUESTION_COL)" \
	  --answer-col "$(ANSWER_COL)"

prepare-reward:
	$(PYTHON) scripts/prepare_reward_data.py \
	  --input "$(ROLLOUT_RAW)" \
	  --output "$(ROLLOUT_PROCESSED)"

merge-reward-shards:
	@test -n "$(DIR)" || (echo "DIR is required" && exit 1)
	@test -n "$(OUT)" || (echo "OUT is required" && exit 1)
	$(PYTHON) scripts/merge_jsonl.py --input-dir "$(DIR)" --output "$(OUT)"

reward:
	PORT="$(PORT)" INPUT="$(REWARD_INPUT)" OUTPUT="$(REWARD_OUTPUT)" bash run_reward.sh

refine:
	PORT="$(REFINE_PORT)" INPUT="$(REFINE_INPUT)" OUTPUT="$(REFINE_OUTPUT)" THEORY_NAME="$(THEORY_NAME)" bash run_refine.sh
