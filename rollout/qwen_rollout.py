#!/usr/bin/env python3
"""
Multi-sample rollout generation with Ray Data + vLLM.

- Reads a JSONL dataset of items with fields:
    - "premise"
    - "question"
    - "id" (dataset-specific id)
    - optional: "answer"

- For each item:
    * Builds a structured chat prompt with explicit reasoning steps.
    * Filters out items whose prompt length exceeds MAX_MODEL_LEN.
    * Replicates each item N_SAMPLES times with distinct seeds.
    * Runs generation using vLLM via Ray Data.
    * Aggregates all N_SAMPLES generations per id into a single JSONL line.

The output JSONL has one record per unique id:
{
  "id": ...,
  "premise": ...,
  "question": ...,
  "answer": ...,
  "generation": [list of raw generations],
  "extracted_answer": [answer tags extracted from generations]
}
"""

import atexit
import gc
import json
import os
import re
import signal
import time
import traceback
from typing import Any, Dict, List, Optional

import ray
from transformers import AutoTokenizer
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# I/O
INPUT_JSONL = "./data.jsonl"
OUTPUT_JSONL = "./qwen_rollout.jsonl"

# Model
MODEL_NAME = "Qwen/Qwen3-8B"

# Sampling / formatting
N_SAMPLES = 5
TEMPERATURE = 1.3
TOP_P = 0.95
MAX_NEW_TOKENS = 3072
BASE_SEED = 20250815

# vLLM engine (OOM-friendly; adjust upward once stable)
GPU_UTIL = 0.60  # fraction of visible VRAM to use
MAX_MODEL_LEN = 3072  # max prompt length in tokens
MAX_NUM_BATCHED_TOKS = 8192  # for older vLLM; limits warm-up / batching
# If your vLLM version supports it, you can instead set: MAX_NUM_SEQS = 16.

# Ray / Data parallelism
VLLM_CONCURRENCY = 1  # number of parallel vLLM replicas
RAY_BATCH_SIZE = 4    # Ray Dataset batch size (data-side, not generations)

# Dataset offset: skip first START_FROM rows if resuming
START_FROM = 0

# CUDA allocation config: helps reduce fragmentation; can be overridden.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Global tokenizer (used for prompt length measurement and chat templating)
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ---------------------------------------------------------------------------
# Cleanup hooks
# ---------------------------------------------------------------------------

_CLEANED = False


def cleanup(reason: str = "normal-exit") -> None:
    """Best-effort teardown to free GPU VRAM even on errors/interrupts."""
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True

    print(f"[cleanup] Cleaning up resources (reason={reason!r})...", flush=True)

    try:
        try:
            ray.shutdown()
        except Exception:
            pass

        # Drop lingering references to large objects
        for name in list(globals()):
            if name in {"ds", "out_ds", "vllm_proc"}:
                try:
                    globals()[name] = None
                except Exception:
                    pass

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    finally:
        # Small delay helps the driver reclaim resources cleanly
        time.sleep(1)


def _handle_signal(sig, frame) -> None:  # type: ignore[override]
    print(f"\n[cleanup] Caught signal {sig}. Cleaning up and exiting...", flush=True)
    try:
        cleanup(reason=f"signal-{sig}")
    finally:
        # Fast exit to avoid stuck threads in Ray / vLLM
        os._exit(1)


for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(_sig, _handle_signal)
    except Exception:
        # Not all platforms support all signals
        pass

atexit.register(lambda: cleanup("atexit"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ANSWER_TAG_RE = re.compile(
    r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>",
    re.IGNORECASE | re.DOTALL,
)


def answer_instruction_for_id(example_id: str) -> str:
    """
    Return a dataset-specific answer-format instruction string for a given id,
    or an empty string if the dataset is unknown.
    """
    prefix = (str(example_id).split("_")[0]).lower()
    mapping: Dict[str, str] = {
        "folio": "Answer with exactly one label: A, B, C.",
        "esnli": (
            "Answer with exactly one label: "
            "entailment, contradiction, or neutral."
        ),
        "multilogieval": "Answer with exactly one label: yes or no.",
        "proofwriter": (
            "Answer with exactly one label: True, False, or Unknown."
        ),
        "prontoqa": "Answer with exactly one label: True or False.",
        "logiqa": (
            "Answer with exactly one label: entailment or not-entailment."
        ),
        "proverqa": (
            "Answer with exactly one label: True, False, or Unknown."
        ),
        "qasc": (
            "Answer with the OPTION LETTER ONLY: "
            "A, B, C, D, E, F, G, or H."
        ),
        # Handles "ar_lsat" etc.
        "ar": (
            "Answer with the OPTION LETTER ONLY: A, B, C, D, or E."
        ),
        "ar_lsat": (
            "Answer with the OPTION LETTER ONLY: A, B, C, D, or E."
        ),
    }
    return mapping.get(prefix, "")


def flatten_tokens(x: Any) -> List[str]:
    """Flatten nested token/segment structures into a flat list of strings."""
    if isinstance(x, list):
        out: List[str] = []
        for y in x:
            out.extend(flatten_tokens(y))
        return out
    if isinstance(x, str):
        return [x]
    return [str(x)]


def to_text(field: Any) -> str:
    """
    Convert nested token-like structures into a single whitespace-normalized
    string, cleaning up spacing around punctuation.
    """
    s = " ".join(flatten_tokens(field)).strip()
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def build_messages(
    premise: str,
    question: str,
    example_id: str,
) -> List[Dict[str, str]]:
    """
    Build chat-style messages (system + user) for the model.

    The prompt asks the model to reason step-by-step in an NLI style, with:
      - Premises
      - Assumptions
      - Conclusion

    It also appends dataset-specific answer-format instructions based on `example_id`.
    """
    system = (
        "You are a meticulous logician. Read the premise and question carefully. "
        "Reason step-by-step explicitly in a Natural Language Inference (NLI) style. "
        "Each reasoning step must follow this format:\n\n"
        "Step X:\n"
        "Premises:\n"
        "- ...\n"
        "- ...\n"
        "Assumptions:\n"
        "- ...\n"
        "Conclusion:\n"
        "- ...\n\n"
        "Number each step (Step 1, Step 2, etc.) in order. "
        "Do not skip steps; ensure each conclusion is directly supported by its "
        "premises and stated assumptions. Please make sure the assumptions make "
        "sense in the given context.\n\n"
        "Ensure that all assumptions are stated clearly and rigorously. For example, "
        "use specific entities instead of vague references such as 'it' or 'this'. "
        "Do not restate premises as assumptions. Premises must remain in the "
        "premises field, while assumptions should capture what is not directly "
        "stated in the premises but can reasonably be inferred using commonsense. "
        "Make sure all necessary information is provided in each step's premises "
        "and assumptions to justify the conclusion.\n\n"
        "After the final step, output exactly ONE final tag:\n"
        "<answer>{LABEL}</answer>\n\n"
        "Rules:\n"
        "1) If the question includes multiple-choice options labeled like "
        "'A) ...', 'B) ...', etc., answer with the OPTION LETTER ONLY (e.g., A).\n"
        "2) Otherwise, answer with a single classification label that fits the "
        "task (e.g., True/False/Unknown, yes/no, "
        "entailment/contradiction/neutral, not-entailment, etc.).\n"
        "3) The label must match the dataset’s expected surface form "
        "(case-sensitive, no extra words).\n"
        "4) After your steps, produce exactly one <answer>...</answer> tag and "
        "nothing else."
    )

    answer_sentence = answer_instruction_for_id(example_id)
    if answer_sentence:
        question_with_constraint = f"{question}\n\n{answer_sentence}"
    else:
        question_with_constraint = question

    user = (
        f"Premise:\n{premise}\n\n"
        f"Question:\n{question_with_constraint}\n\n"
        "Format your reply EXACTLY as follows:\n"
        "Step 1:\n"
        "Premises:\n"
        "- [list the premises used in this step]\n"
        "Assumptions:\n"
        "- [state any implicit assumptions made in this step]\n"
        "Conclusion:\n"
        "- [state the conclusion entailed from those premises and assumptions]\n\n"
        "Step 2:\n"
        "Premises:\n"
        "- ...\n"
        "Assumptions:\n"
        "- ...\n"
        "Conclusion:\n"
        "- ...\n\n"
        "...\n\n"
        "Final:\n"
        "<answer>YOUR_SINGLE_LABEL_OR_OPTION_LETTER</answer>\n\n"
        "Example (for just one step):\n"
        "Step 1:\n"
        "Premises:\n"
        "- All birds have wings.\n"
        "- Penguins are birds.\n"
        "Assumptions:\n"
        "- None\n"
        "Conclusion:\n"
        "- Therefore, penguins have wings.\n\n"
        "Example (assumptions illustrated):\n"
        "Step 1:\n"
        "Premises:\n"
        "- In a school, every student is either studying in the library or "
        "playing in the playground.\n"
        "- Emily was not seen in the library on Friday.\n"
        "Assumptions:\n"
        "- Emily is a student.\n"
        "- \"Not seen in the library\" implies \"not studying in the library.\"\n"
        "Conclusion:\n"
        "- Therefore, Emily was playing in the playground on Friday.\n\n"
        "Final:\n"
        "<answer>yes</answer>\n"
        "Make sure all necessary information is provided in each step's premises "
        "and assumptions to justify the conclusion."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract the contents of the first <answer>...</answer> tag, if present."""
    match = ANSWER_TAG_RE.search(text or "")
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Initialize Ray with 1 GPU and propagate key env vars to actors.
    ray.init(
        num_gpus=1,
        log_to_driver=False,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", ""
                ),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", ""
                ),
            }
        },
    )
    print("Ray resources:", ray.available_resources())

    # -----------------------------------------------------------------------
    # Read JSONL and skip first START_FROM lines
    # -----------------------------------------------------------------------
    text_ds = ray.data.read_text(INPUT_JSONL)
    total = text_ds.count()
    if START_FROM >= total:
        raise ValueError(f"START_FROM ({START_FROM}) >= total rows ({total})")

    _, text_tail = text_ds.split_at_indices([START_FROM])

    # Parse JSON per line
    ds = (
        text_tail.map(lambda r: json.loads(r["text"].strip()))
        .filter(lambda r: r is not None)
    )

    # -----------------------------------------------------------------------
    # Preprocess: strings + messages + prompt_len
    # -----------------------------------------------------------------------
    def _preprocess_strings(row: Dict[str, Any]) -> Dict[str, Any]:
        premise_str = to_text(row["premise"])
        question_str = to_text(row["question"])

        messages = build_messages(
            premise_str,
            question_str,
            row.get("id", ""),
        )

        # Compute decoder prompt length using the SAME chat template.
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            toks = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            prompt_len = len(toks)
        except Exception:
            # Mark as too long (will be filtered out)
            prompt_len = MAX_MODEL_LEN + 1

        return {
            **row,
            "premise_str": premise_str,
            "question_str": question_str,
            "messages": messages,
            "prompt_len": prompt_len,
        }

    ds = ds.map(_preprocess_strings)

    # -----------------------------------------------------------------------
    # Filter: drop items exceeding MAX_MODEL_LEN
    # -----------------------------------------------------------------------
    before = ds.count()
    ds = ds.filter(lambda r: r["prompt_len"] <= MAX_MODEL_LEN)
    after = ds.count()
    print(
        f"Filtered {before - after} items with prompt_len > {MAX_MODEL_LEN}. "
        f"Keeping {after} rows."
    )

    # -----------------------------------------------------------------------
    # Replicate: N_SAMPLES per row with distinct seeds
    # -----------------------------------------------------------------------
    def _replicate_with_sampling(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rep_idx in range(N_SAMPLES):
            sampling_params = {
                "n": 1,  # one sample per replica
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_NEW_TOKENS,
                "seed": BASE_SEED + rep_idx,
            }
            out.append(
                {
                    **row,
                    "rep_idx": rep_idx,
                    "sampling_params": sampling_params,
                }
            )
        return out

    ds = ds.flat_map(_replicate_with_sampling)
    print("Replicated dataset size:", ds.count())

    # -----------------------------------------------------------------------
    # vLLM engine configuration
    # -----------------------------------------------------------------------
    engine_kwargs = {
        "dtype": "bfloat16",
        "gpu_memory_utilization": GPU_UTIL,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKS,
        "tensor_parallel_size": 1,
        "enable_chunked_prefill": True,
        "swap_space": 32,  # GB of host RAM for paged attention
        "enforce_eager": True,
        # "kv_cache_dtype": "fp8",  # if your vLLM build + hardware support it
    }

    config = vLLMEngineProcessorConfig(
        model_source=MODEL_NAME,
        engine_kwargs=engine_kwargs,
        concurrency=VLLM_CONCURRENCY,
        batch_size=RAY_BATCH_SIZE,
    )

    # -----------------------------------------------------------------------
    # vLLM processor I/O
    # -----------------------------------------------------------------------
    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": row["messages"],
            "sampling_params": row["sampling_params"],
            "id": row.get("id"),
            "rep_idx": row["rep_idx"],
            "premise": row["premise_str"],
            "question": row["question_str"],
            "answer": row.get("answer"),
        }

    def _postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        # One output per replica.
        if isinstance(row.get("generated_texts"), list):
            generated = row["generated_texts"][0] if row["generated_texts"] else ""
        else:
            generated = row.get("generated_text", "")

        return {
            "id": row["id"],
            "rep_idx": row["rep_idx"],
            "premise": row["premise"],
            "question": row["question"],
            "answer": row.get("answer"),
            "sample": generated,
            "sample_answer": extract_answer_tag(generated),
        }

    vllm_proc = build_llm_processor(
        config,
        preprocess=_preprocess,
        postprocess=_postprocess,
    )

    out_ds = vllm_proc(ds)

    # -----------------------------------------------------------------------
    # Stream, collate back to one line per id, write JSONL
    # -----------------------------------------------------------------------
    buckets: Dict[Any, Dict[str, Any]] = {}
    wrote = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in out_ds.iter_rows():
            row_id = row["id"]
            record = buckets.get(row_id)
            if record is None:
                record = {
                    "id": row_id,
                    "premise": row["premise"],
                    "question": row["question"],
                    "answer": row.get("answer"),
                    "generation": [],
                    "extracted_answer": [],
                }
                buckets[row_id] = record

            record["generation"].append(row["sample"])
            record["extracted_answer"].append(row["sample_answer"])

            # Write when we collect N_SAMPLES for this id.
            if len(record["generation"]) >= N_SAMPLES:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                wrote += 1
                del buckets[row_id]

        # Flush any partials at the end (if some replicas failed).
        for row_id, record in list(buckets.items()):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            wrote += 1
            del buckets[row_id]

    print(f"Done. Wrote {wrote} rows to {OUTPUT_JSONL}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[cleanup] Unhandled exception; cleaning up...")
        traceback.print_exc()
        cleanup(reason="exception")
        # Uncomment the next line if you want a non-zero exit code for schedulers:
        # raise
    finally:
        cleanup(reason="finally")
