#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-Server, Multi-Thread Refinement Pipeline (CLI-enabled)

Highlights
---------
- Uses exactly one shared Isabelle server on a fixed port (provided via --port or env)
- Gates concurrent calls to the server with a bounded semaphore to avoid overload
- Writes per-thread sharded outputs: ..._1.jsonl, ..._2.jsonl, ...
- Guarantees every processed step gets 'logic_validity' and 'syntactic_validity' fields
- Python 3.9-friendly (no PEP 604 union types, explicit typing)
- CLI overrides for port, input/output paths, processed path, theory name, etc.
- Robust logging (to stdout) with optional --verbose flag

Usage examples
--------------
    python refinement_pipeline_cli.py \
        --port 51234 \
        --input ./reward_data/pr_sampled.jsonl \
        --output ./reward_data/Multi-Thread/pr_sampled_refined_last.jsonl \
        --processed ./reward_data/pr_sampled_refined.jsonl

    # With defaults (port uses --port, else $ISABELLE_PORT, else 51200)
    python refinement_pipeline_cli.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import yaml
from tqdm.auto import tqdm

# ==== Optional rich logging (falls back gracefully) ====
try:
    from rich.logging import RichHandler  # type: ignore
    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _RICH_AVAILABLE = False

# ==== Project imports (adjust if your package layout differs) ====
# These imports are expected to exist in your project.
from refinement.refinement_model import RefinementModel  # type: ignore
from generation.gpt import GPT  # type: ignore
from critique import IsabelleCritique  # type: ignore


# =========================
# Defaults & Constants
# =========================
# DEFAULT_INPUT = "./reward_data/pr_sampled.jsonl"
DEFAULT_OUTPUT = "./reward_data/Multi-Thread/refined_reasoning.jsonl"
# DEFAULT_PROCESSED = "./reward_data/refined_reasoning.jsonl"  # file or directory
DEFAULT_THEORY_NAME = "symb_1"
DEFAULT_MAX_WORKERS = 1
DEFAULT_MAX_ITEMS: Optional[int] = None
DEFAULT_SERVER_PORT = int(os.getenv("ISABELLE_PORT", "51200"))
ISABELLE_MAX_CONCURRENCY = int(os.getenv("ISABELLE_MAX_CONCURRENCY", "2"))
SERVER_READY_TIMEOUT = 30.0  # seconds (kept for parity; server is assumed up)

# Bounded semaphore: limit concurrent calls to the single Isabelle server
ISABELLE_CONN_SEMA = threading.BoundedSemaphore(ISABELLE_MAX_CONCURRENCY)

# Regex to strip trailing "Final: <answer>...</answer>" block from conclusions
FINAL_BLOCK_RE = re.compile(r"(?is)\n\s*Final\s*:\s*<answer>.*?</answer>\s*$", re.MULTILINE)


# =========================
# Logging
# =========================

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if _RICH_AVAILABLE:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        )
    else:  # pragma: no cover - simple fallback
        logging.basicConfig(
            level=level,
            format="[%(levelname)s] %(message)s",
        )


log = logging.getLogger(__name__)


# =========================
# Helpers
# =========================

def as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    return [str(x)]


def extract_hypothesis(conclusion_text: Optional[str]) -> str:
    if conclusion_text is None:
        return ""
    hyp = FINAL_BLOCK_RE.sub("", str(conclusion_text)).strip()
    hyp = re.sub(r"^\s*-\s*", "", hyp)
    return hyp


# =========================
# Sharded output (one file per worker thread)
# =========================

_thread_to_shard: Dict[int, int] = {}
_thread_map_lock = threading.Lock()


def _get_shard_id(max_workers: int) -> int:
    tid = threading.get_ident()
    with _thread_map_lock:
        sid = _thread_to_shard.get(tid)
        if sid is None:
            sid = len(_thread_to_shard) + 1
            if sid > max_workers:
                sid = ((sid - 1) % max_workers) + 1
            _thread_to_shard[tid] = sid
        return sid


def _with_suffix(base_path: str, shard_id: int) -> str:
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_{shard_id}{p.suffix}"))


def save_output_sharded(output_json: dict | str, base_out_path: str, max_workers: int) -> None:
    shard_id = _get_shard_id(max_workers)
    shard_path = _with_suffix(base_out_path, shard_id)
    Path(shard_path).parent.mkdir(parents=True, exist_ok=True)

    line = output_json if isinstance(output_json, str) else json.dumps(output_json, ensure_ascii=False)
    if not line.endswith("\n"):
        line += "\n"

    # append-only JSONL per-shard
    with open(shard_path, "a", encoding="utf-8") as f:
        f.write(line)


# =========================
# Processed IDs loader (file or directory)
# =========================

def load_processed_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    p = Path(path)
    if not p.exists():
        return ids

    def add_from_file(fp: Path) -> None:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if "id" in obj and obj["id"] is not None:
                            ids.add(str(obj["id"]))
                    except Exception:
                        # ignore malformed lines
                        continue
        except Exception as e:  # pragma: no cover
            log.warning("Failed to read processed file %s: %s", fp, e)

    if p.is_dir():
        for fp in sorted(p.glob("*.jsonl")):
            add_from_file(fp)
    else:
        add_from_file(p)
    return ids


# =========================
# LLM / Critique Setup
# =========================

def load_api_key(config_path: str) -> Optional[str]:
    """Load API key from YAML config. Returns None if missing/unreadable."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
            return (cfg.get("gpt-4o") or {}).get("api_key")
    except FileNotFoundError:
        log.warning("config file not found at %s; proceeding without API key", config_path)
    except Exception as e:  # pragma: no cover
        log.warning("failed to read config at %s: %s", config_path, e)
    return None


def build_refiner(api_key: Optional[str], theory_name: str, port: int, prompt_dict: Dict[str, str]) -> RefinementModel:
    # GPT client
    if not api_key:
        log.warning("API key missing; GPT client will likely fail unless alternative auth is used.")
    llm = GPT("gpt-4o", api_key)  # type: ignore[arg-type]

    # Critique solver (shared across items)
    solver = IsabelleCritique(
        generative_model=llm,
        isabelle_session="HOL",
        theory_name=theory_name,
        port=port,
    )
    log.info("Using Isabelle server at :%d", port)

    # Refinement orchestrator
    return RefinementModel(
        generative_model=llm,
        critique_models={"hard": [solver]},
        critique_mode="hard",
        prompt_dict=prompt_dict,
    )


# =========================
# Worker: process one JSONL line
# =========================

def process_one(index: int, line: str, refiner: RefinementModel, max_workers: int) -> Tuple[int, bool, int, int]:
    """Process a single JSONL line.

    Returns: (index, wrote_bool, steps_updated, refine_errors)
    """
    local_updated = 0
    local_errors = 0

    if not line.strip():
        return index, False, local_updated, local_errors

    try:
        item = json.loads(line)
    except Exception:
        log.debug("Skipping malformed JSON at line %d", index)
        return index, False, local_updated, local_errors

    try:
        gens = item.get("generations") or []
        for gi, gen in enumerate(gens):
            steps = gen.get("steps") or []
            for step in steps:
                prem_list = as_list(step.get("premises") or step.get("premise"))
                llm_assumption_in = " ".join(as_list(step.get("assumptions")))
                explanation = " ".join(prem_list) if prem_list else ""
                hypothesis = extract_hypothesis(step.get("conclusion"))

                # Gate concurrent calls into the single server
                with ISABELLE_CONN_SEMA:
                    results = refiner.refine(
                        hypothesis=hypothesis,
                        premise=None,
                        original_explanation=explanation,
                        llm_assumption=llm_assumption_in,
                        iterations=5,
                    )

                refined_text = results.get("llm assumption")
                logic_validity = bool(results.get("semantic validity", False))
                syntactic_validity = bool(results.get("syntactic validity", False))

                step["logic_validity"] = 1.0 if logic_validity else 0.0
                step["syntactic_validity"] = 1.0 if syntactic_validity else 0.0
                if refined_text is not None:
                    step["assumptions"] = refined_text
                    local_updated += 1

            log.debug("Finished generation %s for item %s", gi, item.get("id", ""))

        # Write updated item to this thread's shard file
        out_line = json.dumps(item, ensure_ascii=False)
        save_output_sharded(out_line, args.output_path, max_workers)
        return index, True, local_updated, local_errors

    except KeyboardInterrupt:  # pragma: no cover - surface clean interruption
        raise
    except Exception as e:
        local_errors += 1
        error_msg = {
            "id": item.get("id", ""),
            "error": f"{type(e).__name__}: {e}",
        }
        save_output_sharded(json.dumps(error_msg, ensure_ascii=False), args.output_path, max_workers)
        log.exception("Error while processing item %s", item.get("id", ""))
        return index, True, local_updated, local_errors


# =========================
# CLI parsing
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-Server, Multi-Thread Refinement Pipeline")

    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT,
                        help="Port for the Isabelle server (default: $ISABELLE_PORT or 51200)")
    parser.add_argument("--input", dest="input_path", default=DEFAULT_INPUT,
                        help="Path to input JSONL (default: ./reward_data/pr_sampled.jsonl)")
    parser.add_argument("--output", dest="output_path", default=DEFAULT_OUTPUT,
                        help=("Base output JSONL path used for sharded writes "
                              "(default: ./reward_data/Multi-Thread/pr_sampled_refined_last.jsonl)"))
    parser.add_argument("--processed", dest="processed_path", default=DEFAULT_PROCESSED,
                        help="File or directory of prior results to skip (default: ./reward_data/pr_sampled_refined.jsonl)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Number of worker threads (default: 1)")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS,
                        help="Optional limit on number of items to process (default: process all)")
    parser.add_argument("--theory-name", dest="theory_name", default=DEFAULT_THEORY_NAME,
                        help="Name of the Isabelle theory to use (default: symb_1)")
    parser.add_argument("--config", dest="config_path", default="./config.yaml",
                        help="Path to YAML config containing GPT API key (default: ./config.yaml)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


# =========================
# Main
# =========================

def precount_items(input_path: Path, processed_ids: Set[str], max_items: Optional[int]) -> Tuple[int, int]:
    """Count items to process, skipping those already processed."""
    total_to_process = 0
    skipped_precount = 0
    try:
        with input_path.open("r", encoding="utf-8") as _fin:
            for i, line in enumerate(_fin):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    _id = str(obj.get("id"))
                except Exception:
                    _id = None
                if _id and _id in processed_ids:
                    skipped_precount += 1
                    continue
                total_to_process += 1
                if max_items is not None and total_to_process >= max_items:
                    break
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return total_to_process, skipped_precount


def main(cli_args: Optional[argparse.Namespace] = None) -> None:
    global args
    args = cli_args or parse_args()

    setup_logging(args.verbose)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(args.processed_path)
    log.info("Loaded %d processed IDs from %s", len(processed_ids), args.processed_path)

    # Pre-count for nicer progress bars
    total_to_process, skipped = precount_items(input_path, processed_ids, args.max_items)
    log.info("Pre-count: %d items to process (skipped %d already-processed) from %s.",
             total_to_process, skipped, input_path)

    # Build refiner stack
    api_key = load_api_key(args.config_path)
    if api_key and len(api_key) >= 8:
        mask = f"{api_key[:4]}...{api_key[-4:]}"
    else:
        mask = api_key or "None"
    log.info("Loaded API key: %s", mask)

    prompt_dict = {
        "generate explanation": "get_explanation_prompt.txt",
        "refine no premise": "refine_hard_no_premise_prompt.txt",
        "refine with premise": "refine_hard_with_premise_prompt.txt",
    }

    refiner = build_refiner(api_key, args.theory_name, args.port, prompt_dict)

    processed = 0
    items_written = 0
    updated_steps = 0
    refine_errors = 0

    def limited_reader(fh: Iterable[str]) -> Iterable[Tuple[int, str]]:
        if args.max_items is None:
            yield from enumerate(fh)
        else:
            for i, l in enumerate(fh):
                if i >= args.max_items:
                    break
                yield i, l

    with input_path.open("r", encoding="utf-8") as fin:
        submissions: Dict[object, int] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            for idx, line in limited_reader(fin):
                # skip if already processed
                try:
                    obj = json.loads(line)
                    _id = str(obj.get("id"))
                except Exception:
                    _id = None
                if _id and _id in processed_ids:
                    continue

                fut = ex.submit(process_one, idx, line, refiner, args.max_workers)
                submissions[fut] = idx

            pbar = tqdm(total=len(submissions), desc="Refining (multithreaded)", unit="item")

            for fut in as_completed(submissions):
                idx, wrote, loc_updated, loc_errors = fut.result()
                updated_steps += loc_updated
                refine_errors += loc_errors
                processed += 1
                if wrote:
                    items_written += 1
                pbar.update(1)

            pbar.close()

    log.info("Done. Wrote %d items to sharded files based on %s", items_written, args.output_path)
    log.info("Steps updated: %d | refine errors: %d", updated_steps, refine_errors)


if __name__ == "__main__":
    main()
