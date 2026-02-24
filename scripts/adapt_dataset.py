#!/usr/bin/env python3
"""Adapt external datasets into LogicReward rollout input JSONL format.

Canonical output schema per line:
  {"id": str, "premise": str, "question": str, "answer": str|None}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".jsonl"}


def _norm(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # Common wrapper shapes
        for key in ("data", "records", "items", "examples"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")


def _load_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no}: expected object, got {type(obj).__name__}")
            yield obj


def _load_delimited_records(path: Path, delimiter: str) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def load_records(path: Path) -> Iterable[Dict[str, Any]]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported input extension {ext!r}. Supported: {sorted(SUPPORTED_EXTS)}")
    if ext == ".json":
        return _load_json_records(path)
    if ext == ".jsonl":
        return _load_jsonl_records(path)
    if ext == ".csv":
        return _load_delimited_records(path, ",")
    if ext == ".tsv":
        return _load_delimited_records(path, "\t")
    raise AssertionError("unreachable")


def pick_field(rec: Dict[str, Any], name: Optional[str], fallback_names: List[str]) -> Optional[str]:
    if name:
        return _norm(rec.get(name))
    for key in fallback_names:
        if key in rec:
            val = _norm(rec.get(key))
            if val is not None:
                return val
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert custom data to LogicReward rollout input JSONL.")
    p.add_argument("--input", required=True, help="Input file (.csv, .tsv, .json, .jsonl)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--dataset-prefix", default="custom", help="ID prefix when --id-col is not provided")
    p.add_argument("--id-col", default=None, help="Column/key for example id (optional)")
    p.add_argument("--premise-col", default=None, help="Column/key for premise/context")
    p.add_argument("--question-col", default=None, help="Column/key for question")
    p.add_argument("--answer-col", default=None, help="Column/key for answer/label (optional)")
    p.add_argument("--constant-answer", default=None, help="Use this answer for all records (optional)")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on records")
    p.add_argument("--skip-missing-answer", action="store_true", help="Drop rows missing answer instead of writing null")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(in_path)

    count_in = 0
    count_out = 0
    count_skipped = 0

    premise_fallbacks = ["premise", "context", "passage"]
    question_fallbacks = ["question", "query", "hypothesis", "prompt"]
    answer_fallbacks = ["answer", "label", "target", "gold"]

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(records, start=1):
            if args.limit is not None and count_out >= args.limit:
                break
            count_in += 1

            premise = pick_field(rec, args.premise_col, premise_fallbacks)
            question = pick_field(rec, args.question_col, question_fallbacks)
            answer = _norm(args.constant_answer) if args.constant_answer is not None else pick_field(rec, args.answer_col, answer_fallbacks)

            if premise is None or question is None:
                count_skipped += 1
                continue
            if args.skip_missing_answer and answer is None:
                count_skipped += 1
                continue

            item_id = pick_field(rec, args.id_col, ["id", "uid", "example_id"]) or f"{args.dataset_prefix}_{idx}"
            out = {
                "id": item_id,
                "premise": premise,
                "question": question,
                "answer": answer,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count_out += 1

    print(f"[done] wrote {count_out} records to {out_path}")
    print(f"[info] skipped {count_skipped} rows with missing required fields")


if __name__ == "__main__":
    main()
