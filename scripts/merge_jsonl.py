#!/usr/bin/env python3
"""Merge JSONL shard files and optionally drop error rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge JSONL files in a directory.")
    p.add_argument("--input-dir", required=True, help="Directory containing shard .jsonl files")
    p.add_argument("--output", required=True, help="Merged JSONL output file")
    p.add_argument("--pattern", default="*.jsonl", help="Glob pattern (default: *.jsonl)")
    p.add_argument("--keep-errors", action="store_true", help="Keep rows containing an 'error' key")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.glob(args.pattern) if p.is_file())
    files = [p for p in files if p.resolve() != output_path.resolve()]
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern!r} in {input_dir}")

    total_in = total_out = skipped_error = skipped_parse = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for path in files:
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    total_in += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_parse += 1
                        continue
                    if isinstance(obj, dict) and ("error" in obj) and not args.keep_errors:
                        skipped_error += 1
                        continue
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total_out += 1

    print(f"[done] merged {len(files)} files -> {output_path}")
    print(f"[info] input lines={total_in}, written={total_out}, skipped_error={skipped_error}, skipped_parse={skipped_parse}")


if __name__ == "__main__":
    main()
