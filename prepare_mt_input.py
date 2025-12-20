#!/usr/bin/env python3
"""
Split a JSONL file evenly into N shard files.

Example:
    python split_jsonl.py \
        --input rollout_processed.jsonl \
        --output-dir ./reward_data/Multi-Thread/input \
        --threads 4

Produces:
    ./reward_data/Multi-Thread/input/input_1.jsonl
    ./reward_data/Multi-Thread/input/input_2.jsonl
    ./reward_data/Multi-Thread/input/input_3.jsonl
    ./reward_data/Multi-Thread/input/input_4.jsonl
"""

import argparse
from pathlib import Path

def split_jsonl(input_path: str, output_dir: str, threads: int, prefix: str = "input_") -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all lines once (if extremely large, a streaming approach can be used)
    with input_path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    total = len(lines)
    per_file = (total + threads - 1) // threads  # ceiling division

    print(f"[info] total records: {total}")
    print(f"[info] threads: {threads} -> ~{per_file} records per shard")

    for i in range(threads):
        start = i * per_file
        end = min(start + per_file, total)
        if start >= end:
            break  # no more data

        out_path = output_dir / f"{prefix}{i+1}.jsonl"
        with out_path.open("w", encoding="utf-8") as f_out:
            for line in lines[start:end]:
                f_out.write(line if line.endswith("\n") else line + "\n")

        print(f"[ok] wrote shard {i+1}: {out_path} ({end-start} lines)")

    print("[done] split complete.")


def main():
    parser = argparse.ArgumentParser(description="Split JSONL into multiple shard files.")
    parser.add_argument("--input", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output-dir", required=True, help="Directory to write shard files")
    parser.add_argument("--threads", type=int, required=True, help="Number of shard files")
    args = parser.parse_args()

    split_jsonl(args.input, args.output_dir, args.threads)


if __name__ == "__main__":
    main()
