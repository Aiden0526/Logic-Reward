#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

def merge_jsonl():
    # current directory where this script is located
    input_dir = Path(__file__).resolve().parent
    output_file = input_dir / "merged.jsonl"

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {input_dir}")
        return

    total_in, total_out, total_errors = 0, 0, 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for file in files:
            if file.name == output_file.name:
                continue  # skip the merged file if re-running
            print(f"Processing {file} ...")
            with file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    total_in += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        total_errors += 1
                        continue

                    if "error" in obj:
                        total_errors += 1
                        continue

                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total_out += 1

    print(f"\nDone. Input lines: {total_in}, written: {total_out}, skipped (errors): {total_errors}")
    print(f"Merged file saved to: {output_file}")


if __name__ == "__main__":
    merge_jsonl()
