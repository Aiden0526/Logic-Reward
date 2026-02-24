#!/usr/bin/env python3
"""Parse rollout generations into reward-ready structured JSONL.

Replaces the notebook-based `rollout/prepare_reward_data.ipynb` step with a CLI.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STEP_RE = re.compile(r"(?:^|\n)\s*Step\s+(\d+)\s*:?\s*\n", re.IGNORECASE)
FINAL_RE = re.compile(r"(?:^|\n)\s*Final(?:\s+answer)?\s*: ?", re.IGNORECASE)

SECTION_PATTERNS = {
    "premises": re.compile(r"(?:^|\n)\s*Premises?\s*:\s*", re.IGNORECASE),
    "assumptions": re.compile(r"(?:^|\n)\s*Assumptions?\s*:\s*", re.IGNORECASE),
    # Accept common variants in user-provided custom reasoning data.
    "conclusion": re.compile(r"(?:^|\n)\s*(?:Conclusion|Inference|Inferece)\s*:\s*", re.IGNORECASE),
}

ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
FINAL_BRACKET_RE = re.compile(
    r"(?:^|\n)\s*Final(?:\s+answer)?\s*:\s*\[(.*?)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FINAL_TEXT_RE = re.compile(
    r"(?:^|\n)\s*Final(?:\s+answer)?\s*:\s*([^\n\[][^ \n].*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_bullets(block: str) -> List[str]:
    items: List[str] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\u2022]\s*", "", line)
        items.append(line)
    return items


def _extract_sections(step_text: str) -> Dict[str, Any]:
    found = []
    for key, pat in SECTION_PATTERNS.items():
        m = pat.search(step_text)
        if m:
            found.append((key, m.start(), m.end()))

    if not found:
        return {"premises": [], "assumptions": [], "conclusion": ""}

    found.sort(key=lambda x: x[1])
    raw_blocks: Dict[str, str] = {}
    for i, (key, _start, end) in enumerate(found):
        next_start = len(step_text) if i == len(found) - 1 else found[i + 1][1]
        raw_blocks[key] = step_text[end:next_start].strip()

    premises = _clean_bullets(raw_blocks.get("premises", ""))
    assumptions = _clean_bullets(raw_blocks.get("assumptions", ""))
    conclusion = " ".join(_clean_bullets(raw_blocks.get("conclusion", ""))).strip()
    return {"premises": premises, "assumptions": assumptions, "conclusion": conclusion}


def _find_step_blocks(text: str) -> List[Tuple[int, str]]:
    steps = list(STEP_RE.finditer(text))
    if not steps:
        return []

    final_match = FINAL_RE.search(text)
    final_start = final_match.start() if final_match else len(text)

    blocks: List[Tuple[int, str]] = []
    for i, m in enumerate(steps):
        step_num = int(m.group(1))
        start = m.end()
        end = steps[i + 1].start() if i + 1 < len(steps) else final_start
        block = text[start:end].strip()
        if block:
            blocks.append((step_num, block))
    return blocks


def parse_single_generation(gen_text: str) -> Dict[str, Any]:
    steps_info: List[Dict[str, Any]] = []
    for step_num, block in _find_step_blocks(gen_text or ""):
        sec = _extract_sections(block)
        steps_info.append(
            {
                "n": step_num,
                "premises": sec["premises"],
                "assumptions": sec["assumptions"],
                "conclusion": sec["conclusion"],
            }
        )

    step_numbers = [s["n"] for s in steps_info]
    step_order_ok = step_numbers == sorted(step_numbers)

    missing_steps: List[int] = []
    if step_numbers:
        expected = set(range(1, max(step_numbers) + 1))
        missing_steps = sorted(expected - set(step_numbers))

    counts: Dict[int, int] = {}
    for n in step_numbers:
        counts[n] = counts.get(n, 0) + 1
    duplicate_steps = sorted([n for n, c in counts.items() if c > 1])

    text = gen_text or ""
    final_match = ANSWER_TAG_RE.search(text)
    final_tag: Optional[str] = final_match.group(1).strip() if final_match else None
    if final_tag is None:
        m = FINAL_BRACKET_RE.search(text)
        if m:
            final_tag = m.group(1).strip()
    if final_tag is None:
        m = FINAL_TEXT_RE.search(text)
        if m:
            final_tag = m.group(1).strip()

    return {
        "final_tag": final_tag,
        "step_order_ok": bool(step_order_ok),
        "step_numbers": step_numbers,
        "missing_steps": missing_steps,
        "duplicate_steps": duplicate_steps,
        "steps": sorted(steps_info, key=lambda d: d["n"]),
    }


def transform_record(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": data.get("id"),
        "premise": data.get("premise") or data.get("premises"),
        "question": data.get("question"),
        "answer": data.get("answer"),
        "extracted_answer": data.get("extracted_answer"),
        "generations": [],
    }

    generations = (
        data.get("generation")
        or data.get("generations")
        or data.get("reasoning")
        or data.get("reasoning_data")
        or data.get("reasoning data")
        or []
    )
    if not generations and ("good_answer" in data or "bad_answer" in data):
        generations = [x for x in [data.get("good_answer"), data.get("bad_answer")] if x]
    if isinstance(generations, str):
        generations = [generations]
    for gen_text in generations:
        out["generations"].append(parse_single_generation(str(gen_text)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert rollout JSONL into reward-ready structured JSONL.")
    p.add_argument("--input", required=True, help="Input rollout JSONL (raw generations)")
    p.add_argument("--output", required=True, help="Output JSONL path (structured generations)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if isinstance(raw, dict):
                result: Any = transform_record(raw)
            elif isinstance(raw, list):
                result = [transform_record(rec) for rec in raw if isinstance(rec, dict)]
            else:
                raise ValueError(f"Line {line_no}: unsupported record type {type(raw).__name__}")
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] wrote {written} records to {out_path}")


if __name__ == "__main__":
    main()
