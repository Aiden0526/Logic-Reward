#!/usr/bin/env python3
"""Use LogicReward's existing formalization + Isabelle pipeline on custom data.

Input JSONL (recommended fields per line):
- id
- premise or premises
- explanation or reasoning
- hypothesis or question

Output JSONL adds Isabelle/formalization results, including generated code and
syntactic/semantic validity from LogicReward's IsabelleCritique pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from critique import IsabelleCritique
from generation.gpt import GPT


DEFAULT_PROMPT_DICT = {
    "get davidsonian": "get_davidsonian_form_prompt.txt",
    "get isabelle axiom": "get_isabelle_axiom_prompt.txt",
    "get isabelle theorem no premise": "get_isabelle_theorem_no_premise_prompt.txt",
    "get isabelle theorem with premise": "get_isabelle_theorem_with_premise_prompt.txt",
    "refine contradiction": "refine_contradiction_syntax_error_prompt.txt",
    "refine inner syntax error": "refine_inner_syntax_error_prompt.txt",
    "get isabelle proof": "get_isabelle_proof_prompt.txt",
    "get sentence parse": "get_sentence_parse_prompt.txt",
    "refine davidsonian form": "refine_davidsonian_form_prompt.txt",
    "get logical proposition": "get_logical_proposition_prompt.txt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LogicReward formalization + Isabelle solving on custom JSONL data.")
    p.add_argument("--input", required=True, help="Input JSONL path")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--config", default="config.yaml", help="Path to LogicReward config YAML (default: config.yaml)")
    p.add_argument("--model", default="gpt-4o", help="LLM config section to use (default: gpt-4o)")
    p.add_argument("--port", type=int, default=51200, help="Isabelle server port (default: 51200)")
    p.add_argument("--theory-prefix", default="custom_formal", help="Theory name prefix")
    p.add_argument("--isabelle-session", default="HOL", help="Isabelle session (default: HOL)")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of items")
    p.add_argument("--skip-errors", action="store_true", help="Skip failed items instead of writing error records")
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return obj


def get_api_key(cfg: Dict[str, Any], model_name: str) -> str:
    section = cfg.get(model_name)
    if not isinstance(section, dict):
        raise KeyError(f"Missing config section '{model_name}' in config.yaml")
    api_key = section.get("api_key")
    if not api_key:
        raise ValueError(f"Missing api_key in config section '{model_name}'")
    return str(api_key)


def normalize_premise(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        text = "\n".join(str(x).strip() for x in value if str(x).strip())
        return text or None
    text = str(value).strip()
    return text or None


def first_present(obj: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def build_input_fields(obj: Dict[str, Any]) -> Dict[str, Optional[str]]:
    premise = normalize_premise(first_present(obj, ["premise", "premises"]))
    explanation = first_present(obj, ["explanation", "reasoning", "reasoning_text", "good_answer"])
    hypothesis = first_present(obj, ["hypothesis", "question", "goal"])

    explanation_s = None if explanation is None else str(explanation).strip()
    hypothesis_s = None if hypothesis is None else str(hypothesis).strip()

    return {
        "premise": premise,
        "explanation": explanation_s,
        "hypothesis": hypothesis_s,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    api_key = get_api_key(cfg, args.model)
    isabelle_master_dir = (((cfg.get("isabelle") or {}) if isinstance(cfg.get("isabelle"), dict) else {}).get("master_dir"))
    if not isabelle_master_dir:
        raise ValueError("Missing isabelle.master_dir in config YAML")

    llm = GPT(args.model, api_key)
    critique_model = IsabelleCritique(
        generative_model=llm,
        isabelle_session=args.isabelle_session,
        theory_name=args.theory_prefix,
        port=args.port,
        prompt_dict=DEFAULT_PROMPT_DICT,
    )
    # Allow custom config path instead of hardcoded config.yaml assumptions.
    critique_model.isabelle_dir = str(isabelle_master_dir)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0

    try:
        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line_no, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                if args.limit is not None and processed >= args.limit:
                    break

                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError("record must be a JSON object")

                    fields = build_input_fields(obj)
                    if not fields["explanation"]:
                        raise ValueError("missing 'explanation'/'reasoning'/'good_answer'")
                    if not fields["hypothesis"]:
                        raise ValueError("missing 'hypothesis'/'question'/'goal'")

                    result = critique_model.critique(
                        iteration_number=line_no,
                        explanation=fields["explanation"],
                        hypothesis=fields["hypothesis"],
                        premise=fields["premise"],
                    ) or {}

                    synt = bool(result.get("syntactic validity") or result.get("syntactic_validity"))
                    sem = bool(result.get("semantic validity") or result.get("semantic_validity"))
                    if synt and sem:
                        answer = "provable"
                    elif synt and not sem:
                        answer = "not_provable"
                    else:
                        answer = "syntax_error"

                    out_obj = dict(obj)
                    out_obj["formalization"] = {
                        "answer": answer,
                        "syntactic_validity": synt,
                        "semantic_validity": sem,
                        "solving_time": result.get("solving time", result.get("solving_time")),
                        "proof_tactics": result.get("proof tactics", result.get("proof_tactics", [])),
                        "error_code": result.get("error code", result.get("error_code", "")),
                        "code": result.get("code"),
                        "logical_information": result.get("logical information", result.get("logical_information")),
                    }
                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    processed += 1
                except Exception as e:
                    errors += 1
                    if args.skip_errors:
                        continue
                    err_obj = {"id": None, "line": line_no, "error": f"{type(e).__name__}: {e}"}
                    try:
                        tmp = json.loads(line)
                        if isinstance(tmp, dict) and tmp.get("id") is not None:
                            err_obj["id"] = tmp.get("id")
                    except Exception:
                        pass
                    fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")

        print(f"[done] wrote formalization results to {out_path}")
        print(f"[info] processed={processed}, errors={errors}")
    finally:
        try:
            critique_model.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
