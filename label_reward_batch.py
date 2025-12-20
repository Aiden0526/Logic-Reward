import argparse
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

W_PREM = 0.5
W_LOGIC = 0.5
COUNT_TOTAL = True
MAX_ITEMS: Optional[int] = None
NUM_WORKERS = 5
BUFFER_FACTOR = 5
USE_THREADS = True
START_METHOD = "spawn"
FINAL_BLOCK_RE = re.compile(r"(?is)\n\s*Final\s*:\s*<answer>.*?</answer>\s*$", re.MULTILINE)

MODEL_NAME = "Qwen/Qwen3-8B"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
_model.to(_device)
_model.eval()

_embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

_UPDATE_PREV_ON_ANY_STEP = True
_LOGIC_SOFT_CAP = 0.7
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\n\r]+|^\s*[-•]\s*", re.MULTILINE)


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)


def split_into_atomic_premises(text_or_list) -> List[str]:
    if text_or_list is None:
        return []
    if isinstance(text_or_list, list):
        atoms: List[str] = []
        for x in text_or_list:
            x = str(x)
            parts = [p.strip(" -•\t") for p in _SENT_SPLIT_RE.split(x) if p and p.strip(" -•\t")]
            atoms.extend(parts if parts else [x.strip()])
        return [a for a in atoms if a]
    x = str(text_or_list)
    parts = [p.strip(" -•\t") for p in _SENT_SPLIT_RE.split(x) if p and p.strip(" -•\t")]
    return [p for p in parts if p] or [x.strip()]


@torch.no_grad()
def _embed(texts: List[str]) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 384)
    return _embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)


def _cosine_max(vec: torch.Tensor, bank: torch.Tensor) -> float:
    if bank.numel() == 0 or vec.numel() == 0:
        return 0.0
    v = vec.view(1, -1)
    return float((bank @ v.T).max().item())


class PremiseScorer:
    def __init__(self, item_given_premises):
        base_atoms = split_into_atomic_premises(item_given_premises)
        self.base_texts = [_norm_text(t) for t in base_atoms if t and t.strip()]
        self.base_embs = _embed(self.base_texts)
        self.prev_text: Optional[str] = None
        self.prev_emb: Optional[torch.Tensor] = None

    def update_prev_conclusion(self, text: Optional[str], use_any: bool = _UPDATE_PREV_ON_ANY_STEP):
        t = _norm_text(text or "")
        if not t:
            return
        self.prev_text = t
        self.prev_emb = _embed([t])[0]

    def premise_reward(self, used_premises) -> float:
        used = used_premises or []
        if not isinstance(used, list):
            used = [str(used)]
        else:
            used = [str(u) for u in used]
        used = [u for u in used if u and str(u).strip()]
        if not used:
            return 1.0
        bank = self.base_embs
        rewards: List[float] = []
        for u in used:
            u_vec = _embed([_norm_text(str(u))])[0]
            sims: List[float] = []
            if bank.numel() > 0:
                sims.append(_cosine_max(u_vec, bank))
            if self.prev_emb is not None:
                sims.append(float((self.prev_emb @ u_vec).item()))
            best = max(sims) if sims else 0.0
            rewards.append(max(0.0, min(1.0, best)))
        return float(sum(rewards) / len(rewards))


def _toks(text: str):
    return _tokenizer(text, return_tensors="pt").to(_device)


def _winsorize(x: torch.Tensor, pct: float = 0.10) -> torch.Tensor:
    if x.numel() < 5 or not (0.0 < pct < 0.5):
        return x
    lo = torch.quantile(x, pct)
    hi = torch.quantile(x, 1.0 - pct)
    return torch.clamp(x, lo, hi)


@torch.no_grad()
def soft_step_confidence(
    step_text: str,
    context: Optional[str] = None,
    winsor_pct: float = 0.10,
    gamma: float = 2.0,
    prior_mean: float = -3.0,
    prior_std: float = 1.5,
) -> Tuple[float, float]:
    step_text = (step_text or "").strip()
    if not step_text:
        return 0.0, float("-inf")
    if context:
        context = context.strip()
        full = context + step_text
        enc_full = _toks(full)
        enc_ctx = _toks(context)
        labels = enc_full.input_ids.clone()
        ctx_len = enc_ctx.input_ids.size(1)
        labels[:, :ctx_len] = -100
        logits = _model(**enc_full, labels=labels).logits
        logits = logits[:, :-1, :]
        step_labels = labels[:, 1:]
        valid_mask = (step_labels != -100).reshape(-1)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            step_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        )
        nll = nll[valid_mask]
    else:
        enc = _toks(step_text)
        labels = enc.input_ids.clone()
        logits = _model(**enc, labels=labels).logits
        logits = logits[:, :-1, :]
        step_labels = labels[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            step_labels.reshape(-1),
            reduction="none",
        )
    if nll.numel() == 0:
        return 0.0, float("-inf")
    logprobs = -nll
    logprobs = _winsorize(logprobs, winsor_pct)
    C = logprobs.mean().item()
    Z = (C - prior_mean) / (prior_std + 1e-8)
    conf = float(torch.tanh(torch.tensor(gamma * Z)))
    return conf, C


def soft_score_when_hard_fails(step: Dict) -> float:
    premises = step.get("premises") or step.get("premise")
    explanation = "\n".join(premises) if isinstance(premises, list) else (premises or "")
    hypothesis = step.get("conclusion") or ""
    hypothesis = re.sub(r"(?is)\n\s*Final\s*:\s*<answer>.*?</answer>\s*$", "", hypothesis).strip()
    hypothesis = re.sub(r"^\s*-\s*", "", hypothesis).strip()
    conf, _ = soft_step_confidence(
        step_text=hypothesis if hypothesis else explanation,
        context=explanation if hypothesis else None,
        winsor_pct=0.10,
        gamma=2.0,
    )
    soft_score = max(0.0, (conf + 1.0) / 2.0) * 0.5
    return float(soft_score)


def compute_logic_reward(
    *,
    explanation: str,
    hypothesis: str,
    critique_model,
    soft_step_confidence,
    model,
    tokenizer,
) -> Tuple[float, bool, bool]:
    def _parse_flags(resp: dict) -> Tuple[bool, bool]:
        synt_ok = bool(resp.get("syntactic_validity") or resp.get("syntactic validity"))
        sem_ok = bool(resp.get("semantic_validity") or resp.get("semantic validity"))
        return synt_ok, sem_ok

    last_attempt: Tuple[bool, bool] = (False, False)
    last_syntactic_true: Optional[Tuple[bool, bool]] = None

    for _ in range(3):
        try:
            resp = critique_model.critique(
                iteration_number=2000,
                explanation=explanation,
                hypothesis=hypothesis,
                premise=None,
            ) or {}
            syntactic_ok, semantic_ok = _parse_flags(resp)
        except Exception:
            syntactic_ok, semantic_ok = False, False
        last_attempt = (syntactic_ok, semantic_ok)
        if syntactic_ok and semantic_ok:
            return 1.0, True, True
        if syntactic_ok:
            last_syntactic_true = (syntactic_ok, semantic_ok)

    chosen = last_syntactic_true if last_syntactic_true is not None else last_attempt
    chosen_synt, chosen_sem = chosen

    if chosen_synt:
        return (1.0 if chosen_sem else 0.0), True, chosen_sem

    try:
        step_text = hypothesis if hypothesis else explanation
        ctx = explanation if hypothesis else None
        conf, _ = soft_step_confidence(step_text=step_text, context=ctx)
    except Exception:
        conf = 0.0
    conf = float(max(0.0, min(1.0, conf)))
    return conf, False, False


def _shard_path(base_path: Path) -> Path:
    pid = os.getpid()
    tid = threading.get_ident()
    return base_path.with_suffix(base_path.suffix + f".part-{pid}-{tid}.jsonl")


def _append_to_shard(base_path: Path, line: str) -> Path:
    shard = _shard_path(base_path)
    shard.parent.mkdir(parents=True, exist_ok=True)
    with open(shard, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return shard


def init_worker():
    return None


def load_excluded_ids(path: Optional[Path]) -> Set[str]:
    excluded: Set[str] = set()
    if not path:
        return excluded
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "id" in obj:
                    sid = obj["id"]
                    if sid is not None:
                        excluded.add(str(sid))
                else:
                    excluded.add(s)
            except json.JSONDecodeError:
                excluded.add(s)
    return excluded


def extract_id_from_line(line: str) -> Optional[str]:
    try:
        obj = json.loads(line)
        return None if not isinstance(obj, dict) else (None if "id" not in obj else str(obj["id"]))
    except Exception:
        return None


def process_one_line(line: str):
    isabelle_solver = None
    item_id: Optional[str] = None
    try:
        if not line or not line.strip():
            return True, True, None, None
        item = json.loads(line)
        item_id = str(item.get("id")) if item.get("id") is not None else None

        from refinement.refinement_model import RefinementModel
        from generation.gpt import GPT
        from critique import IsabelleCritique
        import yaml
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity(hf_logging.CRITICAL)
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            api_key = config.get("gpt-4o", {}).get("api_key")
        llm = GPT("gpt-4o", api_key)
        isabelle_solver = IsabelleCritique(generative_model=llm, isabelle_session="HOL")
        critique_model = isabelle_solver

        gold = (item.get("answer") or "").strip()
        gens = item.get("generations") or []
        item_given = item.get("premise") or item.get("premises") or []
        premise_scorer = PremiseScorer(item_given)

        reward_list: List[float] = []
        per_gen_step_metrics: List[List[Dict[str, float]]] = []

        for i, gen in enumerate(gens):
            steps = gen.get("steps") or []
            step_scores: List[float] = []
            step_metrics: List[Dict[str, float]] = []
            last_conclusion_for_bank: Optional[str] = None

            for n, step in enumerate(steps):
                prem_list = (step.get("premises") or step.get("premise")) or []
                if not isinstance(prem_list, list):
                    prem_list = [str(prem_list)]
                else:
                    prem_list = [str(x) for x in prem_list]
                if last_conclusion_for_bank:
                    premise_scorer.update_prev_conclusion(last_conclusion_for_bank)
                explanation = " ".join(f"{p}" for p in prem_list) if prem_list else ""
                conclusion = step.get("conclusion")
                hypothesis = "" if conclusion is None else FINAL_BLOCK_RE.sub("", str(conclusion)).strip()
                hypothesis = re.sub(r"^\s*-\s*", "", hypothesis)
                prem_reward = premise_scorer.premise_reward(prem_list)
                assumptions = step.get("assumption") or []
                explanation = (explanation + " " + " ".join(assumptions)).strip()
                logic_validity = step["logic_validity"] if "logic_validity" in step else None
                if logic_validity is not None:
                    logic_reward = float(logic_validity)
                    syntactic_ok, semantic_ok = True, True
                else:
                    logic_reward, syntactic_ok, semantic_ok = compute_logic_reward(
                        explanation=explanation,
                        hypothesis=hypothesis,
                        critique_model=critique_model,
                        soft_step_confidence=soft_step_confidence,
                        model=_model,
                        tokenizer=_tokenizer,
                    )
                step_score = W_PREM * float(prem_reward) + W_LOGIC * float(logic_reward)
                step_scores.append(float(step_score))
                step_metrics.append(
                    {
                        "premise_validity": float(prem_reward),
                        "logic_validity": float(logic_reward),
                        "step_score": float(step_score),
                        "syntactic_ok": bool(syntactic_ok),
                        "semantic_ok": bool(semantic_ok),
                    }
                )
                last_conclusion_for_bank = hypothesis if hypothesis else None

            num_steps = len(steps)
            reasoning_score = (sum(step_scores) / num_steps) if num_steps > 0 else 0.0
            final_tag = (gen.get("final_tag") or "").strip()
            outcome_score = 1.0 if final_tag == gold else 0.0
            final_reward = 0.5 * reasoning_score + 0.5 * outcome_score
            final_reward = min(1.0, max(0.0, final_reward))
            reward_list.append(float(final_reward))
            per_gen_step_metrics.append(step_metrics)

        item_out = dict(item)
        item_out["reward_score"] = reward_list
        item_out["per_step_validity"] = per_gen_step_metrics
        json_line = json.dumps(item_out, ensure_ascii=False)
        shard = _append_to_shard(Path(OUTPUT_REWARDED_JSONL), json_line)
        try:
            if isabelle_solver is not None:
                isabelle_solver.shutdown()
        except Exception:
            pass
        return True, False, None, str(shard)

    except Exception as e:
        try:
            if isabelle_solver is not None:
                isabelle_solver.shutdown()
        except Exception:
            pass
        if item_id is None:
            try:
                item_id = extract_id_from_line(line)
            except Exception:
                item_id = None
        err_record = {"id": item_id, "error": f"{type(e).__name__}: {e}"}
        shard = _append_to_shard(Path(OUTPUT_REWARDED_JSONL), json.dumps(err_record, ensure_ascii=False))
        return False, False, err_record["error"], str(shard)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pos", nargs="?", type=Path)
    parser.add_argument("output_pos", nargs="?", type=Path)
    parser.add_argument("skip_pos", nargs="?", type=Path)
    parser.add_argument("-i", "--input", dest="input_extracted_jsonl", type=Path)
    parser.add_argument("-o", "--output", dest="output_rewarded_jsonl", type=Path)
    parser.add_argument("-s", "--skip", dest="skip_ids_jsonl", type=Path, default=None)
    parser.add_argument("-d", "--data", dest="data_name", type=str, default="Symb_1")
    parser.add_argument("-p", "--port", dest="port", type=int, default=8000)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker threads/processes")
    parser.add_argument("--buffer-factor", type=int, default=1, help="Max outstanding futures multiplier")
    args = parser.parse_args()

    global INPUT_EXTRACTED_JSONL, OUTPUT_REWARDED_JSONL
    INPUT_EXTRACTED_JSONL = args.input_pos or args.input_extracted_jsonl
    OUTPUT_REWARDED_JSONL = args.output_pos or args.output_rewarded_jsonl
    skip_ids_path = args.skip_pos or args.skip_ids_jsonl

    global NUM_WORKERS, BUFFER_FACTOR
    NUM_WORKERS = max(1, int(args.num_workers))
    BUFFER_FACTOR = max(1, int(args.buffer_factor))

    in_path = Path(INPUT_EXTRACTED_JSONL)
    out_path = Path(OUTPUT_REWARDED_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_ids = load_excluded_ids(skip_ids_path)

    total_lines = None
    if COUNT_TOTAL:
        total = 0
        with open(in_path, "r", encoding="utf-8") as _fin:
            for _line in _fin:
                if not _line.strip():
                    continue
                if excluded_ids:
                    _id = extract_id_from_line(_line)
                    if _id is not None and _id in excluded_ids:
                        continue
                total += 1
        total_lines = total

    processed = 0
    errors = 0
    skipped = 0
    skipped_by_exclusion = 0

    if USE_THREADS:
        Executor = ThreadPoolExecutor
        exec_kwargs = {"max_workers": NUM_WORKERS}
    else:
        ctx = mp.get_context(START_METHOD)
        Executor = ProcessPoolExecutor
        exec_kwargs = {"max_workers": NUM_WORKERS, "mp_context": ctx, "initializer": init_worker}

    shard_paths: Set[str] = set()

    with open(in_path, "r", encoding="utf-8") as fin:
        pbar = tqdm(total=total_lines, desc="Labeling", unit="item")
        with Executor(**exec_kwargs) as ex:
            outstanding = []
            submitted = 0

            def maybe_drain(n_to_take: int):
                nonlocal processed, errors, skipped, shard_paths
                taken = 0
                for fut in as_completed(outstanding[:n_to_take]):
                    ok, was_skipped, err, shard = fut.result()
                    processed += 1
                    if ok and not was_skipped:
                        if shard:
                            shard_paths.add(shard)
                    elif was_skipped:
                        skipped += 1
                    else:
                        errors += 1
                        print(f"Error processing item #{processed}: {err}", file=sys.stderr)
                    pbar.update(1)
                    outstanding.remove(fut)
                    taken += 1
                return taken

            for line in fin:
                if not line.strip():
                    continue
                if excluded_ids:
                    _id = extract_id_from_line(line)
                    if _id is not None and _id in excluded_ids:
                        skipped_by_exclusion += 1
                        continue
                if MAX_ITEMS is not None and submitted >= MAX_ITEMS:
                    break
                outstanding.append(ex.submit(process_one_line, line))
                submitted += 1
                if len(outstanding) >= NUM_WORKERS * BUFFER_FACTOR:
                    maybe_drain(NUM_WORKERS)
            while outstanding:
                maybe_drain(len(outstanding))
        pbar.close()

    for sp in sorted(shard_paths):
        print(sp)
    print(f"Processed: {processed}, Skipped (empty/filtered): {skipped}, Errors: {errors}")
    if skip_ids_path:
        print(f"Skipped by exclusion list: {skipped_by_exclusion}")


if __name__ == "__main__":
    main()
