#!/usr/bin/env python3
"""
Run Kev (jaredpalmer/kev-9b, or any Kev LoRA checkpoint) locally with plain PyTorch, no server.

Kev is a decision model: you give it a document (the "state") and typed questions, and it returns a probability
distribution over each question's options in one forward pass. It does not generate text.

This mirrors the author's loader (github.com/jaredpalmer/kev, kev/checkpoint.py + kev/model.py + kev/api.py):
  - base model  Qwen/Qwen3.5-9B-Base at the revision pinned in head.pt (backbone only, no vocab head)
  - LoRA        adapter_model.safetensors, merged into the base weights
  - head        the pointer head + calibration temperature stored in head.pt

Requirements:
    pip install "torch>=2.6,<2.9" "transformers>=5.17,<6" "peft>=0.21" huggingface_hub

Usage:
    python run_kev.py                                  # built-in example request
    python run_kev.py request.json [more.json ...]     # TypeSafe-style requests (format below); a file may hold a list
    python run_kev.py requests.json > results.json     # answers as JSON on stdout, progress on stderr
    python run_kev.py requests.json --out results.jsonl   # big batches: one line per result, resumable after a crash
    python run_kev.py request.json --device cpu --dtype bf16

Request format (same as the TypeSafe /v1/systemone body):
    {
      "state": "any text, or a JSON object / array",
      "questions": {
        "is_refund": {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
        "topic":     {"type": "choice", "instructions": "What is this about?",
                      "criteria": {"billing": "charges, invoices", "shipping": null, "other": null}},
        "anger":     {"type": "score",  "instructions": "How angry is the customer?",
                      "criteria": ["calm", "annoyed", "angry", "furious"]}
      }
    }

Memory: ~19 GB of RAM / VRAM in bf16, ~38 GB in fp32 (the exact path the model card's numbers use).
"""
import argparse
import copy
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rarely used Qwen special tokens Kev repurposes as delimiters: <state>, <q>, <opt>, </opt>, <decide>
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------

class PointerHead(nn.Module):
    """Scores each option's </opt> hidden state against the question's <decide> hidden state."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K, d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale / self.temperature


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint(ckpt):
    if os.path.isdir(ckpt):
        return ckpt
    from huggingface_hub import snapshot_download
    repo, _, revision = ckpt.partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


def load_kev(checkpoint, device, dtype, base_override=None, temperature=None):
    """-> (tokenizer, backbone, pointer head)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_checkpoint(checkpoint)
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        sys.exit(f"{path} has no adapter_config.json: this script runs Kev LoRA checkpoints (kev-9b, kev-8b, kev-4b, kev-0.8b).")
    meta = torch.load(os.path.join(path, "head.pt"), map_location="cpu", weights_only=True)
    with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    base = base_override or meta["base"]
    revision = None if base_override else meta.get("base_revision")
    print(f"Base model : {base}" + (f" @ {revision}" if revision else ""), file=sys.stderr)
    print(f"Adapter    : {checkpoint}", file=sys.stderr)
    print(f"Device     : {device} ({str(dtype).removeprefix('torch.')})", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    # SDPA on CUDA, eager elsewhere: the same choice the author's code makes
    attn = "sdpa" if device.startswith("cuda") else "eager"
    lm = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=dtype, attn_implementation=attn).model  # backbone only
    lm = PeftModel.from_pretrained(lm, path)
    if not adapter_cfg.get("trainable_token_indices"):
        lm = lm.merge_and_unload()            # fold the LoRA into the weights: faster, and closer to the fp32 numbers in bf16
    lm = lm.to(device=device, dtype=dtype).eval()

    head = PointerHead(lm.config.hidden_size, dp=meta.get("head_dim", 256))
    head.load_state_dict(meta["head"])
    head.temperature = meta.get("temperature", 1.0) if temperature is None else temperature
    head = head.to(device).eval()             # the head always runs in fp32
    print(f"Calibration: T = {head.temperature:.2f}", file=sys.stderr)
    return tok, lm, head


# ----------------------------------------------------------------------------------------------------------------------
# Encoding: [<state> state...]  then per question  [<q> instructions <opt> option </opt> ... <decide>]
# ----------------------------------------------------------------------------------------------------------------------

def user_tokens(tok, text):
    """Tokenize caller text so it can never produce Kev's delimiter tokens."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec):
    """-> (state ids, rows): one row per question, with the offsets of its <decide> token and each option's </opt>."""
    state_id, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
    S = [state_id] + user_tokens(tok, rec["state"])
    rows = []
    for q in rec["questions"]:
        ids = [q_id] + user_tokens(tok, q["instr"])
        opts = []
        for o in q["options"]:
            ids += [o_id] + user_tokens(tok, o) + [c_id]
            opts.append(len(ids) - 1)
        ids.append(d_id)
        rows.append({"ids": ids, "decide": len(ids) - 1, "opts": opts})
    return S, rows


@torch.no_grad()
def score(tok, lm, head, rec, device, reuse_state=True):
    """Probabilities per question. Each question runs as its own causal row (state + its branch), so questions never
    see each other. With reuse_state the state runs once and every question continues from a copy of its cache;
    otherwise the state is recomputed per question (slower, same result)."""
    from transformers import DynamicCache
    S, rows = encode(tok, rec)
    Ls = len(S)
    t = lambda xs: torch.tensor([xs], device=device)
    if reuse_state:
        state_cache = lm(input_ids=t(S), position_ids=t(list(range(Ls))),
                         past_key_values=DynamicCache(config=lm.config), use_cache=True).past_key_values
    probs = []
    for r in rows:
        if reuse_state:
            out = lm(input_ids=t(r["ids"]), position_ids=t(list(range(Ls, Ls + len(r["ids"])))),
                     past_key_values=copy.deepcopy(state_cache), use_cache=True)
            h = out.last_hidden_state[0].float()
        else:
            ids = S + r["ids"]
            h = lm(input_ids=t(ids), position_ids=t(list(range(len(ids))))).last_hidden_state[0].float()[Ls:]
        logits = head(h[r["decide"]], h[torch.tensor(r["opts"], device=device)])
        probs.append(F.softmax(logits, -1).cpu().tolist())
    return probs, Ls


# ----------------------------------------------------------------------------------------------------------------------
# TypeSafe-style request / response (ported from kev/api.py)
# ----------------------------------------------------------------------------------------------------------------------

def render(v, indent=0):
    """Flatten str | object | array into the text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")


def with_date_facts(state):
    """Optional preprocessing (the model card's KEV_DATE_FACTS=1): state every pair of dates as an explicit day count."""
    found = []
    for m in _DATE.finditer(render(state)):
        raw = m.group(0)
        try: d = datetime.strptime(raw, "%B %d, %Y") if "," in raw else datetime.strptime(raw, "%Y-%m-%d")
        except ValueError: continue
        if raw not in [r for r, _ in found]: found.append((raw, d))
    facts = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            n = (found[j][1] - found[i][1]).days
            facts.append(f"{found[j][0]} is {abs(n)} day{'s' if abs(n) != 1 else ''} {'after' if n > 0 else 'before'} {found[i][0]}."
                         if n else f"{found[j][0]} is the same day as {found[i][0]}.")
    facts = " ".join(facts)
    if not facts: return state
    if isinstance(state, dict): return {**state, "date_facts": facts}
    if isinstance(state, list): return state + [{"date_facts": facts}]
    return f"{state}\n\ndate_facts: {facts}"


def to_record(req):
    if "state" not in req or not req.get("questions"):
        raise ValueError("a request needs 'state' and a non-empty 'questions' object")
    qs, meta = [], []
    for qid, q in req["questions"].items():
        kind, crit = q.get("type"), q.get("criteria")
        if kind == "noul":
            c = crit or {}
            opts, keys = [option_text("no", c.get("false")), option_text("yes", c.get("true"))], ["false", "true"]
        elif kind == "choice":
            if not isinstance(crit, dict) or not crit: raise ValueError(f"{qid}: choice needs a 'criteria' object of options")
            opts, keys = [option_text(k, v) for k, v in crit.items()], list(crit)
        elif kind == "score":
            if not isinstance(crit, list) or not crit: raise ValueError(f"{qid}: score needs a 'criteria' list of levels")
            opts, keys = [render(x) for x in crit], [str(i) for i in range(len(crit))]
        else:
            raise ValueError(f"{qid}: type must be noul, choice or score")
        qs.append({"instr": render(q.get("instructions")), "options": opts})
        meta.append({"id": qid, "type": kind, "keys": keys, "legend": dict(zip(keys, opts)) if kind == "score" else None})
    return {"state": render(req["state"]), "questions": qs}, meta


def _normalize(p):
    t = sum(p)
    return [1 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def choice_confidence(p):
    K = len(p)
    return 1.0 if K == 1 else (max(_normalize(p)) - 1 / K) / (1 - 1 / K)


def score_confidence(p):
    L = len(p)
    if L == 1: return 1.0
    p = _normalize(p); mode = max(range(L), key=p.__getitem__)
    D = sum(abs(i - (L - 1) / 2) for i in range(L)) / L
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / D)


def to_answers(probs, meta):
    r = lambda x: round(float(x), 4)
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": r(p[1])}
        elif m["type"] == "choice":
            out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(len(p)), key=p.__getitem__)],
                            "confidence": r(choice_confidence(p)), "probabilities": {k: r(v) for k, v in zip(m["keys"], p)}}
        else:
            out[m["id"]] = {"type": "score", "score": r(sum(i * pi for i, pi in enumerate(p))), "legend": m["legend"],
                            "probabilities": {str(i): r(v) for i, v in enumerate(p)}, "confidence": r(score_confidence(p))}
    return out


EXAMPLE = {
    "state": ("Ticket #4821 from Dana Okafor, received 2026-10-20:\n"
              "Hi, I was charged twice for my October subscription (two R249 charges on 2026-10-01). "
              "I've emailed twice already with no reply. Please refund the duplicate charge today or I'm cancelling."),
    "questions": {
        "wants_refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        "queue": {"type": "choice", "instructions": "Which team should handle this ticket?",
                  "criteria": {"billing": "charges, refunds, invoices", "technical": "bugs, outages, login problems",
                               "sales": "upgrades, pricing questions", "other": None}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["calm", "mildly annoyed", "frustrated", "very angry"]},
    },
}


# ----------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run a Kev decision model locally.", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Example request format: see the docstring at the top of this file.")
    ap.add_argument("requests", nargs="*", help="JSON request files (TypeSafe /v1/systemone format). None = built-in example.")
    ap.add_argument("--checkpoint", default="jaredpalmer/kev-9b", help="Hub id (repo or repo@revision) or local directory")
    ap.add_argument("--base", default=None, help="local copy of the base model, overriding the one head.pt pins")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="bf16 ~19 GB, fp32 ~38 GB (exact path)")
    ap.add_argument("--raw", action="store_true", help="raw logits (temperature 1.0) instead of the calibrated probabilities")
    ap.add_argument("--date-facts", action="store_true", help="append explicit day counts between dates found in the state")
    ap.add_argument("--no-state-cache", action="store_true", help="recompute the state for every question (slower, same result)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for torch (default: torch's choice)")
    ap.add_argument("--out", default=None, help="write results as JSON lines to this file as they finish; rerunning skips ids already there")
    args = ap.parse_args()

    if args.threads: torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    requests = []
    for path in args.requests:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # a file holds one request object, or a list of them (each may carry an "id" to label its output)
        for i, req in enumerate(data if isinstance(data, list) else [data]):
            requests.append((req.get("id") or (f"{path}[{i}]" if isinstance(data, list) else path), req))
    if not requests:
        requests = [("built-in example", EXAMPLE)]

    # --out: one JSON line per result, written as it finishes; ids already in the file are skipped, so a rerun resumes
    done, out = set(), None
    if args.out:
        if os.path.exists(args.out):
            with open(args.out, "rb+") as f:              # drop a last line a crash cut off, so appends start on a fresh line
                data = f.read()
                if data and not data.endswith(b"\n"):
                    f.truncate(data.rfind(b"\n") + 1)
            with open(args.out, encoding="utf-8") as f:
                for line in f:
                    try: done.add(json.loads(line)["id"])
                    except (ValueError, KeyError): pass   # a line cut off by a crash is redone
        ids = [name for name, _ in requests]
        if len(set(ids)) != len(ids):
            sys.exit("--out needs a unique \"id\" on every request, so a rerun knows which ones are finished")
        todo = [(n, r) for n, r in requests if n not in done]
        print(f"{len(requests) - len(todo)} of {len(requests)} already in {args.out}; {len(todo)} to run", file=sys.stderr)
        requests = todo
        if not requests: return
        out = open(args.out, "a", encoding="utf-8")

    t0 = time.perf_counter()
    tok, lm, head = load_kev(args.checkpoint, device, dtype, args.base, 1.0 if args.raw else None)
    print(f"Loaded in {time.perf_counter() - t0:.0f} s\n", file=sys.stderr)

    results, failed, t_start = [], 0, time.perf_counter()
    for i, (name, req) in enumerate(requests, 1):
        try:
            if args.date_facts:
                req = {**req, "state": with_date_facts(req["state"])}
            rec, meta = to_record(req)
            t1 = time.perf_counter()
            probs, n_state = score(tok, lm, head, rec, device, reuse_state=not args.no_state_cache)
            dt = time.perf_counter() - t1
        except Exception as e:
            if not out: raise
            failed += 1                                   # long batch run: log it and keep going; a rerun retries it
            print(f"!!! {name}: {type(e).__name__}: {e}", file=sys.stderr)
            if device.startswith("cuda"): torch.cuda.empty_cache()
            continue
        result = {"id": name, "answers": to_answers(probs, meta)}
        if out:
            out.write(json.dumps(result, ensure_ascii=False) + "\n"); out.flush()
            eta = (time.perf_counter() - t_start) / i * (len(requests) - i)
            print(f"[{i}/{len(requests)}] {name}: {n_state} state tokens, {dt:.2f} s, ETA {eta / 60:.0f} min", file=sys.stderr)
        else:
            print(f"--- {name}: {len(meta)} question(s), {n_state} state tokens, {dt:.2f} s", file=sys.stderr)
            results.append(result)

    if out:
        out.close()
        print(f"Done: {len(requests) - failed} written to {args.out}" + (f", {failed} failed (rerun to retry)" if failed else ""), file=sys.stderr)
    else:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()