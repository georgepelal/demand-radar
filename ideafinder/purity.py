"""Check which excerpts of a cluster really describe the cluster's need, and evaluate that check.

Clustering groups excerpts by meaning, so "fees on invoicing apps" and "fees on a debt-payoff app" end up together.
After the LLM has named a cluster's need, this asks it, a batch of excerpts at a time, which ones are about that
same need. analyze keeps only those when it counts people, threads and "would pay".

`python -m ideafinder purity-eval` measures the check against eval/cluster_labels.jsonl (excerpts of real clusters
labelled on-topic / off-topic by Claude) and appends a line to eval/purity_history.csv.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from . import llm
from .util import log

BATCH = 10
DEFAULT_LABELS = Path(__file__).resolve().parent.parent / "eval" / "cluster_labels.jsonl"

PURITY_SYSTEM = """You check a group of excerpts from online discussions and app/plugin reviews that was formed
automatically around ONE specific need. For each numbered excerpt, decide whether it is evidence for THIS need:
the same problem, for the same kind of person, so that counting its author as "someone who has this need" is fair.

Answer false when the excerpt is about a different problem, even a related one in the same area. Examples:
- need "shared password manager for families"; excerpt about a two-factor authentication app losing codes -> false
- need "self-hosted photo backup"; excerpt asking for better photo editing filters -> false
- need "scheduling software for dog groomers"; excerpt from a pet owner looking for a groomer nearby -> false
- need "offline maps for hikers"; excerpt about a hiking app that removed offline downloads from the free plan -> true

Return one boolean per excerpt, in order."""


def _schema(n: int) -> dict:
    return {"type": "object",
            "properties": {"on_topic": {"type": "array", "items": {"type": "boolean"}, "minItems": n, "maxItems": n}},
            "required": ["on_topic"]}


def _prompt(title: str, problem: str, items: list) -> str:
    lines = [f"NEED: {title}", f"Description: {problem}", "", "Excerpts:"]
    for n, (prob, snippet) in enumerate(items, 1):
        lines.append(f"{n}. need stated: {prob or '-'}\n   text: {(snippet or '')[:260]}")
    return "\n".join(lines)


def check_batch(model: str, title: str, problem: str, items: list) -> list:
    """[True/False per item]; on a malformed answer, retries once per item; if that fails too, keeps it (True)."""
    out = llm.chat_json(model, PURITY_SYSTEM, _prompt(title, problem, items), _schema(len(items)),
                        temperature=0.0, max_tokens=256)
    flags = out.get("on_topic")
    if isinstance(flags, list) and len(flags) == len(items):
        return [bool(f) for f in flags]
    res = []
    for it in items:
        try:
            one = llm.chat_json(model, PURITY_SYSTEM, _prompt(title, problem, [it]), _schema(1),
                                temperature=0.0, max_tokens=32).get("on_topic") or [True]
            res.append(bool(one[0]))
        except llm.OllamaError:
            res.append(True)
    return res


def check_members(model: str, title: str, problem: str, items: list, batch: int = BATCH) -> list:
    """items: [(stated problem, snippet)] -> [bool]. A failed batch keeps its items (fail open)."""
    flags: list = []
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        try:
            flags += check_batch(model, title, problem, chunk)
        except llm.OllamaError as e:
            log(f"  topic check failed for {len(chunk)} excerpts ({str(e)[:80]}); keeping them")
            flags += [True] * len(chunk)
    return flags


# ---------------------------------------------------------------- evaluation
def run_purity_eval(models: list, labels_path: str | Path = DEFAULT_LABELS, show_errors: bool = False) -> None:
    path = Path(labels_path)
    labels = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_cluster: dict = defaultdict(list)
    for it in labels:
        by_cluster[(it["cluster"], it["cluster_problem"])].append(it)
    on = sum(it["on_topic"] for it in labels)
    print(f"{len(labels)} labelled excerpts in {len(by_cluster)} clusters; "
          f"purity before the check: {on}/{len(labels)} = {on / len(labels):.0%}\n")
    hist = path.parent / "purity_history.csv"
    for model in models:
        llm.ensure_model(model)
        t0 = time.time()
        tp = fp = fn = tn = 0
        rows = []
        errors = []
        for (title, problem), items in by_cluster.items():
            flags = check_members(model, title, problem, [(it["problem"], it["snippet"]) for it in items])
            ctp = sum(1 for it, f in zip(items, flags) if f and it["on_topic"])
            cfp = sum(1 for it, f in zip(items, flags) if f and not it["on_topic"])
            cfn = sum(1 for it, f in zip(items, flags) if not f and it["on_topic"])
            tp, fp, fn, tn = tp + ctp, fp + cfp, fn + cfn, tn + (len(items) - ctp - cfp - cfn)
            before = sum(it["on_topic"] for it in items) / len(items)
            after = ctp / (ctp + cfp) if ctp + cfp else 0.0
            rows.append((title, len(items), before, after, ctp + cfp))
            errors += [(title, it, f) for it, f in zip(items, flags) if f != it["on_topic"]]
        secs = time.time() - t0
        purity = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        acc = (tp + tn) / len(labels)
        print(f"== {model}  ({secs:.0f}s, {secs / len(labels):.2f}s per excerpt)")
        print(f"{'cluster':<62} {'n':>3} {'before':>7} {'after':>6} {'kept':>5}")
        for title, n, b, a, kept in rows:
            print(f"{title[:61]:<62} {n:>3} {b:>7.0%} {a:>6.0%} {kept:>5}")
        print(f"\npurity of kept excerpts: {purity:.0%} (was {on / len(labels):.0%})   "
              f"on-topic excerpts kept: {recall:.0%}   accuracy: {acc:.0%}\n")
        if show_errors:
            for title, it, f in errors:
                kind = "kept, but off-topic" if f else "dropped, but on-topic"
                print(f"  [{kind}] {title[:40]} | {it['problem'][:110]}")
            print()
        new = not hist.exists()
        with open(hist, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["date", "model", "items", "purity_before", "purity_after", "recall", "accuracy",
                            "s_per_item"])
            w.writerow([time.strftime("%Y-%m-%d_%H%M"), model, len(labels), round(on / len(labels), 3),
                        round(purity, 3), round(recall, 3), round(acc, 3), round(secs / len(labels), 2)])
