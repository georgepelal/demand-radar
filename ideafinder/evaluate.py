"""`python -m ideafinder evaluate --models a,b`: score Ollama models against a labelled sample.

eval/labels.jsonl holds excerpts labelled by Claude: 250 from discussions (100 random from HN, 150 more aimed at the
rare cases: willingness to pay, people building a solution, hardware, plus Stack Exchange and Bluesky) and ~50 low-star
App Store / WordPress.org reviews. The worked examples shown to the model are marked "example" and not scored; the
rest ("eval") are. Need precision/recall on the reviews is also printed separately.
Labels from a stronger model are careful but not ground truth: treat differences of a few points as noise.

A model spec is either "model" (both stages) or "stage1model+stage2model", e.g. "qwen2.5:7b+gemma4:12b".
Every run saves its answers to eval/runs/ and a summary line to eval/history.csv.
After correcting labels, `evaluate --rescore` scores the saved answers again without asking any model.
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import llm
from .classify import BUILDABLE_TYPES, SCHEMA_VERSION, check_need, classify_one
from .reviews import REVIEW_SOURCES
from .util import log

DEFAULT_LABELS = Path("eval") / "labels.jsonl"
FIELDS = [("solution_type", "type"), ("solo_buildable", "build"), ("speaker", "speaker")]


def _main_group(d: dict) -> bool:
    """What analyze ranks: a need, solved by software/service, from someone who isn't already building it."""
    return (bool(d.get("is_need")) and d.get("solution_type") in BUILDABLE_TYPES
            and d.get("speaker") != "building_solution")


def _is_review(it: dict) -> bool:
    return it["item_id"].split(":")[0] in REVIEW_SOURCES


def _explicit_pay(d: dict) -> bool:
    return bool(d.get("is_need")) and d.get("willingness_to_pay") == "explicit"


def _prf(pairs: list) -> tuple:
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _fmt(t: tuple) -> str:
    return f"{t[0]:.2f}/{t[1]:.2f}/{t[2]:.2f}"


def _predict(spec: str, items: list, workers: int, examples: bool) -> tuple:
    m1, _, m2 = spec.partition("+")
    m2 = m2 or m1
    for m in {m1, m2}:
        llm.ensure_model(m)

    def one(it):
        if not check_need(m1, it["title"], it["snippet"], examples):
            return {"stage1": False, "is_need": False}
        res = classify_one(m2, it["title"], it["snippet"], examples)
        return {"stage1": True, **res}

    preds: list = [None] * len(items)
    errors = 0
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=workers)
    futures = {ex.submit(one, it): i for i, it in enumerate(items)}
    try:
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                preds[futures[fut]] = fut.result()
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log(f"  {spec}: request failed: {e}")
            if n % 40 == 0 or n == len(items):
                log(f"  {spec}: {n}/{len(items)}")
    except KeyboardInterrupt:
        os._exit(130)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return preds, (time.time() - t0) / len(items), errors


def _confusion(pairs: list, title: str) -> None:
    c = Counter(pairs)
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    print(f"  {title} (rows = labels, columns = model):")
    print("    " + " " * 18 + "".join(f"{lab[:10]:>11}" for lab in labels))
    for g in labels:
        print(f"    {g[:18]:<18}" + "".join(f"{c.get((g, p), 0):>11}" for p in labels))


def run_evaluate(models: list, labels_path: str | Path = DEFAULT_LABELS, limit: int | None = None,
                 workers: int = 2, show_errors: bool = False, examples: bool = True) -> None:
    path = Path(labels_path)
    if not path.exists():
        raise SystemExit(f"labels file not found: {path}")
    items = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    items = [it for it in items if it.get("split", "eval") == "eval"]
    if limit:
        items = items[:limit]
    log(f"evaluate: {len(items)} labelled excerpts ({sum(it['gold']['is_need'] for it in items)} real needs, "
        f"{sum(_main_group(it['gold']) for it in items)} software ideas, "
        f"{sum(_explicit_pay(it['gold']) for it in items)} saying they'd pay; "
        f"{sum(_is_review(it) for it in items)} of them reviews)")
    runs_dir = path.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    results = []
    for spec in models:
        preds, secs, errors = _predict(spec, items, workers, examples)
        with open(runs_dir / f"{stamp}_{spec.replace(':', '-').replace('+', '_then_')}.jsonl", "w",
                  encoding="utf-8") as f:
            for it, p in zip(items, preds):
                f.write(json.dumps({"item_id": it["item_id"], "gold": it["gold"], "pred": p}, ensure_ascii=False)
                        + "\n")
        ok = [(it, it["gold"], p) for it, p in zip(items, preds) if p is not None]
        r = _score(spec, ok, secs, errors)
        both = [(g, p) for _, g, p in ok if g["is_need"] and p["is_need"]]
        results.append(r)
        if show_errors:
            print(f"\nMistakes by {spec}:")
            for it, g, p in ok:
                kind = None
                if g["is_need"] != p["is_need"]:
                    kind = "missed need" if g["is_need"] else "false need"
                elif _main_group(g) != _main_group(p):
                    kind = (f"software idea? gold {g.get('solution_type')}/{g.get('speaker')}, "
                            f"model {p.get('solution_type')}/{p.get('speaker')}")
                if kind:
                    print(f"  [{kind}] {it['snippet'][:140]!r}")
            _confusion([(g.get("solution_type"), p.get("solution_type")) for g, p in both], "solution type")
            _confusion([(g.get("speaker"), p.get("speaker")) for g, p in both], "speaker")
            _confusion([(g.get("willingness_to_pay"), p.get("willingness_to_pay")) for g, p in both],
                       "willingness to pay")

    _print_table(results)

    hist = path.parent / "history.csv"
    new = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "spec", "examples", "schema", "items", "need_p", "need_r", "need_f1", "stage1_r",
                        "software_p", "software_r", "software_f1", "pay_p", "pay_r", "pay_f1", "type_acc",
                        "build_acc", "speaker_acc", "s_per_item", "errors"])
        for r in results:
            w.writerow([stamp, r["spec"], int(examples), SCHEMA_VERSION, len(items),
                        *[round(x, 3) for x in r["need"]], round(r["stage1"][1], 3),
                        *[round(x, 3) for x in r["software"]], *[round(x, 3) for x in r["pay"]],
                        *[round(r["acc"][k], 3) for k, _ in FIELDS], round(r["secs"], 2), r["errors"]])


def _score(spec: str, ok: list, secs: float, errors: int) -> dict:
    """ok = [(item, gold, prediction)] -> all the numbers of one row of the table."""
    r = {
        "spec": spec, "secs": secs, "errors": errors,
        "stage1": _prf([(g["is_need"], p["stage1"]) for _, g, p in ok]),
        "need": _prf([(g["is_need"], p["is_need"]) for _, g, p in ok]),
        "need_random": _prf([(g["is_need"], p["is_need"]) for it, g, p in ok
                             if it.get("sample", "random:hn") == "random:hn"]),
        "need_reviews": _prf([(g["is_need"], p["is_need"]) for it, g, p in ok if _is_review(it)]),
        "need_discussions": _prf([(g["is_need"], p["is_need"]) for it, g, p in ok if not _is_review(it)]),
        "n_reviews": sum(1 for it, _, _ in ok if _is_review(it)),
        "software": _prf([(_main_group(g), _main_group(p)) for _, g, p in ok]),
        "pay": _prf([(_explicit_pay(g), _explicit_pay(p)) for _, g, p in ok]),
    }
    both = [(g, p) for _, g, p in ok if g["is_need"] and p["is_need"]]
    r["acc"] = {k: (sum(1 for g, p in both if g.get(k) == p.get(k)) / len(both) if both else 0.0)
                for k, _ in FIELDS}
    return r


def rescore(labels_path: str | Path = DEFAULT_LABELS) -> None:
    """Score every saved run in eval/runs/ against the current labels, without calling any model."""
    path = Path(labels_path)
    items = {it["item_id"]: it for it in (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
             if it.get("split", "eval") == "eval"}
    known: dict = {}  # run file name -> its line in history.csv
    hist = path.parent / "history.csv"
    if hist.exists():
        with open(hist, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                known[f"{row['date']}_{row['spec'].replace(':', '-').replace('+', '_then_')}"] = row
    runs = sorted((path.parent / "runs").glob("*.jsonl"))
    if not runs:
        raise SystemExit(f"no saved runs in {path.parent / 'runs'}")
    revised = sum(1 for it in items.values() if it.get("revised"))
    log(f"rescore: {len(runs)} saved runs against {len(items)} current labels ({revised} corrected since)")
    results = []
    for run in runs:
        rows = [json.loads(line) for line in run.open(encoding="utf-8") if line.strip()]
        ok = [(items[r["item_id"]], items[r["item_id"]]["gold"], r["pred"]) for r in rows
              if r["item_id"] in items and r["pred"] is not None]
        # runs from before the review labels existed are still comparable on the discussions
        covers_reviews = any(_is_review(it) for it, _, _ in ok)
        expected = len(items) if covers_reviews else sum(1 for it in items.values() if not _is_review(it))
        if len(ok) < expected * 0.9:  # a --limit run: not comparable with the others
            continue
        row = known.get(run.stem, {})
        date, hm, name = run.stem.split("_", 2)
        label = (row.get("spec") or name.replace("_then_", "+")) + ("" if row.get("examples", "1") == "1" else
                                                                    " no-ex") + f" ({date[5:]} {hm})"
        results.append(_score(label, ok, float(row.get("s_per_item") or 0), int(row.get("errors") or 0)))
    _print_table(results)
    print("(scores of the saved answers; nothing was sent to a model, eval/history.csv is unchanged)")


def _print_table(results: list) -> None:
    print()
    w = max([30] + [len(r["spec"]) + 2 for r in results])
    head = (f"{'model (stage1+stage2)':<{w}}{'need P/R/F1':>16}{'stage1 recall':>15}{'software P/R/F1':>18}"
            f"{'pays P/R/F1':>16}" + "".join(f"{s:>9}" for _, s in FIELDS) + f"{'s/item':>8}{'errors':>7}")
    print(head)
    print("-" * len(head))
    for r in results:
        print(f"{r['spec']:<{w}}{_fmt(r['need']):>16}{r['stage1'][1]:>15.2f}{_fmt(r['software']):>18}"
              f"{_fmt(r['pay']):>16}" + "".join(f"{r['acc'][k]:>9.0%}" for k, _ in FIELDS)
              + f"{r['secs']:>8.1f}{r['errors']:>7}")
    if any(r["n_reviews"] for r in results):
        print("\nneed P/R/F1 on reviews only / on discussions only (the latter compares with runs before reviews):")
        for r in results:
            print(f"  {r['spec']:<{w}}reviews {_fmt(r['need_reviews'])} ({r['n_reviews']} items) · "
                  f"discussions {_fmt(r['need_discussions'])}")
    print("\nneed = is it a real need (on the random HN sample only: "
          + ", ".join(f"{r['spec']} {_fmt(r['need_random'])}" for r in results) + ")"
          "\nstage1 recall = share of real needs the quick first check lets through"
          "\nsoftware = need + software/service + not already being built by the author: what analyze ranks"
          "\npays = says they would pay · type/build/speaker = agreement where both sides call it a need"
          "\ns/item = wall time per excerpt")
