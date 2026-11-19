"""Command line: python -m ideafinder <command> [options]"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

# macOS's built-in Python is linked against LibreSSL, which makes urllib3 print a harmless warning.
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

from . import db as dbmod  # noqa: E402
from . import llm  # noqa: E402
from .util import log  # noqa: E402


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["auto", "llm", "raw"], default="auto",
                   help="llm = only LLM-confirmed needs; raw = all regex excerpts; auto = llm once >=100 needs")
    p.add_argument("--embed", choices=["auto", "ollama", "tfidf"], default="auto")
    p.add_argument("--embed-model", default=llm.DEFAULT_EMBED)
    p.add_argument("--model", default=llm.DEFAULT_LLM, help="Ollama chat model (default: %(default)s)")
    p.add_argument("--summary-model", help="model for the cluster descriptions (default: --model); "
                                           "only ~25 calls, so a bigger, slower model is fine here")
    p.add_argument("--min-cluster-size", type=int, help="default: automatic (5-12 depending on data size)")
    p.add_argument("--top", type=int, default=40,
                   help="needs to describe in detail and put in brief.md (default: %(default)s)")
    p.add_argument("--no-summary", action="store_true", help="don't ask the LLM to describe clusters")
    p.add_argument("--no-topic-check", action="store_true",
                   help="count every excerpt of a described cluster, without checking it is about the same need "
                        "(faster, but the numbers mix related problems)")
    p.add_argument("--out", default="reports")
    p.add_argument("--seed", type=int, default=42)


def _add_collect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="hn,se,bsky,as,wp",
                   help="comma-separated: hn, se (Stack Exchange), bsky (Bluesky), as (App Store reviews), "
                        "wp (WordPress.org plugin reviews); default: all")
    p.add_argument("--days", type=int, default=1095, help="Hacker News and Bluesky: how far back (default: 3 years)")
    p.add_argument("--se-days", type=int, default=2190,
                   help="Stack Exchange: how far back (default: 6 years; it has been much quieter since 2023)")
    p.add_argument("--se-sites", help="comma-separated Stack Exchange sites (default: built-in list)")
    p.add_argument("--bsky-per-month", type=int, default=15,
                   help="Bluesky: posts kept per phrase and month (default: %(default)s)")
    p.add_argument("--phrases-file", help="Hacker News: one search phrase per line (default: built-in list)")
    p.add_argument("--app-genres", help="App Store: comma-separated genre ids (default: 10 software genres, "
                                        "see collect_as.py)")
    p.add_argument("--apps-per-genre", type=int, default=15,
                   help="App Store: top-grossing apps per genre (default: %(default)s)")
    p.add_argument("--review-pages", type=int, default=4,
                   help="App Store: pages of 50 'most helpful' reviews per app (default: %(default)s, max 10)")
    p.add_argument("--wp-plugins", type=int, default=100,
                   help="WordPress.org: how many popular plugins (default: %(default)s)")
    p.add_argument("--per-app", type=int, default=8,
                   help="App Store / WordPress.org: most informative reviews kept per app or plugin "
                        "(default: %(default)s)")


def _collect(con, a) -> None:
    sources = {s.strip() for s in a.source.split(",") if s.strip()}
    if "hn" in sources:
        from .collect_hn import collect, load_phrases
        collect(con, days=a.days, phrases=load_phrases(a.phrases_file))
    if "se" in sources:
        from .collect_se import collect_se
        sites = [s.strip() for s in a.se_sites.split(",")] if a.se_sites else None
        collect_se(con, days=a.se_days, sites=sites)
    if "bsky" in sources:
        from .collect_bsky import collect_bsky
        collect_bsky(con, days=a.days, per_month=a.bsky_per_month)
    if "as" in sources:
        from .collect_as import collect_as
        genres = [int(g) for g in a.app_genres.split(",") if g.strip()] if a.app_genres else None
        collect_as(con, genres=genres, apps_per_genre=a.apps_per_genre, pages=min(a.review_pages, 10),
                   per_app=a.per_app)
    if "wp" in sources:
        from .collect_wp import collect_wp
        collect_wp(con, n_plugins=a.wp_plugins, per_plugin=a.per_app)


def _analyze(con, a, no_llm: bool = False):
    from .analyze import run_analyze
    return run_analyze(
        con, mode="raw" if no_llm else a.mode, embed="tfidf" if no_llm else a.embed,
        embed_model=a.embed_model, llm_model=a.summary_model or a.model, min_cluster_size=a.min_cluster_size,
        top=a.top, summary=not (no_llm or a.no_summary), out_dir=a.out, seed=a.seed,
        purity=not getattr(a, "no_topic_check", False),
    )


def stats(con) -> None:
    from .classify import SCHEMA_VERSION
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    by_source = con.execute("SELECT source, COUNT(*) FROM items GROUP BY 1 ORDER BY 2 DESC").fetchall()
    print(f"items:            {q('SELECT COUNT(*) FROM items')} ("
          + ", ".join(f"{s} {n}" for s, n in by_source) + ")")
    print(f"pain excerpts:    {q('SELECT COUNT(*) FROM pains')}")
    print(f"  with money:     {q('SELECT COUNT(*) FROM pains WHERE money = 1')}")
    print(f"  LLM-checked:    {q('SELECT COUNT(*) FROM pains WHERE llm_model IS NOT NULL')}")
    print(f"  LLM-confirmed:  {q('SELECT COUNT(*) FROM pains WHERE is_pain = 1')}")
    cur = f"SELECT COUNT(*) FROM pains WHERE is_pain = 1 AND schema_version >= {SCHEMA_VERSION}"
    software = cur + " AND solution_type IN ('software', 'service') AND speaker != 'building_solution'"
    print(f"    with current fields: {q(cur)} (software/service ideas: {q(software)}); "
          f"waiting for stage 2: {q('SELECT COUNT(*) FROM pains WHERE is_pain = 1 AND COALESCE(schema_version, 1) < ' + str(SCHEMA_VERSION))}")
    counts: dict = {}
    for (p,) in con.execute("SELECT patterns FROM pains"):
        for name in json.loads(p):
            counts[name] = counts.get(name, 0) + 1
    for name, n in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"    {name:<16}{n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ideafinder",
                                 description="Find business ideas from unmet needs in online discussions.")
    ap.add_argument("--db", default=str(dbmod.DEFAULT_DB), help="SQLite file (default: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="1. download matching posts and reviews (Hacker News, Stack Exchange, "
                                                          "Bluesky, App Store, WordPress.org)")
    _add_collect_args(c)

    sub.add_parser("extract", help="2. find pain-point excerpts with regex")

    k = sub.add_parser("classify", help="3. check excerpts with a local LLM (Ollama)")
    k.add_argument("--model", default=llm.DEFAULT_LLM)
    k.add_argument("--limit", type=int, help="max excerpts to check in this run")
    k.add_argument("--workers", type=int, default=2)
    k.add_argument("--include-weak", action="store_true",
                   help="also check excerpts that only vent ('so frustrating', 'too expensive')")
    k.add_argument("--recheck-rejected", action="store_true",
                   help="also re-check excerpts an older version rejected (slower; finds needs it missed)")
    k.add_argument("--detail-model", help="model for stage 2, the detailed fields (default: --model)")
    k.add_argument("--no-examples", action="store_true", help="don't show the worked examples to the model")
    k.add_argument("--dry-run", action="store_true", help="only show how many excerpts would be checked and how long")

    a = sub.add_parser("analyze", help="4. cluster, score and write a report")
    _add_analyze_args(a)

    r = sub.add_parser("run", help="all steps: collect -> extract -> classify -> analyze")
    _add_collect_args(r)
    r.add_argument("--no-llm", action="store_true", help="no Ollama at all: regex + TF-IDF only")
    r.add_argument("--limit", type=int, help="max excerpts to classify with the LLM in this run")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--include-weak", action="store_true")
    r.add_argument("--detail-model", help="model for classify stage 2 (default: --model)")
    _add_analyze_args(r)

    sub.add_parser("stats", help="show what is in the database")

    e = sub.add_parser("evaluate", help="score Ollama models against the labelled sample in eval/labels.jsonl")
    e.add_argument("--models", default=llm.DEFAULT_LLM, help="comma-separated, e.g. qwen2.5:7b,gemma4:12b")
    e.add_argument("--labels", default="eval/labels.jsonl")
    e.add_argument("--limit", type=int)
    e.add_argument("--workers", type=int, default=2)
    e.add_argument("--show-errors", action="store_true", help="print the excerpts each model got wrong")
    e.add_argument("--no-examples", action="store_true", help="don't show the worked examples to the model")
    e.add_argument("--rescore", action="store_true",
                   help="score the saved answers in eval/runs/ against the current labels (no model calls)")

    pe = sub.add_parser("purity-eval", help="measure the cluster topic check against eval/cluster_labels.jsonl")
    pe.add_argument("--models", default=llm.DEFAULT_LLM, help="comma-separated, e.g. qwen2.5:7b,gemma4:12b")
    pe.add_argument("--labels", default=None, help="default: eval/cluster_labels.jsonl")
    pe.add_argument("--show-errors", action="store_true", help="print the excerpts the model got wrong")

    m = sub.add_parser("compete", help="5. for the top ideas of a report: what already exists, is there a gap?")
    m.add_argument("--model", default=llm.DEFAULT_LLM)
    m.add_argument("--top", type=int, default=15, help="how many top ideas to check (default: %(default)s)")
    m.add_argument("--report", help="report folder (default: the latest one in --out)")
    m.add_argument("--out", default="reports")

    d = sub.add_parser("doctor", help="check that an Ollama model works with this tool, and how fast")
    d.add_argument("--model", default=llm.DEFAULT_LLM)
    d.add_argument("--timeout", type=int, default=120, help="seconds per test (default: %(default)s)")

    a = ap.parse_args(argv)
    con = dbmod.connect(a.db)
    try:
        if a.cmd == "collect":
            _collect(con, a)
        elif a.cmd == "extract":
            from .extract import run_extract
            run_extract(con)
        elif a.cmd == "classify":
            from .classify import run_classify
            run_classify(con, model=a.model, limit=a.limit, workers=a.workers, include_weak=a.include_weak,
                         recheck_rejected=a.recheck_rejected, detail_model=a.detail_model,
                         examples=not a.no_examples, dry_run=a.dry_run)
        elif a.cmd == "analyze":
            _analyze(con, a)
        elif a.cmd == "run":
            from .classify import run_classify
            from .extract import run_extract
            if not a.no_llm:
                llm.ensure_model(a.model)  # fail early, before the long download
            _collect(con, a)
            run_extract(con)
            if not a.no_llm:
                run_classify(con, model=a.model, limit=a.limit, workers=a.workers, include_weak=a.include_weak,
                             detail_model=a.detail_model)
            _analyze(con, a, no_llm=a.no_llm)
        elif a.cmd == "stats":
            stats(con)
        elif a.cmd == "evaluate":
            from .evaluate import rescore, run_evaluate
            if a.rescore:
                rescore(a.labels)
                return 0
            run_evaluate([m.strip() for m in a.models.split(",") if m.strip()], a.labels, a.limit, a.workers,
                         a.show_errors, examples=not a.no_examples)
        elif a.cmd == "purity-eval":
            from .purity import DEFAULT_LABELS, run_purity_eval
            run_purity_eval([m.strip() for m in a.models.split(",") if m.strip()], a.labels or DEFAULT_LABELS,
                            a.show_errors)
        elif a.cmd == "compete":
            from .compete import run_compete
            run_compete(con, report=a.report, out_dir=a.out, top=a.top, model=a.model)
        elif a.cmd == "doctor":
            from .doctor import run_doctor
            run_doctor(a.model, a.timeout)
    except llm.OllamaError as e:
        log(f"Ollama error: {e}")
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
