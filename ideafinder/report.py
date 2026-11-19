"""Write report.md, brief.md and CSV/JSON files for one analysis run.

report.md  everything, for browsing
brief.md   the evidence dossier for the final review: the top needs plus the strongest needs of each smaller source,
           with numbers, quotes and links, sized to be read in one go by a session with web search
"""
from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .util import log

SOURCE_NAMES = {"hn": "Hacker News", "se": "Stack Exchange", "bsky": "Bluesky", "as": "App Store",
                "wp": "WordPress.org"}
SHORT = {"hn": "HN", "se": "SE", "bsky": "Bluesky", "as": "AppStore", "wp": "WP"}


def _date(ts) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else "?"


def _cell(s) -> str:
    return str(s).replace("|", "/").replace("\n", " ")


def _title(c) -> str:
    return (c.summary or {}).get("title") or ", ".join(c.keywords[:3]) or f"cluster {c.label}"


def _short(text: str, n: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _sources(meta) -> str:
    return ", ".join(f"{SOURCE_NAMES.get(s, s)} {n}" for s, n in sorted(meta.get("sources", {}).items(),
                                                                       key=lambda t: -t[1])) or "?"


def _src_mix(c) -> str:
    return " · ".join(f"{SHORT.get(s, s)} {n}" for s, n in c.sources.items())


def _pay(c) -> str:
    return f"{c.pay_explicit:.0%}" + (f" (+{c.pay_implied:.0%})" if c.pay_implied >= 0.005 else "")


def _overview(clusters, detailed: bool) -> list:
    L = ["| # | Need | Score | People | Threads | Sources | Would pay | Growth |" + (" Exists |" if detailed else ""),
         "|---|---|---|---|---|---|---|---|" + ("---|" if detailed else "")]
    for rank, c in enumerate(clusters, 1):
        mixed = " ⚠" if c.mixed else ""
        row = (f"| {rank} | {_cell(_title(c))}{mixed} | {c.final:.1f} | {c.authors} | {c.threads} | {_src_mix(c)} | "
               f"{_pay(c)} | ×{c.growth:.1f} |")
        if detailed:
            exists = len(c.show_hn) + len(c.builders)
            row += f" {exists or ''} |"
        L.append(row)
    return L


def _summary_lines(s: dict) -> list:
    L = []
    for key, label in (("problem", "Problem"), ("who", "Who has it"), ("wants", "What they ask for"),
                       ("current_solutions", "What they use today")):
        if s.get(key):
            L.append(f"- **{label}:** {s[key]}")
    if s.get("verify"):
        L.append(f"- **Check first:** {', '.join(s['verify'])}")
    return L


def _exists_lines(c, rows, builders: int = 3) -> list:
    L = []
    if c.reviewed:
        L.append("- Low-star reviews of: " + ", ".join(f"{p} ({k}×)" if k > 1 else p for p, k in c.reviewed))
    reviewed = {p for p, _ in c.reviewed}
    mentioned = [(p, k) for p, k in c.products if p not in reviewed]  # reviewers name the reviewed product too
    if mentioned:
        L.append(f"- Mentioned: {', '.join(f'{p} ({k}×)' if k > 1 else p for p, k in mentioned)}")
    for t, url in c.show_hn:
        L.append(f"- Came up under a launch: [{_cell(t)}]({url})")
    seen = set()
    for i in c.builders:
        r = rows[i]
        if r["snippet"] in seen or len(seen) >= builders:
            continue
        seen.add(r["snippet"])
        L.append(f"- Someone building it: “{_short(r['snippet'], 160)}” · [link]({r['url']})")
    return L


def _quote(r, n: int) -> str:
    where = SOURCE_NAMES.get(r["source"], r["source"])
    if r.get("reviewed"):
        where += f" review of {r['reviewed']} ({r['stars']}★)"
    return f"> {_short(r['snippet'], n)}  \n> — {where}, {r['author']}, {_date(r['created_at'])} · [link]({r['url']})"


def render_markdown(main, other, rows, meta) -> str:
    top = meta["top"]
    g = meta["groups"]
    L = [f"# Idea Finder report — {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    L.append(
        f"Sources used: {_sources(meta)} · {meta['excerpts_total']} excerpts found, {meta['llm_checked']} checked by "
        f"LLM · mode `{meta['mode']}` · embeddings `{meta['embedding']}`"
    )
    if meta["detailed"]:
        L.append(f"  \nNeeds used: **{g.get('main', 0)} that software or a service could solve**, "
                 f"{g.get('other', 0)} other (hardware, content, policy), "
                 f"{g.get('builder', 0)} from people already building a solution")
    L.append(f"  \nClustering {meta['clustering']}")
    L.append("")
    L.append(
        "> **Score** (evidence only) = log(1+people) × log(1+threads) × (1 + 1.5 × share who would pay + 0.5 × share "
        "with implied costs) × growth (0.5–2, most recent third of each source's period vs. expected) × "
        "(1 + 0.3 per extra community). For App Store / WordPress.org reviews a thread is one app or plugin, so "
        "threads = how many different products share the complaint, and reviews don't count towards growth. "
        "⚠ = the excerpts don't share one problem, so the numbers mix needs (×0.6). "
        "**Exists** = Show HN launches + people building a solution in these threads. "
        "These are leads to investigate, not validated ideas: see brief.md for the final review."
    )
    title = "Software / service needs" if meta["detailed"] else "Overview"
    L += ["", f"## {title}", ""] + _overview(main, meta["detailed"]) + [""]

    rank_of = {id(c): r for r, c in enumerate(main, 1)}
    for rank, c in enumerate(main[:top], 1):
        s = c.summary or {}
        L += [f"### {rank}. {_title(c)}", ""]
        L += [f"score **{c.final:.1f}** · {c.authors} people · {c.threads} threads · {c.n} excerpts · "
              f"{_src_mix(c)} · would pay {_pay(c)} · growth ×{c.growth:.1f}"
              + (f" · coherence {s['coherence']}/5" if s.get("coherence") else "")
              + (" ⚠ mixed" if c.mixed else ""), ""]
        if c.related:
            L += ["Related: " + ", ".join(f"#{rank_of[id(o)]} {_cell(_title(o))}" for o in c.related
                                          if id(o) in rank_of), ""]
        L.append(f"**Keywords:** {', '.join(c.keywords)}")
        if c.audiences:
            L.append(f"  \n**Audiences:** {'; '.join(c.audiences)}")
        L.append("")
        if s:
            L += _summary_lines(s) + [""]
        ex = _exists_lines(c, rows)
        if ex:
            L += ["**Already exists / being built:**"] + ex + [""]
        L += [f"**Excerpts** ({min(5, len(c.reps))} of {c.n}):", ""]
        for i in c.reps[:5]:
            L += [_quote(rows[i], 380), ""]

    if other:
        L += ["## Other needs (hardware, content, policy)", "",
              "Not described in detail; here for completeness.", ""] + _overview(other, False) + [""]
    return "\n".join(L)


BRIEF_INTRO = """This file lists unmet needs that people described in public online discussions, grouped and counted
by a local pipeline. Nothing here has been checked against the market yet: that is your job.

For each need:
1. Search the web for existing products (start with "Check first", then search freely).
2. Decide: unsolved, badly solved, overpriced, or well served? Who would pay, and roughly how much?
3. Could one developer working full-time ship a useful first version in about 3 months?
Then return a shortlist of the 5–10 most promising needs with the reasoning and the products you found.

How to read the numbers: people = different authors; threads = different discussions; sources = how many excerpts
came from each community (several communities = stronger signal); would pay = share of excerpts that say they would
pay or already pay (in brackets: implied costs); growth = how much more often it appears in the most recent third of
the period than expected (×1.0 = steady). ⚠ = the grouped excerpts mix different problems.

App Store and WordPress.org excerpts are 1–3 star reviews of top-grossing apps and popular plugins: the product
exists, and the reviewer says what it lacks, paywalls, overprices or took away. For them a "thread" is one app or
plugin, so many threads = the same gap across many products. Reviews don't count towards growth. For these needs the
question is less "does a product exist?" (it does) than "is there one that does this well, and would these people
switch?"."""


def _brief_entry(n: int, c, rows, rank_of: dict, quotes: int = 4, long: bool = True) -> list:
    s = c.summary or {}
    L = [f"## {n}. {_title(c)}" + (" ⚠" if c.mixed else ""), "",
         f"**{c.authors} people · {c.threads} threads · {_src_mix(c)} · would pay {_pay(c)} · "
         f"growth ×{c.growth:.1f}** (report #{rank_of.get(id(c), '?')})", ""]
    L += _summary_lines(s) if s else [f"- **Keywords:** {', '.join(c.keywords)}"]
    ex = _exists_lines(c, rows, builders=2 if long else 1)
    if ex:
        L += ["- **Already exists / being built:**"] + ["  " + e for e in ex]
    if c.related and long:
        L.append("- **Related:** " + ", ".join(_cell(_title(o)) for o in c.related))
    L.append("")
    for i in c.quotes[:quotes]:
        L += [_quote(rows[i], 300 if long else 220), ""]
    return L


def render_brief(main, rows, meta, extra: dict) -> str:
    top = main[:meta["top"]]
    rank_of = {id(c): r for r, c in enumerate(main, 1)}
    L = [f"# Idea Finder brief — {datetime.now().strftime('%Y-%m-%d')}", "", BRIEF_INTRO, "",
         f"Data: {_sources(meta)} excerpts · {meta['groups'].get('main', meta['excerpts'])} needs that software or a "
         f"service could solve · local models: `{meta['summaries'] or 'none'}` for descriptions, "
         f"`{meta['embedding']}` for grouping.", "",
         "## Overview", "", "| # | Need | People | Sources | Would pay | Growth |", "|---|---|---|---|---|---|"]
    for n, c in enumerate(top, 1):
        L.append(f"| {n} | {_cell(_title(c))}{' ⚠' if c.mixed else ''} | {c.authors} | {_src_mix(c)} | {_pay(c)} | "
                 f"×{c.growth:.1f} |")
    L.append("")
    for n, c in enumerate(top, 1):
        L += _brief_entry(n, c, rows, rank_of)
    if extra:
        L += ["# Strongest needs of the smaller sources", "",
              "Needs where the smaller communities contribute most, outside the list above (their volume is lower, "
              "so they rarely reach the top on counts alone).", ""]
        n = len(top)
        for src, clusters in extra.items():
            L += [f"### From {SOURCE_NAMES.get(src, src)}", ""]
            for c in clusters:
                n += 1
                L += _brief_entry(n, c, rows, rank_of, quotes=3, long=False)
    return "\n".join(L)


def write_report(out_dir, main, other, rows, meta, extra: dict | None = None) -> Path:
    out = Path(out_dir) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_markdown(main, other, rows, meta), encoding="utf-8")
    try:  # the brief must never cost us the rest of the report
        (out / "brief.md").write_text(render_brief(main, rows, meta, extra or {}), encoding="utf-8")
    except Exception:
        log("could not write brief.md (the rest of the report is fine):\n" + traceback.format_exc())

    where = {}
    for prefix, clusters in (("", main), ("other-", other)):
        for rank, c in enumerate(clusters, 1):
            for i in c.members:
                where[i] = f"{prefix}{rank}"
            for i in c.builders:
                where[i] = f"{prefix}{rank} (builder)"
    with open(out / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "rank", "title", "final_score", "score", "people", "threads", "excerpts", "sources",
                    "would_pay", "implied_pay", "growth", "coherence_5", "show_hn", "builders", "products",
                    "keywords", "problem", "who", "wants", "current_solutions", "check_first", "reviewed_products"])
        for group, clusters in (("software", main), ("other", other)):
            for rank, c in enumerate(clusters, 1):
                s = c.summary or {}
                w.writerow([group, rank, _title(c), round(c.final, 3), round(c.score, 3), c.authors, c.threads, c.n,
                            _src_mix(c), round(c.pay_explicit, 3), round(c.pay_implied, 3), round(c.growth, 3),
                            s.get("coherence", ""), len(c.show_hn), len(c.builders),
                            "; ".join(p for p, _ in c.products), "; ".join(c.keywords), s.get("problem", ""),
                            s.get("who", ""), s.get("wants", ""), s.get("current_solutions", ""),
                            "; ".join(s.get("verify", [])), "; ".join(p for p, _ in c.reviewed)])
    with open(out / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "group", "source", "date", "author", "url", "solution_type", "buildable", "would_pay",
                    "speaker", "products", "snippet", "problem", "audience", "category"])
        for i, r in enumerate(rows):
            w.writerow([where.get(i, ""), r["group"], r["source"], _date(r["created_at"]), r["author"], r["url"],
                        r.get("solution_type") or "", r.get("buildable") if r.get("buildable") is not None else "",
                        r.get("wtp") or "", r.get("speaker") or "", "; ".join(r["products"]), r["snippet"],
                        r["problem"] or "", r["audience"] or "", r["category"] or ""])
    data = {"meta": meta, "clusters": [
        {"group": c.group, "rank": rank, "title": _title(c), "final_score": c.final, "score": c.score,
         "people": c.authors, "threads": c.threads, "excerpts": c.n, "sources": c.sources,
         "would_pay": c.pay_explicit, "implied_pay": c.pay_implied, "growth": c.growth, "mixed": c.mixed,
         "keywords": c.keywords, "products": c.products, "show_hn": c.show_hn, "reviewed": c.reviewed,
         "builders": [rows[i]["item_id"] for i in c.builders], "related": [_title(o) for o in c.related],
         "summary": c.summary, "representatives": [rows[i]["item_id"] for i in c.reps],
         "quotes": [rows[i]["item_id"] for i in c.quotes]}
        for clusters in (main, other) for rank, c in enumerate(clusters, 1)]}
    (out / "clusters.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
