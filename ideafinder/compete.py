"""Step 5: for the top ideas of a report, look for what already exists and judge whether a gap is left.

For each idea: the LLM writes a few search queries -> we search Hacker News launches ("Show HN") and GitHub
repositories -> the LLM compares what it found with the needs people described and gives a verdict.
Writes compete.md and compete.json next to the report. Free APIs only; set GITHUB_TOKEN for higher GitHub limits.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import llm
from .util import log

HN_SEARCH = "https://hn.algolia.com/api/v1/search"
GH_SEARCH = "https://api.github.com/search/repositories"
UA = "ideafinder/0.1 (personal research script)"

QUERY_SYSTEM = """You help check whether products already exist for a user need.
Given the need and examples of how people describe it, return:
- product: one sentence describing the product someone would build for it.
- queries: 3 short search queries (2-5 words each) that would find EXISTING products or open-source projects
  solving it. Use the words a product page would use, not the complaint."""
QUERY_SCHEMA = {
    "type": "object",
    "properties": {"product": {"type": "string"}, "queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["product", "queries"],
}

GAP_SYSTEM = """You are a skeptical market analyst. You get a user need (with real quotes), the product idea,
and search results: product launches on Hacker News and GitHub repositories (with stars and last update).
Judge from this material, not from memory alone:
- solutions: up to 5 existing products or projects that really address the need (name + one short note each).
  Ignore search results that are unrelated. If nothing addresses it, return [].
- verdict: "open" = nothing found really solves it; "some competition" = a few partial or small solutions;
  "crowded" = several established solutions.
- gap: 0-2. 2 = a clear gap remains; 1 = a gap for a niche, cheaper, simpler or better-made version;
  0 = well served.
- wedge: one sentence: how a newcomer could stand out (a niche, a feature, a price, a platform),
  or "none" if there is no realistic one.
- reason: one or two sentences explaining the verdict."""
GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "solutions": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["open", "some competition", "crowded"]},
        "gap": {"type": "integer"},
        "wedge": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["solutions", "verdict", "gap", "wedge", "reason"],
}


def _get_json(session: requests.Session, url: str, params: dict, wait_on_limit: bool = True):
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 200:
            return r.json(), r.headers
        if r.status_code in (403, 429) and wait_on_limit:  # GitHub rate limit
            reset = int(r.headers.get("X-RateLimit-Reset", "0") or 0)
            wait = max(5, min(70, reset - int(time.time()) + 1)) if reset else 30
            log(f"  rate limited by {url.split('/')[2]}, waiting {wait}s")
            time.sleep(wait)
            continue
        return None, r.headers
    return None, {}


def search_show_hn(session, query: str, n: int = 6) -> list:
    data, _ = _get_json(session, HN_SEARCH, {"query": query, "tags": "show_hn", "hitsPerPage": n})
    out = []
    for h in (data or {}).get("hits", []):
        out.append({"source": "Show HN", "name": h.get("title", ""), "points": h.get("points") or 0,
                    "date": (h.get("created_at") or "")[:10],
                    "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                    "link": h.get("url") or ""})
    return out


def search_github(session, query: str, n: int = 6) -> list:
    data, headers = _get_json(session, GH_SEARCH, {"q": f"{query} in:name,description", "sort": "stars",
                                                   "order": "desc", "per_page": n})
    if headers and headers.get("X-RateLimit-Remaining") == "0":
        time.sleep(max(1, int(headers.get("X-RateLimit-Reset", "0") or 0) - int(time.time()) + 1))
    out = []
    for r in (data or {}).get("items", []):
        out.append({"source": "GitHub", "name": r.get("full_name", ""), "stars": r.get("stargazers_count", 0),
                    "updated": (r.get("pushed_at") or "")[:10], "description": (r.get("description") or "")[:200],
                    "url": r.get("html_url", "")})
    return out


def _latest_report(out_dir: str) -> Path:
    dirs = sorted(p for p in Path(out_dir).glob("*/clusters.json"))
    if not dirs:
        raise SystemExit(f"no report found in {out_dir}/ (run `analyze` first)")
    return dirs[-1].parent


def _need_text(con, cluster: dict, k: int = 8) -> tuple:
    ids = cluster.get("representatives", [])[:k]
    rows = []
    for item_id in ids:
        r = con.execute("SELECT p.problem, p.snippet FROM pains p WHERE p.item_id = ?", (item_id,)).fetchone()
        if r:
            rows.append((r[0] or "", r[1] or ""))
    s = cluster.get("summary") or {}
    head = s.get("problem") or "; ".join(cluster.get("keywords", [])[:6])
    return head, rows


def run_compete(con, report: str | None = None, out_dir: str = "reports", top: int = 15,
                model: str = llm.DEFAULT_LLM) -> Path:
    llm.ensure_model(model)
    folder = Path(report) if report else _latest_report(out_dir)
    data = json.loads((folder / "clusters.json").read_text(encoding="utf-8"))
    clusters = [c for c in data["clusters"] if c.get("group", "main") == "main"][:top]
    log(f"compete: checking the top {len(clusters)} ideas of {folder} with {model}")
    session = requests.Session()
    session.headers["User-Agent"] = UA
    gh = requests.Session()
    gh.headers.update({"User-Agent": UA, "Accept": "application/vnd.github+json"})
    if os.environ.get("GITHUB_TOKEN"):
        gh.headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    gh_pause = 1 if os.environ.get("GITHUB_TOKEN") else 6.5  # unauthenticated search: 10 requests per minute

    results = []
    for n, c in enumerate(clusters, 1):
        need, examples = _need_text(con, c)
        quotes = "\n".join(f"- {p or s[:200]}" for p, s in examples)
        user = f"Need: {need}\nTitle: {c.get('title', '')}\n\nHow people describe it:\n{quotes}"
        try:
            q = llm.chat_json(model, QUERY_SYSTEM, user, QUERY_SCHEMA, temperature=0.2)
        except llm.OllamaError as e:
            log(f"  [{n}] query generation failed: {e}")
            continue
        queries = [str(x).strip() for x in q.get("queries", []) if str(x).strip()][:3]
        found: dict = {}
        for query in queries:
            for hit in search_show_hn(session, query):
                found.setdefault(hit["url"], hit)
            for hit in search_github(gh, query):
                found.setdefault(hit["url"], hit)
            time.sleep(gh_pause)
        hits = sorted(found.values(), key=lambda h: -(h.get("stars") or h.get("points") or 0))[:20]
        listing = "\n".join(
            f"- [{h['source']}] {h['name']}"
            + (f" ({h['stars']} stars, last push {h['updated']}): {h['description']}" if h["source"] == "GitHub"
               else f" ({h['points']} points, {h['date']})")
            for h in hits) or "(no results)"
        mentioned = ", ".join(p for p, _ in c.get("products", [])) or "none"
        user2 = (f"{user}\n\nProduct idea: {q.get('product', '')}\n"
                 f"Products the people themselves mention: {mentioned}\n\nSearch results:\n{listing}")
        try:
            g = llm.chat_json(model, GAP_SYSTEM, user2, GAP_SCHEMA, temperature=0.2, max_tokens=1024)
        except llm.OllamaError as e:
            log(f"  [{n}] gap judgement failed: {e}")
            continue
        g["solutions"] = [str(x) for x in g.get("solutions") or []
                          if str(x).strip() and not str(x).lower().startswith(("none", "no ", "n/a"))]
        try:
            g["gap"] = int(min(max(int(g.get("gap", 1)), 0), 2))
        except (TypeError, ValueError):
            g["gap"] = 1
        results.append({"rank": c.get("rank"), "title": c.get("title"),
                        "score": c.get("final_score"), "people": c.get("people"), "product": q.get("product", ""),
                        "queries": queries, "hits": hits, **g})
        log(f"  [{n}/{len(clusters)}] {c.get('title')}: {g.get('verdict')} (gap {g['gap']}), {len(hits)} results")

    (folder / "compete.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (folder / "compete.md").write_text(_render(results, folder, model), encoding="utf-8")
    log(f"compete written to {folder}/compete.md")
    return folder


def _render(results: list, folder: Path, model: str) -> str:
    order = {"open": 0, "some competition": 1, "crowded": 2}
    L = [f"# What already exists — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", "",
         f"Report: `{folder.name}` · judged by `{model}` from Show HN launches and GitHub repositories. "
         "A verdict is only as good as the search: an \"open\" idea still needs a manual check "
         "(Google, Product Hunt, app stores) before you build anything.", "",
         "| Report # | Idea | Verdict | Gap /2 | People |", "|---|---|---|---|---|"]
    ranked = sorted(results, key=lambda r: (order.get(r.get("verdict"), 3), -r["gap"], -(r.get("people") or 0)))
    for r in ranked:
        L.append(f"| {r['rank']} | {str(r['title']).replace('|', '/')} | {r.get('verdict')} | {r['gap']} | "
                 f"{r.get('people')} |")
    L.append("")
    for r in ranked:
        L += [f"## #{r['rank']} {r['title']}", "",
              f"**{r.get('verdict')}** · gap {r['gap']}/2 · {r.get('people')} people", "",
              f"- **Product idea:** {r.get('product', '')}",
              f"- **Why:** {r.get('reason', '')}",
              f"- **How to stand out:** {r.get('wedge', '')}"]
        if r.get("solutions"):
            L.append(f"- **What exists:** {'; '.join(map(str, r['solutions']))}")
        L.append(f"- **Searched for:** {', '.join(r.get('queries', []))}")
        L += ["", "Search results:", ""]
        for h in r["hits"][:8]:
            if h["source"] == "GitHub":
                L.append(f"- GitHub [{h['name']}]({h['url']}) · {h['stars']}★ · last push {h['updated']} · "
                         f"{h['description']}")
            else:
                L.append(f"- Show HN [{h['name']}]({h['url']}) · {h['points']} points · {h['date']}")
        if not r["hits"]:
            L.append("- nothing found")
        L.append("")
    return "\n".join(L)
