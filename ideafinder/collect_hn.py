"""Collect stories and comments from Hacker News via the free Algolia search API.

Docs: https://hn.algolia.com/api  (no key needed; ~10k requests/hour per IP)
"""
from __future__ import annotations

import time

import requests

from .db import get_meta, set_meta
from .util import html_to_text, log

API = "https://hn.algolia.com/api/v1/search_by_date"
MAX_RESULTS = 1000      # the API returns at most 1000 hits per query, so big windows get split
MIN_WINDOW = 3600       # never split a time window below one hour
USER_AGENT = "ideafinder/0.1 (personal research script)"

# Exact phrases (the API matches them as quoted phrases). Every phrase here is also
# recognised by the regexes in patterns.py, so collected items can be extracted.
DEFAULT_PHRASES = [
    # something is missing
    "i wish there was", "i wish there were", "wish someone would build",
    "someone should build", "someone should make", "somebody should build",
    "why is there no", "why isn't there a", "why doesn't this exist",
    "surprised there isn't", "surprised nobody has",
    "nothing on the market", "doesn't exist yet",
    # searching for a solution
    "is there a tool", "is there an app", "is there a service", "is there any tool",
    "does anyone know of a", "looking for a tool", "looking for an app",
    "looking for a service", "looking for an alternative", "is there an alternative",
    "a good alternative to",
    # willingness to pay
    "i would pay for", "i'd pay for", "i would happily pay", "i'd happily pay",
    "would gladly pay", "take my money", "too expensive for",
    # pain / workarounds
    "there is no good way", "there's no good way", "there's no easy way", "no good solution",
    "biggest pain point", "pain in the ass", "so frustrating", "incredibly frustrating",
    "drives me crazy", "huge time sink", "waste so much time",
    "we still manually", "have to manually",
    "every tool i've tried", "none of the existing", "existing tools are",
]


def load_phrases(path: str | None = None) -> list[str]:
    if not path:
        return list(DEFAULT_PHRASES)
    with open(path, encoding="utf-8") as f:
        return [ln.strip().lower() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _get(session: requests.Session, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 2
            log(f"  HN API error ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _fetch(session: requests.Session, phrase: str, start: int, end: int, stats: dict) -> list[dict]:
    params = {
        "query": f'"{phrase}"',
        "tags": "(story,comment)",
        "hitsPerPage": MAX_RESULTS,
        "numericFilters": f"created_at_i>={start},created_at_i<{end}",
    }
    data = _get(session, params)
    stats["requests"] += 1
    time.sleep(0.2)  # be polite
    if data.get("nbHits", 0) > MAX_RESULTS and end - start > MIN_WINDOW:
        mid = (start + end) // 2
        return _fetch(session, phrase, start, mid, stats) + _fetch(session, phrase, mid, end, stats)
    return data.get("hits", [])


def _hit_to_row(h: dict, phrase: str, now: int):
    tags = h.get("_tags") or []
    oid = str(h["objectID"])
    if "story" in tags:
        kind, title = "story", h.get("title") or ""
        text = html_to_text(h.get("story_text"))
        thread = oid
    else:
        kind, title = "comment", h.get("story_title") or ""
        text = html_to_text(h.get("comment_text"))
        thread = str(h.get("story_id") or oid)
    if not text and not (kind == "story" and title):
        return None  # deleted / empty
    return (
        f"hn:{oid}", "hn", kind, h.get("author"), h.get("created_at_i"), title, text,
        f"https://news.ycombinator.com/item?id={oid}", f"hn:{thread}", h.get("points"), phrase, now,
    )


def collect(con, days: int = 365, phrases: list[str] | None = None) -> int:
    """Fetch the last `days` days for every phrase. Re-runs only fetch what is new."""
    phrases = phrases or DEFAULT_PHRASES
    now = int(time.time())
    start = now - days * 86400
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    stats = {"requests": 0}
    total_new = 0
    for n, phrase in enumerate(phrases, 1):
        key = f"hn_coverage:{phrase}"
        cov = get_meta(con, key)
        ranges = []
        if cov:
            lo, hi = cov
            if start < lo:
                ranges.append((start, lo))
            ranges.append((max(start, hi - 2 * 86400), now))  # re-check last 2 days for late comments
        else:
            ranges.append((start, now))
        hits: list[dict] = []
        for a, b in ranges:
            if b > a:
                hits += _fetch(session, phrase, a, b, stats)
        before = con.total_changes
        con.executemany(
            "INSERT OR IGNORE INTO items (id, source, kind, author, created_at, title, text, url,"
            " thread_id, points, query, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [row for row in (_hit_to_row(h, phrase, now) for h in hits) if row],
        )
        new = con.total_changes - before
        set_meta(con, key, [min(start, cov[0]) if cov else start, now])
        con.commit()
        total_new += new
        log(f'[{n}/{len(phrases)}] "{phrase}": {len(hits)} hits, {new} new')
    log(f"collect done: {total_new} new items ({stats['requests']} API requests)")
    return total_new
