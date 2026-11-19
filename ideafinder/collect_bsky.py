"""Collect posts from Bluesky via its public search API (no account needed).

Bluesky adds everyday consumers and creators. Posts are short and often jokes, so fewer of them are real needs
(about 1 in 4 in a sample, versus 1 in 2 on Stack Exchange); the LLM filters them.
Without an account the API serves only the first page of each search, so we search month by month and keep
at most `per_month` posts per phrase and month: a sample, not everything.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import requests

from .db import get_meta, set_meta
from .util import log

API = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
DEFAULT_PHRASES = [
    "i wish there was", "is there an app", "is there a tool", "is there a website",
    "i would pay for", "someone should build", "someone should make",
]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _month_windows(start: int, end: int):
    t = datetime.fromtimestamp(start, tz=timezone.utc)
    stop = datetime.fromtimestamp(end, tz=timezone.utc)
    while t < stop:
        nxt = min(t + timedelta(days=30), stop)
        yield int(t.timestamp()), int(nxt.timestamp())
        t = nxt


def _row(p: dict, phrase: str, now: int):
    rec = p.get("record") or {}
    text = (rec.get("text") or "").strip()
    if not text:
        return None
    try:
        created = int(datetime.fromisoformat(rec.get("createdAt", "").replace("Z", "+00:00")).timestamp())
    except ValueError:
        created = None
    uri = p.get("uri", "")
    handle = (p.get("author") or {}).get("handle", "")
    rkey = uri.rsplit("/", 1)[-1]
    root = ((rec.get("reply") or {}).get("root") or {}).get("uri") or uri
    return (f"bsky:{uri}", "bsky", "post", handle, created, "", text.replace("’", "'"),
            f"https://bsky.app/profile/{handle}/post/{rkey}", f"bsky:{root}", p.get("likeCount"), phrase, now)


def collect_bsky(con, days: int = 1095, phrases: list[str] | None = None, per_month: int = 25) -> int:
    phrases = phrases or DEFAULT_PHRASES
    now = int(time.time())
    start = now - days * 86400
    session = requests.Session()
    session.headers["User-Agent"] = "ideafinder/0.1 (personal research script)"
    total_new = 0
    for n, phrase in enumerate(phrases, 1):
        key = f"bsky_coverage:{phrase}"
        cov = get_meta(con, key)
        ranges = [(start, now)] if not cov else (
            ([(start, cov[0])] if start < cov[0] else []) + [(max(start, cov[1] - 2 * 86400), now)])
        got = new = 0
        blocked = False
        for a, b in ranges:
            for wa, wb in _month_windows(a, b):
                params = {"q": f'"{phrase}"', "limit": min(per_month, 100), "sort": "latest", "lang": "en",
                          "since": _iso(wa), "until": _iso(wb)}
                for attempt in range(4):
                    try:
                        r = session.get(API, params=params, timeout=30)
                    except requests.RequestException:
                        time.sleep(2 ** attempt * 2)
                        continue
                    if r.status_code == 429 or r.status_code >= 500:
                        time.sleep(2 ** attempt * 5)
                        continue
                    break
                else:
                    continue
                if r.status_code in (401, 403):
                    blocked = True
                    break
                if r.status_code != 200:
                    continue
                posts = r.json().get("posts", [])
                got += len(posts)
                before = con.total_changes
                con.executemany(
                    "INSERT OR IGNORE INTO items (id, source, kind, author, created_at, title, text, url,"
                    " thread_id, points, query, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [row for row in (_row(p, phrase, now) for p in posts) if row],
                )
                new += con.total_changes - before
                time.sleep(0.4)  # be polite
            if blocked:
                break
        if blocked:
            log(f'[{n}/{len(phrases)}] Bluesky "{phrase}": search refused (HTTP 403); skipping this phrase')
        else:
            set_meta(con, key, [min(start, cov[0]) if cov else start, now])
        con.commit()
        total_new += new
        log(f'[{n}/{len(phrases)}] Bluesky "{phrase}": {got} posts, {new} new')
    log(f"Bluesky done: {total_new} new posts")
    return total_new
