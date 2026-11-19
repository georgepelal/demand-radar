"""Collect questions from Stack Exchange sites via the free API (https://api.stackexchange.com/docs).

Stack Exchange adds people who are not developers: personal finance, travel, DIY, parenting, the workplace...
Questions like "Is there a website that..." are close to unmet needs by construction.

Without a key the API allows about 300 requests per day per IP address; with a free key, 10,000.
To get one, register an app at https://stackapps.com/apps/oauth/register and set IDEAFINDER_SE_KEY.
When the daily quota runs out, collection stops cleanly and the next run continues where it stopped.
Content is CC BY-SA: the report links back to every question.
"""
from __future__ import annotations

import os
import re
import time

import requests

from .db import get_meta, set_meta
from .util import html_to_text, log

API = "https://api.stackexchange.com/2.3/search/advanced"
KEY = os.environ.get("IDEAFINDER_SE_KEY", "").strip()

# Sites with everyday or professional needs (not programming Q&A, which HN already covers).
DEFAULT_SITES = [
    "softwarerecs", "hardwarerecs", "webapps", "money", "workplace", "freelancing", "diy", "parenting",
    "travel", "cooking", "gardening", "pets", "photo", "music", "academia", "android", "apple",
    "bicycles", "fitness", "outdoors", "video", "writing", "woodworking", "expatriates", "ux",
]
# Phrases suited to questions (each site x phrase is one request, so keep this list short).
DEFAULT_PHRASES = [
    "is there a website", "is there a tool", "is there an app", "is there a service",
    "is there any software", "i wish there was", "would pay for",
]


class QuotaExhausted(Exception):
    pass


def _get(session: requests.Session, params: dict, state: dict) -> dict:
    if KEY:
        params = {**params, "key": KEY}
    for attempt in range(4):
        if state.get("backoff_until", 0) > time.time():
            time.sleep(state["backoff_until"] - time.time())
        try:
            r = session.get(API, params=params, timeout=30)
        except requests.RequestException as e:
            log(f"  Stack Exchange request failed ({e}); retrying")
            time.sleep(2 ** attempt * 2)
            continue
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if data.get("backoff"):  # the API asks clients to wait before the next request
            state["backoff_until"] = time.time() + int(data["backoff"]) + 1
        if r.status_code == 200:
            state["quota"] = data.get("quota_remaining")
            return data
        name, message = data.get("error_name", ""), str(data.get("error_message", ""))
        wait_s = re.search(r"available in (\d+) seconds", message)
        if name == "throttle_violation" and wait_s and int(wait_s.group(1)) > 300:
            raise QuotaExhausted(f"daily quota used up ({message})")
        if name == "throttle_violation" or r.status_code >= 500:
            wait = 2 ** attempt * 5
            log(f"  Stack Exchange throttling ({data.get('error_message', r.status_code)}); waiting {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Stack Exchange API error {r.status_code}: {data.get('error_message', r.text[:200])}")
    raise RuntimeError("Stack Exchange API keeps failing; try again later")


def _row(q: dict, site: str, phrase: str, now: int):
    text = html_to_text(q.get("body"))
    title = html_to_text(q.get("title"))
    if not title and not text:
        return None
    qid = q["question_id"]
    owner = q.get("owner") or {}
    return (
        f"se:{site}:{qid}", f"se:{site}", "question", owner.get("display_name") or owner.get("user_id"),
        q.get("creation_date"), title, text, q.get("link") or f"https://{site}.stackexchange.com/q/{qid}",
        f"se:{site}:{qid}", q.get("score"), phrase, now,
    )


def collect_se(con, days: int = 1095, sites: list[str] | None = None, phrases: list[str] | None = None) -> int:
    sites = sites or DEFAULT_SITES
    phrases = phrases or DEFAULT_PHRASES
    now = int(time.time())
    start = now - days * 86400
    session = requests.Session()
    session.headers["User-Agent"] = "ideafinder/0.1 (personal research script)"
    state: dict = {}
    total_new = 0
    pairs = [(s, p) for s in sites for p in phrases]
    try:
        for n, (site, phrase) in enumerate(pairs, 1):
            key = f"se_coverage:{site}:{phrase}"
            cov = get_meta(con, key)
            lo = start if not cov else min(start, cov[0])
            ranges = [(start, now)] if not cov else (
                ([(start, cov[0])] if start < cov[0] else []) + [(max(start, cov[1] - 2 * 86400), now)])
            got, new = 0, 0
            for a, b in ranges:
                page = 1
                while True:
                    data = _get(session, {"site": site, "q": f'"{phrase}"', "fromdate": a, "todate": b,
                                          "sort": "creation", "order": "desc", "pagesize": 100, "page": page,
                                          "filter": "withbody"}, state)
                    items = data.get("items", [])
                    got += len(items)
                    before = con.total_changes
                    con.executemany(
                        "INSERT OR IGNORE INTO items (id, source, kind, author, created_at, title, text, url,"
                        " thread_id, points, query, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        [row for row in (_row(q, site, phrase, now) for q in items) if row],
                    )
                    new += con.total_changes - before
                    if not data.get("has_more") or page >= 5:
                        break
                    page += 1
            set_meta(con, key, [lo, now])
            con.commit()
            total_new += new
            if got:
                log(f"[{n}/{len(pairs)}] {site} \"{phrase}\": {got} questions, {new} new "
                    f"(quota left: {state.get('quota', '?')})")
            if state.get("quota") is not None and state["quota"] < 3:
                raise QuotaExhausted("daily quota used up")
    except QuotaExhausted as e:
        con.commit()
        log(f"Stack Exchange: {e}. Run `collect` again tomorrow to continue"
            + ("" if KEY else ", or set IDEAFINDER_SE_KEY for a 10,000/day quota (see collect_se.py)") + ".")
    log(f"Stack Exchange done: {total_new} new questions")
    return total_new
