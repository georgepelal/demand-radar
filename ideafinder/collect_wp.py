"""Collect 1-3 star reviews of popular WordPress.org plugins (public plugins API + review RSS feeds, no key).

Popular plugins run on millions of sites and many sell a premium version, so their low-star reviews show what site
owners need and don't get ("the free version removed X", "no way to Y", "switched to Z"). WordPress also fits a
PHP developer. Plugins written by Google or WordPress.org itself are left out.
Each review feed holds the 30 latest reviews of one star count; only the most informative ones per plugin are kept
(see reviews.py).
"""
from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

from .reviews import review_title, short_name, store_best
from .util import html_to_text, log

PLUGINS_API = "https://api.wordpress.org/plugins/info/1.2/"
FEED = "https://wordpress.org/support/plugin/{slug}/reviews/feed/?filter={stars}"
EXCLUDED_AUTHORS = ("/google/", "/wordpressdotorg/")

_DC = "{http://purl.org/dc/elements/1.1/}creator"
_REV_LOG = re.compile(r"(?s)<ul[^>]*bbp-topic-revision-log.*?</ul>")
_META_P = re.compile(r"(?si)<p>\s*(?:Replies|Rating):.*?</p>")
_STARS_SUFFIX = re.compile(r"\s*\(\d stars?\)\s*$")


def popular_plugins(session: requests.Session, n: int) -> tuple:
    plugins, skipped, page = [], [], 1
    while len(plugins) + len(skipped) < n:
        r = session.get(PLUGINS_API, params={
            "action": "query_plugins", "request[browse]": "popular", "request[per_page]": min(n, 100),
            "request[page]": page, "request[fields][description]": 0, "request[fields][sections]": 0,
            "request[fields][short_description]": 0}, timeout=30)
        r.raise_for_status()
        batch = r.json().get("plugins", [])
        if not batch:
            break
        for p in batch:
            if len(plugins) + len(skipped) >= n:
                break
            if any(a in (p.get("author_profile") or "") for a in EXCLUDED_AUTHORS):
                skipped.append(short_name(p["name"]))
            else:
                plugins.append((p["slug"], short_name(p["name"])))
        page += 1
    return plugins, skipped


def plugin_reviews(session: requests.Session, slug: str, name: str, now: int) -> tuple:
    fetched, out = 0, []
    for stars in (1, 2, 3):
        try:
            r = session.get(FEED.format(slug=slug, stars=stars), timeout=30)
            items = ET.fromstring(r.content).findall(".//item") if r.status_code == 200 else []
        except (requests.RequestException, ET.ParseError):
            items = []
        fetched += len(items)
        for it in items:
            link = (it.findtext("link") or "").strip()
            topic = link.rstrip("/").rsplit("/", 1)[-1]
            if not topic:
                continue
            title = _STARS_SUFFIX.sub("", html.unescape(it.findtext("title") or "")).strip()
            body = html_to_text(_META_P.sub("", _REV_LOG.sub("", it.findtext("description") or "")))
            text = f"{title}\n{body}".strip() if title else body
            if len(text) < 25:
                continue
            try:
                created = int(parsedate_to_datetime(it.findtext("pubDate")).timestamp())
            except (TypeError, ValueError):
                created = None
            out.append({
                "id": f"wp:{slug}:{topic}", "source": "wp", "author": (it.findtext(_DC) or "").strip(),
                "created_at": created, "title": review_title(name, "WordPress plugin", "WordPress.org", stars),
                "text": text, "url": link, "points": None, "query": "popular plugins", "fetched_at": now,
                "stars": stars,
            })
        time.sleep(0.3)
    return fetched, out


def collect_wp(con, n_plugins: int = 100, per_plugin: int = 8) -> int:
    now = int(time.time())
    session = requests.Session()
    session.headers["User-Agent"] = "ideafinder/0.1 (personal research script)"
    plugins, skipped = popular_plugins(session, n_plugins)
    log(f"WordPress.org: {len(plugins)} popular plugins" + (f" (left out: {', '.join(skipped)})" if skipped else ""))
    total_new = 0
    for n, (slug, name) in enumerate(plugins, 1):
        thread = f"wp:{slug}"
        have = con.execute("SELECT COUNT(*) FROM items WHERE thread_id = ?", (thread,)).fetchone()[0]
        if have >= per_plugin:
            continue
        fetched, cands = plugin_reviews(session, slug, name, now)
        signal, new = store_best(con, thread, cands, per_plugin)
        con.commit()
        total_new += new
        log(f"[{n}/{len(plugins)}] {name[:40]}: {fetched} low-star reviews, {signal} with a need signal, {new} kept")
    log(f"WordPress.org done: {total_new} new reviews")
    return total_new
