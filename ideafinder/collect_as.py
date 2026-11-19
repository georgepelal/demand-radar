"""Collect 1-3 star App Store reviews of top-grossing apps via Apple's public RSS feeds (no key, no account).

Top-grossing = apps people actually pay for, so their complaints ("lacks X", "moved X behind a subscription",
"removed Y") point at gaps in products with proven demand. Platform giants and dating apps are left out: nobody can
compete with them on their own platform, and dating complaints are about matches, not software.
The feeds serve up to 50 reviews per page; they often answer with an empty page, so each page is retried with a
cache buster. Only the most informative low-star reviews per app are kept (see reviews.py).
"""
from __future__ import annotations

import random
import re
import time
from datetime import datetime

import requests

from .reviews import review_title, short_name, store_best
from .util import html_to_text, log

COUNTRY = "us"
TOP_URL = "https://itunes.apple.com/{cc}/rss/topgrossingapplications/limit={limit}/genre={genre}/json"
REVIEWS_URL = "https://itunes.apple.com/{cc}/rss/customerreviews/page={page}/id={app}/sortby=mosthelpful/json"

# Software genres only (no games, music, social networking, shopping or travel apps).
GENRES = {
    6007: "Productivity", 6000: "Business", 6015: "Finance", 6013: "Health & Fitness", 6017: "Education",
    6008: "Photo & Video", 6002: "Utilities", 6012: "Lifestyle", 6027: "Graphics & Design", 6026: "Developer Tools",
}
# Platform giants (Google, Apple, Meta/Instagram, Snap, Amazon, OpenAI, Anthropic, X, ByteDance) and dating apps.
_EXCLUDED_ARTIST = re.compile(
    r"^(?:google|apple)\b|\b(?:google llc|meta platforms|instagram|facebook|whatsapp|snap,? inc|amzn|amazon"
    r"|openai|anthropic|x corp|twitter|bytedance|tiktok|lemon inc|match group|tinder|hinge|bumble)\b", re.I)
# dating apps, recognised by name or description (Feeld, Raya, Cuffed... don't say "dating" in their name)
_DATING = re.compile(r"\b(?:dating|singles|hook ?ups?|soulmate|matchmaking|tinder|hinge|bumble)\b", re.I)


def _get_entries(session: requests.Session, url: str, tries: int = 4) -> list:
    """Entries of one feed page; empty answers are retried with a cache buster (the feed is flaky)."""
    for attempt in range(tries):
        u = url if attempt == 0 else f"{url}?cb={random.randint(1, 10 ** 9)}"
        try:
            r = session.get(u, timeout=30)
            if r.status_code == 200:
                entries = r.json().get("feed", {}).get("entry", [])
                entries = [entries] if isinstance(entries, dict) else entries
                if entries:
                    return entries
            elif r.status_code not in (403, 404, 429) and r.status_code < 500:
                return []
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1 + attempt * 2)
    return []


def top_apps(session: requests.Session, genre: int, n: int) -> tuple:
    """([(app_id, name, artist)], [skipped names]) for the top-grossing apps of one genre."""
    apps, skipped = [], []
    for e in _get_entries(session, TOP_URL.format(cc=COUNTRY, limit=min(4 * n, 100), genre=genre)):
        try:
            app_id = e["id"]["attributes"]["im:id"]
            name, artist = e["im:name"]["label"], e["im:artist"]["label"]
        except (KeyError, TypeError):
            continue
        about = f"{name} {(e.get('summary') or {}).get('label', '')}"
        if _EXCLUDED_ARTIST.search(artist) or _DATING.search(about):
            skipped.append(name)
            continue
        apps.append((app_id, short_name(name), artist))
        if len(apps) >= n:
            break
    return apps, skipped


def _ts(s: str):
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except (TypeError, ValueError):
        return None


def app_reviews(session: requests.Session, app_id: str, name: str, genre: str, pages: int, now: int) -> tuple:
    """(reviews fetched, low-star candidates) for one app."""
    fetched, out = 0, []
    for page in range(1, pages + 1):
        entries = [e for e in _get_entries(session, REVIEWS_URL.format(cc=COUNTRY, page=page, app=app_id))
                   if isinstance(e, dict) and "im:rating" in e]
        fetched += len(entries)
        for e in entries:
            try:
                stars = int(e["im:rating"]["label"])
                rid = e["id"]["label"]
            except (KeyError, TypeError, ValueError):
                continue
            if stars > 3:
                continue
            title = html_to_text((e.get("title") or {}).get("label", ""))
            body = html_to_text((e.get("content") or {}).get("label", ""))
            text = f"{title}\n{body}".strip() if title and not body.startswith(title) else body
            if len(text) < 25:
                continue
            try:
                votes = int((e.get("im:voteSum") or {}).get("label", 0))
            except ValueError:
                votes = 0
            out.append({
                "id": f"as:{app_id}:{rid}", "source": "as", "author": (e.get("author") or {}).get("name", {}).get(
                    "label", ""), "created_at": _ts((e.get("updated") or {}).get("label")),
                "title": review_title(name, f"{genre} app", "App Store", stars), "text": text,
                "url": f"https://apps.apple.com/{COUNTRY}/app/id{app_id}?see-all=reviews", "points": votes,
                "query": f"top-grossing {genre}", "fetched_at": now, "stars": stars,
            })
        if len(entries) < 50:  # last page
            break
        time.sleep(0.3)
    return fetched, out


def collect_as(con, genres: list[int] | None = None, apps_per_genre: int = 15, pages: int = 4,
               per_app: int = 8) -> int:
    genres = genres or list(GENRES)
    now = int(time.time())
    session = requests.Session()
    session.headers["User-Agent"] = "ideafinder/0.1 (personal research script)"
    total_new, seen = 0, set()
    for n, genre in enumerate(genres, 1):
        gname = GENRES.get(genre, str(genre))
        apps, skipped = top_apps(session, genre, apps_per_genre)
        apps = [a for a in apps if a[0] not in seen]
        seen.update(a[0] for a in apps)
        log(f"[{n}/{len(genres)}] App Store {gname}: {len(apps)} top-grossing apps"
            + (f" (left out: {', '.join(skipped[:8])})" if skipped else ""))
        if not apps:
            log("  the feed returned nothing; run `collect --source as` again later to fill the gap")
        for app_id, name, _artist in apps:
            thread = f"as:{app_id}"
            have = con.execute("SELECT COUNT(*) FROM items WHERE thread_id = ?", (thread,)).fetchone()[0]
            if have >= per_app:
                continue
            fetched, cands = app_reviews(session, app_id, name, gname, pages, now)
            signal, new = store_best(con, thread, cands, per_app)
            con.commit()
            total_new += new
            log(f"  {name[:40]}: {fetched} reviews, {len(cands)} with 1-3 stars, {signal} with a need signal, "
                f"{new} kept")
            time.sleep(0.3)
    log(f"App Store done: {total_new} new reviews")
    return total_new
