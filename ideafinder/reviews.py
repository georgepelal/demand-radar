"""Shared code for the review sources: App Store (collect_as.py) and WordPress.org plugins (collect_wp.py).

Reviews of products people already pay for or depend on: the gap is "it exists, but it lacks / paywalls / took away X",
a stronger signal than "is there a tool". Each app or plugin is one "thread", so a cluster's thread count says how many
DIFFERENT products share the complaint (a category-wide gap rather than one product's quirk).
Only the few most informative reviews per product are kept (default 8), so a handful of popular products can't drown
the rest, and classify stays around 2 hours for a full run.
"""
from __future__ import annotations

import html
import math
import re

from .patterns import REVIEW_ALL, WEAK_PATTERNS

REVIEW_SOURCES = {"as": "App Store", "wp": "WordPress.org"}

# weight of each signal when ranking the reviews of one product; the usual need patterns count 2, venting 0.5
_WEIGHTS = {"rv_missing": 2.0, "rv_request": 2.0, "rv_removed": 2.0, "rv_price": 1.0, "rv_switch": 1.0}
_TITLE = re.compile(r"^(.*) \(([^()]*), (App Store|WordPress\.org)\) - (\d)-star review$")


def review_title(product: str, kind_label: str, store: str, stars: int) -> str:
    """e.g. "Dropbox (Productivity app, App Store) - 2-star review"."""
    return f"{product} ({kind_label}, {store}) - {stars}-star review"


def short_name(name: str) -> str:
    """'UpdraftPlus: WP Backup & Migration Plugin' -> 'UpdraftPlus' (the brand, without the marketing tagline)."""
    name = html.unescape(name).strip()
    short = re.split(r" [–—-] |: ", name, maxsplit=1)[0].strip()
    return short if len(short) > 4 else name[:60]  # "VPN - Super Unlimited Proxy" stays whole


def parse_title(title: str | None):
    """(product, stars) from a review title, or None for other items."""
    m = _TITLE.match(title or "")
    return (m.group(1), int(m.group(4))) if m else None


def matched(text: str) -> set:
    return {name for name, rx in REVIEW_ALL.items() if rx.search(text)}


def informativeness(text: str, stars: int, votes: int = 0):
    """How useful a review is likely to be; None if it has no need signal at all (pure praise, bug or venting)."""
    names = matched(text)
    strong = names - WEAK_PATTERNS
    if not strong:
        return None
    strength = sum(_WEIGHTS.get(n, 2.0) for n in strong) + 0.5 * len(names & WEAK_PATTERNS)
    # capped, so a long rant that hits every pattern doesn't beat a short, specific request
    return min(strength, 6.0) + 0.5 * math.log1p(max(votes or 0, 0)) + (1.0 if stars in (2, 3) else 0.0)


def store_best(con, thread_id: str, candidates: list, per_product: int) -> tuple:
    """Insert the most informative new reviews of one product, up to per_product stored in total.

    candidates: dicts with the items columns plus "stars". Returns (candidates with a signal, newly stored)."""
    have = con.execute("SELECT COUNT(*) FROM items WHERE thread_id = ?", (thread_id,)).fetchone()[0]
    scored = []
    for c in candidates:
        s = informativeness(c["text"], c["stars"], c.get("points") or 0)
        if s is not None:
            scored.append((s, c))
    room = per_product - have
    if room <= 0 or not scored:
        return len(scored), 0
    ids = [c["id"] for _, c in scored]
    known = {r[0] for r in con.execute(f"SELECT id FROM items WHERE id IN ({','.join('?' * len(ids))})", ids)}
    fresh = sorted(((s, c) for s, c in scored if c["id"] not in known), key=lambda t: -t[0])[:room]
    con.executemany(
        "INSERT OR IGNORE INTO items (id, source, kind, author, created_at, title, text, url, thread_id, points,"
        " query, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(c["id"], c["source"], "review", c["author"], c["created_at"], c["title"], c["text"], c["url"],
          thread_id, c.get("points"), c["query"], c["fetched_at"]) for _, c in fresh],
    )
    return len(scored), len(fresh)
