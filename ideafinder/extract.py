"""Step 2: find the sentences that express a need, using the regexes in patterns.py."""
from __future__ import annotations

import json
import re

from .patterns import COMPILED, MONEY_PATTERN, REVIEW_ALL
from .util import log

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_SNIPPET = 700
MIN_SNIPPET = 25


def _strip_quotes(text: str) -> str:
    # On HN, lines starting with ">" quote the parent comment: not the author's own words.
    return "\n".join(ln for ln in text.split("\n") if not ln.lstrip().startswith(">"))


def find_snippet(title: str, text: str, kind: str):
    """Return (snippet, matched_pattern_names, has_money_signal) or None.

    Reviews (App Store, WordPress.org) also use the review patterns, and a short review is kept whole: the
    collector already picked the most informative ones, and its headline and context matter."""
    review = kind == "review"
    patterns = REVIEW_ALL if review else COMPILED
    body = (text or "") if review else _strip_quotes(text or "")
    if kind in ("story", "question") and title:  # the title is the author's own words for posts and questions
        body = f"{title}\n{body}"
    sents = [s.strip() for s in _SENT_SPLIT.split(body) if s and s.strip()]
    if review and sents and sents[0][-1] not in ".!?…:":
        sents[0] += "."  # the review's headline
    hits, names = [], set()
    for i, s in enumerate(sents):
        matched = [n for n, rx in patterns.items() if rx.search(s)]
        if matched:
            hits.append(i)
            names.update(matched)
    if not hits:
        return None
    if review:
        if len(" ".join(sents)) <= MAX_SNIPPET:
            snippet = " ".join(sents)
            return snippet, sorted(names), bool(MONEY_PATTERN.search(snippet))
        hits = sorted({0} | set(hits))  # long review: keep its headline with the matching sentences

    def join(indices):
        parts, prev = [], None
        for j in indices:
            if prev is not None and j != prev + 1:
                parts.append("…")
            parts.append(sents[j])
            prev = j
        return " ".join(parts)

    context = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(sents)})
    snippet = join(context)
    if len(snippet) > MAX_SNIPPET:
        snippet = join(hits)
    if len(snippet) > MAX_SNIPPET:
        snippet = snippet[:MAX_SNIPPET].rsplit(" ", 1)[0] + " …"
    if len(snippet) < MIN_SNIPPET:
        return None
    return snippet, sorted(names), bool(MONEY_PATTERN.search(snippet))


def run_extract(con) -> int:
    rows = con.execute(
        "SELECT id, kind, title, text FROM items WHERE id NOT IN (SELECT item_id FROM processed)"
    ).fetchall()
    found = 0
    for r in rows:
        res = find_snippet(r["title"] or "", r["text"] or "", r["kind"] or "")
        if res:
            snippet, names, money = res
            con.execute(
                "INSERT OR IGNORE INTO pains (item_id, snippet, patterns, money) VALUES (?,?,?,?)",
                (r["id"], snippet, json.dumps(names), int(money)),
            )
            found += 1
        con.execute("INSERT OR IGNORE INTO processed (item_id) VALUES (?)", (r["id"],))
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM pains").fetchone()[0]
    log(f"extract: scanned {len(rows)} new items, {found} pain-point excerpts found ({total} in total)")
    return found
