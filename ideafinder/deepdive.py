"""`python -m ideafinder deepdive --cluster N`: look closely at one need of a report.

collect keeps only the 8 most useful reviews per product, often old "most helpful" ones, so a cluster can show 4
reviews of a product that has 80 recent complaints about the same thing. This step takes the App Store apps and
WordPress plugins whose reviews are in the cluster and fetches ALL their 1-3 star reviews since a date:
App Store from several countries, newest first (the feed stops at 500 per country); WordPress.org from the review feed
and, where the feed's 30 latest don't reach the date, from the review pages.
The local model then names the recurring complaints and tags every review, so the result says how many people, of
which product, complain about what, in which half-year, and where they say they are moving to.

Output (in the report's folder): deepdive_<rank>/deepdive.md and reviews.json. Nothing goes into the database.
"""
from __future__ import annotations

import csv
import html
import json
import random
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from . import llm
from .collect_as import _get_entries
from .collect_wp import _DC, _META_P, _REV_LOG, _STARS_SUFFIX
from .reviews import parse_title
from .util import html_to_text, log

COUNTRIES = ["us", "gb", "ca", "au"]
AS_RECENT = "https://itunes.apple.com/{cc}/rss/customerreviews/page={page}/id={app}/sortby=mostrecent/json"
AS_PAGES = 10  # Apple serves at most 10 pages of 50
WP_FEED = "https://wordpress.org/support/plugin/{slug}/reviews/feed/?filter={stars}"
WP_PAGE = "https://wordpress.org/support/plugin/{slug}/reviews/page/{page}/?filter={stars}"
_WP_TOPIC = re.compile(r'bbp-topic-permalink" href="([^"]+)">([^<]*)')
_WP_DATE = re.compile(r'title="([A-Z][a-z]+ \d{1,2}, \d{4})')
BATCH = 10


# ---------------------------------------------------------------- which products
def latest_report(out_dir: str) -> Path:
    reports = sorted(p for p in Path(out_dir).iterdir() if (p / "clusters.json").exists())
    if not reports:
        raise SystemExit(f"no report found in {out_dir}/; run `analyze` first")
    return reports[-1]


def cluster_products(con: sqlite3.Connection, report: Path, rank: int) -> tuple:
    """(cluster dict, [(thread_id, product name, on-topic reviews in the cluster)])."""
    data = json.loads((report / "clusters.json").read_text(encoding="utf-8"))
    found = [c for c in data["clusters"] if c["group"] == "main" and c["rank"] == rank]
    if not found:
        raise SystemExit(f"no software/service need #{rank} in {report}")
    with open(report / "items.csv", encoding="utf-8") as f:
        urls = [r["url"] for r in csv.DictReader(f) if r["cluster"] == str(rank) and r["source"] in ("as", "wp")]
    counts: Counter = Counter()
    names = {}
    for url in urls:
        row = con.execute("SELECT thread_id, title FROM items WHERE url = ? LIMIT 1", (url,)).fetchone()
        if row:
            counts[row[0]] += 1
            parsed = parse_title(row[1])
            names[row[0]] = parsed[0] if parsed else row[0]
    return found[0], [(t, names[t], n) for t, n in counts.most_common()]


def product_names(con: sqlite3.Connection, thread_ids: list) -> list:
    out = []
    for t in thread_ids:
        row = con.execute("SELECT title FROM items WHERE thread_id = ? LIMIT 1", (t,)).fetchone()
        parsed = parse_title(row[0]) if row else None
        out.append((t, parsed[0] if parsed else t, 0))
    return out


# ---------------------------------------------------------------- fetching
def fetch_app_store(session, app_id: str, name: str, since: datetime, countries: list) -> tuple:
    reviews, notes = [], []
    for cc in countries:
        read = low = 0
        oldest = None
        complete = False
        for page in range(1, AS_PAGES + 1):
            entries = [e for e in _get_entries(session, AS_RECENT.format(cc=cc, page=page, app=app_id), tries=6)
                       if isinstance(e, dict) and "im:rating" in e]
            stop = False
            for e in entries:
                try:
                    when = datetime.fromisoformat(e["updated"]["label"])
                    stars = int(e["im:rating"]["label"])
                except (KeyError, TypeError, ValueError):
                    continue
                read += 1
                oldest = when if oldest is None or when < oldest else oldest
                if when < since:
                    stop = True
                    continue
                if stars > 3:
                    continue
                title = html_to_text((e.get("title") or {}).get("label", ""))
                body = html_to_text((e.get("content") or {}).get("label", ""))
                low += 1
                reviews.append({"product": name, "store": f"App Store {cc.upper()}", "stars": stars,
                                "date": when.date().isoformat(), "title": title, "text": body,
                                "id": f"as:{app_id}:{e.get('id', {}).get('label', '')}",
                                "url": f"https://apps.apple.com/{cc}/app/id{app_id}?see-all=reviews"})
            if stop or len(entries) < 50:
                complete = stop or len(entries) < 50
                break
            time.sleep(0.4)
        notes.append({"product": name, "where": f"App Store {cc.upper()}", "read": read, "low_star": low,
                      "oldest": oldest.date().isoformat() if oldest else "", "complete": complete or read == 0})
    return reviews, notes


def _wp_topic_text(session, url: str) -> str:
    try:
        page = session.get(url, timeout=30).text
    except requests.RequestException:
        return ""
    i = page.find('class="bbp-topic-content"')
    if i < 0:
        return ""
    j = page.find('class="bbp-reply-content"', i)
    seg = _REV_LOG.sub("", page[i:j if j > 0 else i + 8000])
    return html_to_text(seg.split(">", 1)[1] if ">" in seg else seg)[:2000]


def fetch_wordpress(session, slug: str, name: str, since: datetime) -> tuple:
    reviews, notes, seen = [], [], set()
    for stars in (1, 2, 3):
        try:
            r = session.get(WP_FEED.format(slug=slug, stars=stars), timeout=30)
            items = ET.fromstring(r.content).findall(".//item") if r.status_code == 200 else []
        except (requests.RequestException, ET.ParseError):
            items = []
        dates = []
        for it in items:
            try:
                when = parsedate_to_datetime(it.findtext("pubDate"))
            except (TypeError, ValueError):
                continue
            dates.append(when)
            link = (it.findtext("link") or "").strip()
            if when < since or link in seen:
                continue
            seen.add(link)
            title = _STARS_SUFFIX.sub("", html.unescape(it.findtext("title") or "")).strip()
            body = html_to_text(_META_P.sub("", _REV_LOG.sub("", it.findtext("description") or "")))
            reviews.append({"product": name, "store": "WordPress.org", "stars": stars,
                            "date": when.date().isoformat(), "title": title, "text": body, "id": link, "url": link,
                            "author": (it.findtext(_DC) or "").strip()})
        extra = 0
        if len(items) >= 30 and dates and min(dates) >= since:  # the feed doesn't reach back far enough
            for page in range(2, 20):
                try:
                    listing = session.get(WP_PAGE.format(slug=slug, page=page, stars=stars), timeout=30).text
                except requests.RequestException:
                    break
                topics = []
                for block in re.split(r'<ul id="bbp-topic-', listing)[1:]:
                    m, d = _WP_TOPIC.search(block), _WP_DATE.search(block)
                    if m and d:
                        topics.append((m.group(1), html.unescape(m.group(2)).strip(),
                                       datetime.strptime(d.group(1), "%B %d, %Y").replace(tzinfo=timezone.utc)))
                if not topics:
                    break
                for link, title, when in topics:
                    if when >= since and link not in seen:
                        seen.add(link)
                        extra += 1
                        reviews.append({"product": name, "store": "WordPress.org", "stars": stars,
                                        "date": when.date().isoformat(), "title": title,
                                        "text": _wp_topic_text(session, link), "id": link, "url": link})
                        time.sleep(0.4)
                if min(w for _, _, w in topics) < since:
                    break
                time.sleep(0.4)
        notes.append({"product": name, "where": f"WordPress.org {stars}★", "read": len(items) + extra,
                      "low_star": sum(1 for r in reviews if r["stars"] == stars),
                      "oldest": min(dates).date().isoformat() if dates else "", "complete": True})
        time.sleep(0.4)
    return reviews, notes


# ---------------------------------------------------------------- themes
THEMES_SYSTEM = """You read 1-3 star reviews of products that people use for one need, and list the complaints
that come back again and again. Be specific: "price increases for long-time customers" or "money held after a
payment", not "pricing" or "issues". Only list a theme that several reviews support. Return 4-8 themes, each with a
short name (max 6 words) and a one-sentence definition that makes clear which reviews belong to it."""

TAG_SYSTEM = """You tag 1-3 star reviews with the complaint themes they contain. A review can have several themes
or none. Also give `switching_to`: the product the reviewer says they moved to, are moving to, or recommend instead,
exactly as written; "" if none is named. Return one entry per review, in order."""


def _short(r: dict, n: int) -> str:
    return re.sub(r"\s+", " ", f"{r['title']} — {r['text']}")[:n]


def find_themes(model: str, need: dict, reviews: list, sample: int = 80, seed: int = 42) -> list:
    rnd = random.Random(seed)
    by_product = defaultdict(list)
    for r in reviews:
        by_product[r["product"]].append(r)
    picked = []
    while len(picked) < min(sample, len(reviews)):  # round-robin over products so one big product doesn't dominate
        for rs in by_product.values():
            if rs:
                picked.append(rs.pop(rnd.randrange(len(rs))))
    lines = [f"Need: {need.get('title', '')}. {need.get('summary', {}).get('problem', '')}", "", "Reviews:"]
    lines += [f"- [{r['product']}, {r['stars']}★] {_short(r, 320)}" for r in picked[:sample]]
    schema = {"type": "object", "properties": {"themes": {"type": "array", "minItems": 3, "maxItems": 8, "items": {
        "type": "object", "properties": {"name": {"type": "string"}, "definition": {"type": "string"}},
        "required": ["name", "definition"]}}}, "required": ["themes"]}
    out = llm.chat_json(model, THEMES_SYSTEM, "\n".join(lines), schema, temperature=0.2, max_tokens=1200)
    return [t for t in out.get("themes", []) if str(t.get("name", "")).strip()][:8]


def tag_reviews(model: str, themes: list, reviews: list) -> None:
    head = "Themes:\n" + "\n".join(f"{k}. {t['name']}: {t['definition']}" for k, t in enumerate(themes, 1))
    for b in range(0, len(reviews), BATCH):
        chunk = reviews[b:b + BATCH]
        user = head + "\n\nReviews:\n" + "\n".join(f"{n}. {_short(r, 600)}" for n, r in enumerate(chunk, 1))
        schema = {"type": "object", "properties": {"reviews": {
            "type": "array", "minItems": len(chunk), "maxItems": len(chunk), "items": {
                "type": "object", "properties": {"themes": {"type": "array", "items": {"type": "integer"}},
                                                 "switching_to": {"type": "string"}},
                "required": ["themes", "switching_to"]}}}, "required": ["reviews"]}
        try:
            out = llm.chat_json(model, TAG_SYSTEM, user, schema, temperature=0.0, max_tokens=800).get("reviews")
        except llm.OllamaError as e:
            log(f"  tagging failed for {len(chunk)} reviews ({str(e)[:80]})")
            out = None
        if not isinstance(out, list) or len(out) != len(chunk):
            out = [{"themes": [], "switching_to": ""}] * len(chunk)
            for r in chunk:
                r["untagged"] = True
        for r, o in zip(chunk, out):
            r["themes"] = sorted({int(k) for k in o.get("themes", []) if str(k).lstrip("-").isdigit()
                                  and 1 <= int(k) <= len(themes)})
            r["switching_to"] = str(o.get("switching_to") or "").strip()
        if (b // BATCH) % 10 == 9:
            log(f"  {b + len(chunk)}/{len(reviews)} reviews tagged")


# ---------------------------------------------------------------- report
def _half(date: str) -> str:
    return date[:4] + ("-H1" if date[5:7] <= "06" else "-H2")


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render(need: dict, rank: int, since: str, products: list, notes: list, themes: list, reviews: list,
           model: str) -> str:
    prods = [p for _, p, _ in products]
    halves = sorted({_half(r["date"]) for r in reviews})
    L = [f"# Deep dive: #{rank} {need.get('title', '')}", "",
         f"All 1–3★ reviews since {since} of the products whose reviews are in this need's cluster: "
         f"{len(reviews)} reviews, themes found and tagged by `{model}`. A review can have several themes.", "",
         "Products: " + ", ".join(f"{p} (`{t}`" + (f", {n} reviews in the cluster)" if n else ")")
                                  for t, p, n in products)
         + ". A product that doesn't fit the need can be left out with `--apps`.", "",
         "## Coverage", "", "| Product | Where | Reviews read | 1–3★ since date | Oldest reached | Complete |",
         "|---|---|---|---|---|---|"]
    for n in notes:
        L.append(f"| {_cell(n['product'])} | {n['where']} | {n['read']} | {n['low_star']} | {n['oldest']} | "
                 f"{'yes' if n['complete'] else '**no**: Apple serves only the newest 500'} |")
    L += ["", "## Themes by product", "",
          "| Theme | " + " | ".join(_cell(p) for p in prods) + " | Total |",
          "|---|" + "---|" * (len(prods) + 1)]
    per_product = Counter(r["product"] for r in reviews)
    L.append("| *all 1–3★ reviews* | " + " | ".join(str(per_product[p]) for p in prods) + f" | {len(reviews)} |")
    for k, t in enumerate(themes, 1):
        c = Counter(r["product"] for r in reviews if k in r.get("themes", []))
        L.append(f"| {_cell(t['name'])} | " + " | ".join(str(c[p]) for p in prods) + f" | {sum(c.values())} |")
    none = sum(1 for r in reviews if not r.get("themes"))
    L += [f"| *no theme* | " + " | ".join(str(sum(1 for r in reviews if r['product'] == p and not r.get('themes')))
                                         for p in prods) + f" | {none} |", ""]
    L += ["## Themes over time", "", "| Theme | " + " | ".join(halves) + " |", "|---|" + "---|" * len(halves)]
    L.append("| *all 1–3★ reviews* | " + " | ".join(str(sum(1 for r in reviews if _half(r['date']) == h))
                                                    for h in halves) + " |")
    for k, t in enumerate(themes, 1):
        L.append(f"| {_cell(t['name'])} | " + " | ".join(
            str(sum(1 for r in reviews if k in r.get("themes", []) and _half(r["date"]) == h)) for h in halves) + " |")
    L += ["", "The newest half-year is not over yet; App Store rows marked incomplete above miss older reviews.", ""]
    vague = re.compile(r"(?i)^(?:a|an|another|other|some|any|the)\b|\b(?:app|apps|platform|one|alternative)s?$")
    shown: dict = {}  # "quickbooks" -> "QuickBooks" (first spelling seen)
    moving: Counter = Counter()
    for r in reviews:
        name = r.get("switching_to", "")
        if name and not vague.search(name):
            moving[shown.setdefault(name.lower(), name)] += 1
    L += ["## Where they say they are moving", ""]
    L += [f"- {_cell(name)}: {n}" for name, n in moving.most_common(12)] or ["- nobody named a product"]
    L += ["", "## Themes", ""]
    for k, t in enumerate(themes, 1):
        rs = sorted((r for r in reviews if k in r.get("themes", [])), key=lambda r: r["date"], reverse=True)
        L += [f"### {k}. {t['name']} ({len(rs)})", "", t["definition"], ""]
        for r in rs[:4]:
            L += [f"> {_cell(_short(r, 300))}  ", f"> — {r['product']}, {r['stars']}★, {r['store']}, {r['date']} · "
                  f"[link]({r['url']})", ""]
    untagged = sum(1 for r in reviews if r.get("untagged"))
    if untagged:
        L += [f"*{untagged} reviews could not be tagged (model errors) and count as \"no theme\".*", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main entry
def run_deepdive(con, rank: int, report: str | None = None, out_dir: str = "reports", apps: list | None = None,
                 since: str = "2025-01-01", countries: list | None = None, model: str = llm.DEFAULT_LLM) -> Path:
    llm.ensure_model(model)
    rep = Path(report) if report else latest_report(out_dir)
    need, products = cluster_products(con, rep, rank)
    if apps:
        products = product_names(con, apps)
    if not products:
        raise SystemExit(f"need #{rank} has no App Store or WordPress.org reviews; pass products with --apps "
                         "(e.g. --apps as:592163563,wp:cookie-law-info)")
    log(f"deep dive into #{rank} {need['title']} (report {rep.name}): "
        + ", ".join(f"{name} ({t})" for t, name, _ in products))
    start = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    session = requests.Session()
    session.headers["User-Agent"] = "ideafinder/0.1 (personal research script)"
    reviews, notes = [], []
    for thread, name, _ in products:
        src, ident = thread.split(":", 1)
        if src == "as":
            rs, ns = fetch_app_store(session, ident, name, start, countries or COUNTRIES)
        elif src == "wp":
            rs, ns = fetch_wordpress(session, ident, name, start)
        else:
            continue
        reviews += [r for r in rs if (r.get("title") or r.get("text"))]
        notes += ns
        log(f"  {name}: {len(rs)} reviews with 1-3 stars since {since}")
    if not reviews:
        raise SystemExit("no reviews found for these products in that period")
    log(f"finding the recurring complaints in {len(reviews)} reviews with {model}")
    themes = find_themes(model, need, list(reviews))
    log("  themes: " + "; ".join(t["name"] for t in themes))
    tag_reviews(model, themes, reviews)
    out = rep / f"deepdive_{rank}"
    out.mkdir(exist_ok=True)
    (out / "reviews.json").write_text(json.dumps({"need": need.get("title"), "since": since, "themes": themes,
                                                  "coverage": notes, "reviews": reviews}, indent=1,
                                                 ensure_ascii=False), encoding="utf-8")
    (out / "deepdive.md").write_text(render(need, rank, since, products, notes, themes, reviews, model),
                                     encoding="utf-8")
    log(f"deep dive written to {out}/deepdive.md")
    return out
