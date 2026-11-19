"""Step 3: ask a local LLM which excerpts are real needs, and what kind of solution they call for.

Two stages:
  1. a short, recall-first question for every excerpt: "is this a need?"
  2. the detailed fields, only for the excerpts stage 1 kept (about half), which can still reject them.
Willingness to pay comes mostly from a regex ("I'd pay for", "take my money"): on the labelled sample it was far
more reliable than the LLM's own judgement.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import llm
from .db import get_meta, set_meta
from .patterns import COMPILED, MONEY_PATTERN, WEAK_PATTERNS
from .util import log

SCHEMA_VERSION = 3  # bump when the fields below change; confirmed needs from older versions get re-checked

SOLUTION_TYPES = ["software", "hardware", "service", "content", "policy", "other"]
WTP_LEVELS = ["none", "implied", "explicit"]
SPEAKERS = ["has_problem", "building_solution", "observer"]
CATEGORIES = [
    "developer tools", "ai and llm tools", "infrastructure and self-hosting", "security", "privacy",
    "productivity", "communication", "money and finance", "shopping", "media and entertainment", "gaming",
    "education and learning", "health", "home", "devices and gadgets", "transport and travel", "work and hiring",
    "creators and publishing", "data and research", "other",
]
# What a solo developer can turn into a product (used by analyze and evaluate).
BUILDABLE_TYPES = {"software", "service"}

_INTRO = ("You read excerpts from online discussions (Hacker News comments, Stack Exchange questions, Bluesky posts) "
          "and low-star reviews of App Store apps and WordPress plugins, to find unmet needs that a new product could "
          "address.")
_NEED_RULES = """is_need = true if the author (or people they describe) has a specific problem, a tedious manual workaround
or a missing tool, or says they would pay for something, AND some product or service could plausibly address it.
Also true: asking for a tool/app/website/service; saying they would pay for one; wishing a device existed;
describing the problem their own project solves.
is_need = false for: jokes or sarcasm; politics, society or the economy; complaints about one company's own product
decision (a missing option, a price) with no product angle; feature requests or feedback for the specific project in
the thread; how-to questions that well-known tools or techniques already answer; programming-language or compiler
feature requests; praise; vague statements.
Reviews (the title names an app or plugin and says "App Store" or "WordPress.org") follow their own rule, because
another product could offer what the reviewed one fails to: is_need = true when the reviewer wants something the
product lacks, puts behind a paywall or prices too high, took away, or limits (sync, export, devices, integrations,
privacy, offline use, data ownership). is_need = false when the review is only about bugs, crashes or slowness of that
product; account, login, billing, refund or customer-support problems; ads or upsell prompts alone; praise; or taste
(a redesign they dislike)."""

NEED_SYSTEM = f"""{_INTRO}

Decide only one thing: is_need.
{_NEED_RULES}
When unsure, answer true: a second step looks closer."""

NEED_SCHEMA = {"type": "object", "properties": {"is_need": {"type": "boolean"}}, "required": ["is_need"]}

SYSTEM = f"""{_INTRO}

{_NEED_RULES}

If is_need is true, fill in:
- problem: ONE sentence saying what cannot be done (or is painful) and why. Start with the task itself (what they
  want to get done), not with "pay for", "find a tool" or "Users need". Use the title to understand what "this" or
  "it" refers to. No usernames. Format: "<task>, because <why it is hard today>".
- audience: who has the problem, judged from the excerpt (a profession, a kind of business, a hobby...),
  or "general consumers".
- solution_type: what would solve it. software = app, website, SaaS, extension, library, CLI.
  hardware = a physical device. service = mostly done by people. content = a book, course, guide or dataset.
  policy = a law, a standard, or a company changing its own product decision. other = anything else.
- solo_buildable: true only if ONE experienced developer could ship a useful first version within about 3 months,
  without manufacturing, regulatory licenses, training large models or needing millions of users first.
- willingness_to_pay: explicit = says they would pay, already pay for something similar, or names a price.
  implied = describes a business cost or lost time/money. none = otherwise.
- speaker: has_problem = the author (or their team) wants this or suffers from it. This is the usual case and
  includes "I wish there was...", "is there a tool...", "I'd pay for...". building_solution = the author says they
  built, are building or sell something for it ("I made", "we launched"); commenting under someone else's launch is
  not. observer = the author explicitly describes OTHER people's problem ("parents struggle to...").
- products_mentioned: named products or companies (proper names) given as current solutions or alternatives,
  at most 5; not generic words like "apps" or "a book". [] if none.
- category: the closest one from the list.
For reviews: problem says what the reviewer wants done and what the product fails at ("Export notes as Markdown,
because <app> has no export"); audience = people who use that kind of app or plugin; speaker = has_problem;
products_mentioned starts with the reviewed product; willingness_to_pay = explicit when they say they pay or paid for
it (subscription, premium, pro, lifetime licence), otherwise none.
If is_need is false, the other fields do not matter: empty strings, [], false, solution_type "other",
willingness_to_pay "none", speaker "observer", category "other"."""

SCHEMA = {
    "type": "object",
    "properties": {
        "is_need": {"type": "boolean"},
        "problem": {"type": "string"},
        "audience": {"type": "string"},
        "solution_type": {"type": "string", "enum": SOLUTION_TYPES},
        "solo_buildable": {"type": "boolean"},
        "willingness_to_pay": {"type": "string", "enum": WTP_LEVELS},
        "speaker": {"type": "string", "enum": SPEAKERS},
        "products_mentioned": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string", "enum": CATEGORIES},
    },
    "required": ["is_need", "problem", "audience", "solution_type", "solo_buildable", "willingness_to_pay",
                 "speaker", "products_mentioned", "category"],
}

# Worked examples shown to the model before each question (they are excluded from the evaluation set).
_NO = {"is_need": False, "problem": "", "audience": "", "solution_type": "other", "solo_buildable": False,
       "willingness_to_pay": "none", "speaker": "observer", "products_mentioned": [], "category": "other"}
EXAMPLES = [
    ("Burger King will use AI to check if employees say 'please' and 'thank you'",
     'Nobody wants to be "engaged with" at your bakery. Give me my order, take my money, and let me go.', _NO),
    ("Show HN: CLI to order groceries via reverse-engineered REWE API",
     "An aggregator like this that could surface the same good for the cheapest price all inclusive of delivery "
     "would be something I would pay for!",
     {"is_need": True, "problem": "Find the cheapest grocery basket across delivery services including fees, "
      "because prices have to be compared shop by shop by hand", "audience": "online grocery shoppers",
      "solution_type": "software", "solo_buildable": True, "willingness_to_pay": "explicit",
      "speaker": "has_problem", "products_mentioned": ["REWE"], "category": "shopping"}),
    ("Most Americans don't pay for news and don't think they need to",
     "I'd pay for news if there were any at least semi-mainstream left wing news sources. However, I'm not going "
     "to pay for enlightened centrism.", _NO),
    ("Bikeshedding, or why I want to build a laptop",
     "I really want this non-enshittified macbook alternative!!!! Shut up and take my money.",
     {"is_need": True, "problem": "Buy a laptop with MacBook-level build quality that the owner fully controls, "
      "because current options are either locked down or poorly made", "audience": "laptop buyers who value control "
      "and repairability", "solution_type": "hardware", "solo_buildable": False, "willingness_to_pay": "explicit",
      "speaker": "has_problem", "products_mentioned": ["MacBook"], "category": "devices and gadgets"}),
    ("Show HN: Set Purchasing Power Parity app prices for 160 countries in Google Play",
     "I built Play Console Pricing Adjuster, a Chrome browser extension that automates global pricing inside Google "
     "Play Console. If you want fair, localized pricing, you have to manually update ~170 countries.",
     {"is_need": True, "problem": "Set localized app prices for about 170 countries in Google Play Console, because "
      "each country has to be updated by hand", "audience": "mobile app developers", "solution_type": "software",
      "solo_buildable": True, "willingness_to_pay": "none", "speaker": "building_solution",
      "products_mentioned": ["Google Play Console"], "category": "developer tools"}),
    ("Show HN: Vibe coding embedded systems with simulators",
     "I created an account to try it out, but it won't even let me access a demo project... I wish there was at "
     "least a sandbox to test it in!!", _NO),
    ("Don't make me talk to your chatbot",
     "It's incredibly frustrating to spend a good 10 minutes navigating a website's complex web of menus to get a "
     "phone number. Then spend another 5 minutes listening to bots telling me to press 1 for English, only to fall "
     "into the wrong menu.",
     {"is_need": True, "problem": "Reach a human at a company's support line quickly, because menus and chatbots "
      "hide the way to an agent", "audience": "customers contacting support", "solution_type": "software",
      "solo_buildable": True, "willingness_to_pay": "none", "speaker": "has_problem", "products_mentioned": [],
      "category": "communication"}),
    ("I won't buy Framework laptop only because of the touchscreen",
     "They call it the ultimate laptop, extremely configurable and repairable. Then why is there no option to have "
     "a non-touchscreen display", _NO),
    ("Some notes on starting to use Django",
     "Thanks for this! I wish there were more cross-comparisons like this out there of what it is actually like to "
     "use some of these frameworks.",
     {"is_need": True, "problem": "Compare what web frameworks are like to use in practice, because few honest "
      "side-by-side write-ups exist", "audience": "developers choosing a web framework", "solution_type": "content",
      "solo_buildable": True, "willingness_to_pay": "none", "speaker": "has_problem",
      "products_mentioned": ["Django"], "category": "developer tools"}),
    ("Is there a service for calculating and paying federal taxes on shareholder distributions?",
     "Is there a service for calculating and paying federal taxes on shareholder distribution payments? I currently "
     "run 1 man LLC that's taxed as an S-Corp in the US.",
     {"is_need": True, "problem": "Calculate and pay federal taxes on S-corp shareholder distributions, because a "
      "one-person company has no simple service for it", "audience": "owners of one-person S-corp companies",
      "solution_type": "software", "solo_buildable": True, "willingness_to_pay": "implied",
      "speaker": "has_problem", "products_mentioned": [], "category": "money and finance"}),
    ("A tool that can drill a hole in a stud 6 feet away?",
     "There is a firebreak 6 feet away and I need to drill a hole. Is there a tool that could do this?", _NO),
    ("Age verification for social media, the beginning of the end for a free internet?",
     "Why isn't there a simple solution in the software world that's functionally equivalent to flashing your ID at "
     "the liquor store? They get the verification they need with no permanent record of your PII.",
     {"is_need": True, "problem": "Prove one's age online without leaving a permanent record of personal data, "
      "because current checks collect and store ID documents", "audience": "people asked to verify their age online",
      "solution_type": "software", "solo_buildable": False, "willingness_to_pay": "none", "speaker": "observer",
      "products_mentioned": [], "category": "privacy"}),
    ("", "Is there an app that will centralize all of the various newsletters I subscribe to (substack and "
     "elsewhere) without trying to be a social media app?",
     {"is_need": True, "problem": "Read all newsletter subscriptions in one place, because they are scattered "
      "across platforms that push social features", "audience": "newsletter readers", "solution_type": "software",
      "solo_buildable": True, "willingness_to_pay": "none", "speaker": "has_problem",
      "products_mentioned": ["Substack"], "category": "media and entertainment"}),
    ("", "Do you know how much I would pay for a Death of Stalin-like film when It Happens?", _NO),
    ("UpdraftPlus (WordPress plugin, WordPress.org) - 2-star review",
     "Only Paid Version Backs Up Critical Files. Might be OK if you're paying for the premium version, but the free "
     "version has serious limitations and can leave you in a bad position. For example, the free version does not "
     "backup the critical .htaccess or wp-config.php files. If you don't know what these are and you're just assuming "
     "you have a full \"backup\", good luck. Still looking for a better alternative.",
     {"is_need": True, "problem": "Back up a whole WordPress site, including .htaccess and wp-config.php, without "
      "paying, because the free UpdraftPlus leaves those files out", "audience": "WordPress site owners",
      "solution_type": "software", "solo_buildable": True, "willingness_to_pay": "none", "speaker": "has_problem",
      "products_mentioned": ["UpdraftPlus"], "category": "infrastructure and self-hosting"}),
    ("CamScanner (Productivity app, App Store) - 1-star review",
     "Disappointed by App Store and the app team. … I accidentally forgot to cancel the auto-renewal for the app and "
     "was charged for a one-year subscription. … This is incredibly frustrating and unfair. … The lack of flexibility "
     "and understanding from both the App Store and the app's developers reflects poorly on their customer service.",
     _NO),
]


def _user_text(title: str, snippet: str) -> str:
    return f"Title: {title or '(none)'}\n\nExcerpt:\n\"\"\"\n{snippet}\n\"\"\""


def _examples(stage: int) -> list:
    out = []
    for title, snippet, answer in EXAMPLES:
        a = {"is_need": answer["is_need"]} if stage == 1 else answer
        out.append((_user_text(title, snippet), json.dumps(a)))
    return out


_NOT_A_PRODUCT = {"", "[]", "none", "n/a", "na", "null", "nothing", "unknown", "-"}


def clean_products(products) -> list:
    out = []
    for p in products if isinstance(products, list) else []:
        name = str(p).strip().strip("'\"")
        if name.lower() not in _NOT_A_PRODUCT and name not in out:
            out.append(name)
    return out[:5]


def _pick(value, allowed: list[str]) -> str:
    value = str(value or "").strip().lower()
    if value in allowed:
        return value
    return "other" if "other" in allowed else allowed[0]


def resolve_wtp(snippet: str, llm_value: str) -> str:
    """Regex first: on the labelled sample "I'd pay for / take my money" was right every time it fired."""
    if COMPILED["would_pay"].search(snippet):
        return "explicit"
    if llm_value == "explicit" and MONEY_PATTERN.search(snippet):
        return "explicit"
    return "implied" if llm_value in ("explicit", "implied") else "none"


def check_need(model: str, title: str, snippet: str, examples: bool = True) -> bool:
    """Stage 1: recall-first yes/no."""
    out = llm.chat_json(model, NEED_SYSTEM, _user_text(title, snippet), NEED_SCHEMA, max_tokens=64,
                        examples=_examples(1) if examples else None)
    return bool(out.get("is_need"))


def classify_one(model: str, title: str, snippet: str, examples: bool = True) -> dict:
    """Stage 2: the detailed fields (and a second, closer is_need decision)."""
    out = llm.chat_json(model, SYSTEM, _user_text(title, snippet), SCHEMA,
                        examples=_examples(2) if examples else None)
    problem = str(out.get("problem") or "").strip()
    if not (bool(out.get("is_need")) and len(problem) > 10):
        return {"is_need": False}
    llm_wtp = _pick(out.get("willingness_to_pay"), WTP_LEVELS)
    return {
        "is_need": True,
        "problem": problem,
        "audience": str(out.get("audience") or "").strip(),
        "solution_type": _pick(out.get("solution_type"), SOLUTION_TYPES),
        "solo_buildable": bool(out.get("solo_buildable")),
        "willingness_to_pay": resolve_wtp(snippet, llm_wtp),
        "wtp_llm": llm_wtp,
        "speaker": _pick(out.get("speaker"), SPEAKERS),
        "products_mentioned": clean_products(out.get("products_mentioned") or []),
        "category": _pick(out.get("category"), CATEGORIES),
    }


def _is_weak(row) -> bool:
    return not (set(json.loads(row["patterns"])) - WEAK_PATTERNS)


def _run_pool(fn, rows, workers: int, label: str, on_result) -> tuple:
    """Run fn(row) in parallel; on_result(row, result) is called in this thread. Returns (done, failed)."""
    total = len(rows)
    if not total:
        return 0, 0
    done = failed = consecutive_fail = 0
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=workers)
    futures = {ex.submit(fn, r): r for r in rows}
    try:
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # keep going; the row is retried next run
                failed += 1
                consecutive_fail += 1
                if failed <= 5:
                    log(f"  failed on {row['item_id']}: {e}")
                if consecutive_fail >= 20:
                    raise llm.OllamaError("20 failures in a row; is Ollama still running?")
                continue
            consecutive_fail = 0
            on_result(row, res)
            done += 1
            if done in (1, 5) or done % 25 == 0 or done == total:
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  {label}: {done}/{total}, ~{(total - done) / rate / 60:.0f} min left")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return done, failed


# Seconds per excerpt (wall clock, 2 parallel requests) measured with gemma4:12b on the M5 Mac; replaced by the
# speeds each real run measures and stores.
DEFAULT_SPEED = {"stage1": 0.65, "stage2": 4.5, "pass_rate": 0.55}


def _estimate(con, model: str, detail_model: str, n1: int, n2_known: int) -> tuple:
    s1 = get_meta(con, f"speed:stage1:{model}") or DEFAULT_SPEED["stage1"]
    s2 = get_meta(con, f"speed:stage2:{detail_model}") or DEFAULT_SPEED["stage2"]
    rate = get_meta(con, "stage1_pass_rate") or DEFAULT_SPEED["pass_rate"]
    n2 = n2_known + rate * n1
    return n1 * s1, n2 * s2, n2


def _hm(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


def run_classify(con, model: str = llm.DEFAULT_LLM, limit: int | None = None, workers: int = 2,
                 include_weak: bool = False, recheck_rejected: bool = False, detail_model: str | None = None,
                 examples: bool = True, dry_run: bool = False) -> None:
    """Classify new excerpts and upgrade needs found by older versions.

    model: stage 1 (is it a need?), detail_model: stage 2 (fields); defaults to model.
    recheck_rejected also re-checks excerpts an older version rejected (slower, finds needs it missed)."""
    detail_model = detail_model or model
    if not dry_run:
        llm.ensure_model(model)
        if detail_model != model:
            llm.ensure_model(detail_model)
    rows = con.execute(
        "SELECT p.item_id, p.snippet, p.patterns, p.money, p.llm_model, p.is_pain, i.title, i.created_at "
        "FROM pains p JOIN items i ON i.id = p.item_id "
        "WHERE p.llm_model IS NULL OR (COALESCE(p.schema_version, 1) < ? AND (p.is_pain = 1 OR ?))",
        (SCHEMA_VERSION, int(recheck_rejected)),
    ).fetchall()
    if not include_weak:
        skipped = sum(1 for r in rows if r["llm_model"] is None and _is_weak(r))
        rows = [r for r in rows if r["llm_model"] is not None or not _is_weak(r)]
        if skipped:
            log(f"classify: skipping {skipped} excerpts that only vent (use --include-weak to check them too)")
    # Most promising first, so a --limit run still covers the best excerpts:
    # needs already confirmed, then strong signals before pure venting, then money mentions, then newest.
    rows.sort(key=lambda r: (r["is_pain"] != 1, _is_weak(r), -r["money"], -(r["created_at"] or 0)))
    if limit:
        rows = rows[:limit]
    if not rows:
        log("classify: nothing new to classify")
        return
    stage1 = [r for r in rows if r["is_pain"] != 1]          # unchecked, or rejected earlier (--recheck-rejected)
    keep = {r["item_id"] for r in rows if r["is_pain"] == 1}  # already needs: straight to stage 2
    t1, t2, n2 = _estimate(con, model, detail_model, len(stage1), len(keep))
    log(f"classify: {len(stage1)} excerpts for stage 1 ({model}), {len(keep)} known needs go straight to stage 2 "
        f"({detail_model}); about {n2:.0f} in stage 2 in total.")
    log(f"estimated time: ~{_hm(t1 + t2)} (stage 1 ~{_hm(t1)}, stage 2 ~{_hm(t2)}), from "
        + ("speeds measured in earlier runs" if get_meta(con, f"speed:stage2:{detail_model}") else
           "the gemma4:12b speeds measured on your Mac") + ". Ctrl+C is safe: progress is saved.")
    if dry_run:
        log("dry run: nothing was classified")
        return

    def save_stage1(row, is_need):
        if is_need:
            keep.add(row["item_id"])
            con.execute("UPDATE pains SET llm_model=?, is_pain=1, schema_version=NULL WHERE item_id=?",
                        (model, row["item_id"]))
        else:
            con.execute("UPDATE pains SET llm_model=?, is_pain=0, schema_version=?, problem='', solution_type=NULL "
                        "WHERE item_id=?", (model, SCHEMA_VERSION, row["item_id"]))
        con.commit()

    def save_stage2(row, res):
        con.execute(
            "UPDATE pains SET llm_model=?, schema_version=?, is_pain=?, problem=?, audience=?, category=?, "
            "solution_type=?, buildable=?, wtp=?, speaker=?, products=? WHERE item_id=?",
            (detail_model, SCHEMA_VERSION, int(res["is_need"]), res.get("problem", ""), res.get("audience", ""),
             res.get("category", ""), res.get("solution_type"),
             int(res["solo_buildable"]) if res["is_need"] else None, res.get("willingness_to_pay"),
             res.get("speaker"), json.dumps(res.get("products_mentioned", [])), row["item_id"]),
        )
        con.commit()

    try:
        t0 = time.time()
        d1, f1 = _run_pool(lambda r: check_need(model, r["title"], r["snippet"], examples), stage1, workers,
                           "stage 1", save_stage1)
        if d1 >= 50:  # remember the real speed and pass rate for the next estimate
            set_meta(con, f"speed:stage1:{model}", round((time.time() - t0) / d1, 3))
            set_meta(con, "stage1_pass_rate", round(sum(1 for r in stage1 if r["item_id"] in keep) / d1, 3))
            con.commit()
        stage2 = [r for r in rows if r["item_id"] in keep]
        kept = []
        t0 = time.time()
        d2, f2 = _run_pool(lambda r: classify_one(detail_model, r["title"], r["snippet"], examples), stage2, workers,
                           "stage 2", lambda row, res: (save_stage2(row, res), kept.append(res["is_need"])))
        if d2 >= 50:
            set_meta(con, f"speed:stage2:{detail_model}", round((time.time() - t0) / d2, 3))
            con.commit()
    except KeyboardInterrupt:
        con.commit()
        log("interrupted: progress saved, run again to continue")
        os._exit(130)  # exit now instead of waiting for the requests Ollama is still working on
    log(f"classify done: stage 1 checked {d1} ({f1} failed); stage 2 checked {d2} ({f2} failed), "
        f"{sum(kept)} real needs")
