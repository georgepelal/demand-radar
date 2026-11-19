"""Step 4: embed -> reduce dimensions -> cluster -> score -> (LLM summary) -> report."""
from __future__ import annotations

import json
import math
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import llm
from .classify import BUILDABLE_TYPES, SCHEMA_VERSION, clean_products
from .embed import embed_ollama, embed_tfidf
from .patterns import DOMAIN_STOP_WORDS, WEAK_PATTERNS, neutralize
from .report import write_report
from .reviews import REVIEW_SOURCES, parse_title
from .util import log

MIN_CONFIRMED = 100  # "auto" mode uses LLM-confirmed needs once there are at least this many


@dataclass
class Cluster:
    label: int
    members: list                      # row indices
    group: str = "main"                # main = software a solo dev can build; other = everything else
    keywords: list = field(default_factory=list)
    authors: int = 0
    threads: int = 0
    pay_explicit: float = 0.0          # share saying they'd pay / already pay / naming a price
    pay_implied: float = 0.0
    growth: float = 1.0
    score: float = 0.0
    final: float = 0.0
    reps: list = field(default_factory=list)
    audiences: list = field(default_factory=list)
    products: list = field(default_factory=list)   # [(name, count)]
    show_hn: list = field(default_factory=list)    # [(title, url)] launches whose threads hold member excerpts
    builders: list = field(default_factory=list)   # row indices of people building a solution
    summary: dict = field(default_factory=dict)
    related: list = field(default_factory=list, repr=False, compare=False)  # nearby clusters (objects)
    attached: int = 0                  # unassigned excerpts that were close enough to join
    sources: dict = field(default_factory=dict)    # {"hn": 12, "se": 3, ...} members per source
    reviewed: list = field(default_factory=list)   # [(product, people)] apps/plugins whose low-star reviews are members
    quotes: list = field(default_factory=list)     # row indices for the brief: closest per source first

    @property
    def n(self) -> int:
        return len(self.members)

    @property
    def mixed(self) -> bool:
        """The description step found that the excerpts don't share one problem."""
        return bool(self.summary) and self.summary.get("coherence", 5) <= 2


# ---------------------------------------------------------------- data
def _thread_url(thread_id: str) -> str:
    return "https://news.ycombinator.com/item?id=" + str(thread_id).split(":")[-1]


def load_rows(con, mode: str):
    """Return rows with a "group": main (software/service ideas), other, or builder.

    Buildability is left to the final review (with web search): one comment rarely says enough."""
    total = con.execute("SELECT COUNT(*) FROM pains").fetchone()[0]
    checked = con.execute("SELECT COUNT(*) FROM pains WHERE llm_model IS NOT NULL").fetchone()[0]
    confirmed = con.execute("SELECT COUNT(*) FROM pains WHERE is_pain = 1").fetchone()[0]
    if mode == "auto":
        mode = "llm" if confirmed >= MIN_CONFIRMED else "raw"
    if mode == "llm":
        unchecked = sum(
            1 for (p,) in con.execute("SELECT patterns FROM pains WHERE llm_model IS NULL")
            if set(json.loads(p)) - WEAK_PATTERNS
        )
        if unchecked:
            log(f"note: {unchecked} excerpts not checked by the LLM yet are left out (run `classify`)")
    where = "WHERE p.is_pain = 1 AND p.problem != ''" if mode == "llm" else ""
    rows = [dict(r) for r in con.execute(
        "SELECT p.item_id, p.snippet, p.problem, p.audience, p.category, p.money, p.patterns, "
        "p.schema_version, p.solution_type, p.buildable, p.wtp, p.speaker, p.products, "
        "i.author, i.created_at, i.thread_id, i.url, i.title, i.kind "
        f"FROM pains p JOIN items i ON i.id = p.item_id {where} ORDER BY i.created_at"
    )]
    if mode == "raw":
        rows = [r for r in rows if set(json.loads(r["patterns"])) - WEAK_PATTERNS]
    detailed = mode == "llm" and any((r["schema_version"] or 1) >= SCHEMA_VERSION for r in rows)
    if detailed:
        old = sum(1 for r in rows if (r["schema_version"] or 1) < SCHEMA_VERSION)
        if old:
            log(f"note: {old} needs from an older classify run are left out until `classify` re-checks them")
        rows = [r for r in rows if (r["schema_version"] or 1) >= SCHEMA_VERSION]
    elif mode == "llm":
        log("note: needs have no solution type yet; run `classify` to split software ideas from the rest")
    for r in rows:
        r["products"] = clean_products(json.loads(r["products"])) if r.get("products") else []
        r["source"] = r["item_id"].split(":")[0]
        r["in_show_hn"] = r["source"] == "hn" and (r["title"] or "").lower().startswith("show hn")
        parsed = parse_title(r["title"]) if r["source"] in REVIEW_SOURCES else None
        r["reviewed"], r["stars"] = parsed if parsed else (None, None)
        if not detailed:
            r["group"] = "main"
        elif r["speaker"] == "building_solution":
            r["group"] = "builder"
        elif r["solution_type"] in BUILDABLE_TYPES:
            r["group"] = "main"
        else:
            r["group"] = "other"
    return rows, mode, total, checked, detailed


# ---------------------------------------------------------------- clustering
def exact_knn(X: np.ndarray, k: int) -> tuple:
    """Exact cosine nearest neighbours (each point first, at distance 0), in the format UMAP accepts.

    UMAP would otherwise use pynndescent for more than 4096 points, and pynndescent 0.7 does not compile with
    numba 0.60, the last numba for the Mac's Python 3.9. Exact search is also better and fast at this size."""
    from sklearn.neighbors import NearestNeighbors
    dist, idx = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X).kneighbors(X)
    return idx.astype(np.int32), np.clip(dist, 0, None).astype(np.float32)


def reduce_dims(X: np.ndarray, seed: int = 42):
    n = len(X)
    try:
        import umap
    except ImportError:
        umap = None
    if umap is not None and n >= 30:
        k = min(15, n - 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reducer = umap.UMAP(n_neighbors=k, n_components=5, min_dist=0.0, metric="cosine",
                                    random_state=seed, precomputed_knn=(*exact_knn(X, k), None))
                return reducer.fit_transform(X), "umap"
        except Exception as e:  # never lose hours of classify work over the layout step
            log(f"warning: UMAP failed ({type(e).__name__}: {str(e)[:120]}); using PCA instead, "
                "clusters will be rougher")
    from sklearn.decomposition import PCA
    k = max(2, min(20, X.shape[1], n - 1))
    return PCA(n_components=k, random_state=seed).fit_transform(X), "pca"


def run_hdbscan(Z: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HDBSCAN(min_cluster_size=min_cluster_size,
                       min_samples=max(2, min_cluster_size // 2)).fit_predict(Z)


def cluster_group(X: np.ndarray, idx: list[int], group: str, min_cluster_size: int | None, seed: int):
    """Cluster the rows in `idx`; returns (clusters, info string)."""
    idx = [i for i in idx if np.linalg.norm(X[i]) > 0]
    if len(idx) < 30:
        return [], f"{group}: only {len(idx)} excerpts, not clustered"
    # Small enough to keep specific needs apart (a niche of 12 people is still a lead), large enough to be a pattern.
    mcs = min_cluster_size or max(5, min(12, len(idx) // 150))
    Z, how = reduce_dims(X[idx], seed=seed)
    labels = run_hdbscan(Z, mcs)
    clusters = []
    for lab in sorted(set(labels.tolist()) - {-1}):
        members = [idx[j] for j in np.where(labels == lab)[0]]
        clusters.append(Cluster(label=len(clusters), members=members, group=group))
    noise = int((labels == -1).sum())
    return clusters, (f"{group}: {how} + HDBSCAN(min_cluster_size={mcs}) -> {len(clusters)} clusters, "
                      f"{noise} of {len(idx)} excerpts unassigned")


def cluster_keywords(texts: list[str], groups: list[list[int]], top: int = 8) -> list[list[str]]:
    """c-TF-IDF: words frequent inside a cluster but rare across clusters."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    docs = [" ".join(texts[i] for i in g) for g in groups]
    cv = CountVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS), ngram_range=(1, 2),
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#]+\b", max_features=50000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = cv.fit_transform(docs).toarray().astype(float)
    vocab = np.array(cv.get_feature_names_out())
    tf = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    avg_words = C.sum() / max(len(groups), 1)
    idf = np.log(1 + avg_words / np.maximum(C.sum(axis=0), 1))
    W = tf * idf
    out = []
    for row in W:
        picked: list[str] = []
        for j in np.argsort(-row):
            if row[j] <= 0 or len(picked) >= top:
                break
            term = vocab[j]
            if any(term in p or p in term for p in picked):  # skip "invoice" if "invoice tracking" is there
                continue
            picked.append(term)
        out.append(picked)
    return out


def _sims(X: np.ndarray, rows_idx: list[int], centroid: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):  # numpy 2.0 + macOS Accelerate can print spurious matmul warnings
        return X[rows_idx].astype(np.float64) @ centroid


def _centroid(X: np.ndarray, members: list[int]) -> np.ndarray:
    c = X[members].astype(np.float64).mean(axis=0)
    return c / (np.linalg.norm(c) or 1)


def attach_to_nearest(X: np.ndarray, clusters: list[Cluster], candidates: list[int], pct: float) -> dict:
    """{cluster index: [rows]} for candidates at least as close to a cluster as its pct-th percentile member."""
    out: dict = {}
    candidates = [i for i in candidates if np.linalg.norm(X[i]) > 0]
    if not clusters or not candidates:
        return out
    cents = np.vstack([_centroid(X, c.members) for c in clusters])
    cut = [float(np.percentile(_sims(X, c.members, cents[k]), pct)) for k, c in enumerate(clusters)]
    with np.errstate(all="ignore"):
        S = X[candidates].astype(np.float64) @ cents.T
    for pos, i in enumerate(candidates):
        k = int(np.argmax(S[pos]))
        if S[pos, k] >= cut[k]:
            out.setdefault(k, []).append(i)
    return out


def attach_builders(X: np.ndarray, clusters: list[Cluster], builder_idx: list[int]) -> None:
    """Give each "I'm building X" excerpt to the closest cluster, if it is as close as a typical member."""
    for k, idxs in attach_to_nearest(X, clusters, builder_idx, 25).items():
        clusters[k].builders.extend(idxs)


def attach_unassigned(X: np.ndarray, clusters: list[Cluster], group_idx: list[int]) -> int:
    """HDBSCAN leaves many excerpts out. Put back those as close to a cluster as its median member."""
    taken = {i for c in clusters for i in c.members}
    attached = attach_to_nearest(X, clusters, [i for i in group_idx if i not in taken], 50)
    for k, idxs in attached.items():
        clusters[k].members.extend(idxs)
        clusters[k].attached += len(idxs)
    return sum(len(v) for v in attached.values())


def split_cluster(X: np.ndarray, c: Cluster, seed: int) -> list[Cluster]:
    """Split a cluster whose excerpts don't share one problem in two (k-means on the embeddings)."""
    from sklearn.cluster import KMeans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(X[c.members].astype(np.float64))
    parts = [[c.members[j] for j in np.where(labels == v)[0]] for v in (0, 1)]
    if min(len(p) for p in parts) < 5:
        return [c]
    out = [Cluster(label=c.label * 10 + k, members=p, group=c.group) for k, p in enumerate(parts)]
    for k, idxs in attach_to_nearest(X, out, c.builders, 0).items():
        out[k].builders.extend(idxs)
    return out


def link_related(X: np.ndarray, clusters: list[Cluster], most: int = 2) -> None:
    """Remember the 1-2 clusters whose centroids are unusually close (top 3% of all pairs)."""
    if len(clusters) < 3:
        return
    cents = np.vstack([_centroid(X, c.members) for c in clusters])
    with np.errstate(all="ignore"):
        S = cents @ cents.T
    off = S[~np.eye(len(clusters), dtype=bool)]
    cut = float(np.percentile(off, 97))
    for a, c in enumerate(clusters):
        order = [b for b in np.argsort(-S[a]) if b != a and S[a, b] >= cut][:most]
        c.related = [clusters[b] for b in order]


# ---------------------------------------------------------------- scoring
def _recency(rows: list[dict]) -> tuple:
    """Per row: is it in the most recent third of ITS source's period, and how likely that is for the source.

    Sources cover different periods (Stack Exchange goes further back), so growth is measured within each source.
    Review sources stay neutral (0 observed, 0 expected): "most helpful" and "latest 30" feeds say nothing about trends."""
    recent = np.zeros(len(rows), dtype=bool)
    expected = np.zeros(len(rows))
    for src in {r["source"] for r in rows} - set(REVIEW_SOURCES):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        t = np.array([rows[i]["created_at"] or 0 for i in idx])
        cut = t.min() + (t.max() - t.min()) * 2 / 3
        is_recent = t >= cut
        recent[idx] = is_recent
        expected[idx] = is_recent.mean()
    return recent, expected


def score_clusters(rows: list[dict], clusters: list[Cluster], detailed: bool) -> None:
    recent_flag, expected = _recency(rows)
    for c in clusters:
        m = c.members
        c.authors = len({rows[i]["author"] for i in m})
        c.threads = len({rows[i]["thread_id"] for i in m})
        if detailed:
            c.pay_explicit = float(np.mean([rows[i]["wtp"] == "explicit" for i in m]))
            c.pay_implied = float(np.mean([rows[i]["wtp"] == "implied" for i in m]))
        else:  # older data: only the regex money signal
            c.pay_explicit = float(np.mean([rows[i]["money"] for i in m]))
        # observed recent excerpts vs. what the cluster's sources would give on average (smoothed)
        c.growth = (float(recent_flag[m].sum()) + 1) / (float(expected[m].sum()) + 1)
        growth = min(max(c.growth, 0.5), 2.0)
        c.sources = dict(Counter(rows[i]["source"] for i in m).most_common())
        reviewed: dict = {}
        for i in m:
            if rows[i].get("reviewed"):
                reviewed.setdefault(rows[i]["reviewed"], set()).add(rows[i]["author"])
        c.reviewed = sorted(((p, len(a)) for p, a in reviewed.items()), key=lambda t: -t[1])[:8]
        # many different people, in many different discussions and communities, who say they'd pay, recently
        c.score = (math.log1p(c.authors) * math.log1p(c.threads)
                   * (1 + 1.5 * c.pay_explicit + 0.5 * c.pay_implied) * growth
                   * (1 + 0.3 * (len(c.sources) - 1)))
        c.final = c.score * (0.6 if c.mixed else 1.0)
        prods = Counter(p for i in m + c.builders for p in rows[i]["products"])
        c.products = prods.most_common(6)
        launches = {}
        for i in m:
            if rows[i]["in_show_hn"]:
                launches[rows[i]["title"]] = _thread_url(rows[i]["thread_id"])
        c.show_hn = sorted(launches.items())[:6]


def pick_representatives(X: np.ndarray, rows: list[dict], members: list[int], k: int = 12) -> list[int]:
    sims = _sims(X, members, _centroid(X, members))
    order = [members[i] for i in np.argsort(-sims)]
    picked, seen = [], set()
    for i in order:  # one excerpt per author
        if rows[i]["author"] in seen:
            continue
        picked.append(i)
        seen.add(rows[i]["author"])
        if len(picked) >= k:
            break
    return picked


def pick_quotes(X: np.ndarray, rows: list[dict], c: Cluster, k: int = 4) -> list[int]:
    """Quotes for the brief: the most typical excerpt of each source first, then the next most typical ones."""
    sims = _sims(X, c.members, _centroid(X, c.members))
    order = [c.members[i] for i in np.argsort(-sims)]
    picked, seen_src, seen_auth = [], set(), set()
    for i in order:
        if rows[i]["source"] not in seen_src:
            picked.append(i)
            seen_src.add(rows[i]["source"])
            seen_auth.add(rows[i]["author"])
    for i in order:
        if len(picked) >= k:
            break
        if i not in picked and rows[i]["author"] not in seen_auth:
            picked.append(i)
            seen_auth.add(rows[i]["author"])
    return picked[:max(k, len(seen_src))]


# ---------------------------------------------------------------- LLM summary
SUMMARY_SYSTEM = """You summarize a cluster of real excerpts from online discussions (Hacker News, Stack Exchange,
Bluesky) and low-star reviews of App Store apps and WordPress plugins, in which different people describe similar
needs. Describe what the excerpts say. Do NOT judge the market or the business potential: a later step checks
competitors on the web. A review is about an existing product that falls short: name that product as a current
solution and say what it lacks.

Return:
- title: a short, specific name for the NEED, max 8 words. Good: "Buy-to-own video without subscriptions".
  Bad: "Analysis of excerpts", "Shared problem: ...".
- problem: the most specific problem that most excerpts share, in 1-2 sentences.
- who: the specific group of people who have it (not "users").
- wants: what they ask for: features, constraints, price points; only what the excerpts say (1-2 sentences).
- current_solutions: what they use or mention today, including the listed launches and builders,
  or "none mentioned".
- verify: 2-4 existing products or product categories someone should look up before building this.
- coherence 1-5: 5 = all excerpts describe the same problem; 3 = same area but different problems; 1 = unrelated."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "problem": {"type": "string"},
        "who": {"type": "string"},
        "wants": {"type": "string"},
        "current_solutions": {"type": "string"},
        "verify": {"type": "array", "items": {"type": "string"}},
        "coherence": {"type": "integer"},
    },
    "required": ["title", "problem", "who", "wants", "current_solutions", "verify", "coherence"],
}


def _clamp(value, lo: int, hi: int, default: int) -> int:
    try:
        return int(min(max(int(value), lo), hi))
    except (TypeError, ValueError):
        return default


def summarize(c: Cluster, rows: list[dict], model: str, mode: str) -> dict:
    lines = []
    for n, i in enumerate(c.reps, 1):
        r = rows[i]
        text = r["snippet"][:400]
        if mode == "llm" and r["problem"]:
            text += f"\n   (need: {r['problem']})"
        lines.append(f"{n}. [thread: {r['title'][:80]}] {text}")
    context = []
    if c.products:
        context.append("Products mentioned in the excerpts: " + ", ".join(f"{p} ({k}x)" for p, k in c.products))
    if c.show_hn:
        context.append("Product launches (Show HN) whose threads contain these excerpts: "
                       + "; ".join(t for t, _ in c.show_hn))
    if c.reviewed:
        context.append("Products whose low-star reviews are among these excerpts (reviewers): "
                       + ", ".join(f"{p} ({k})" for p, k in c.reviewed))
    for i in c.builders[:4]:
        context.append(f"Someone building a solution says: {rows[i]['snippet'][:250]}")
    user = (f"Cluster of {c.n} excerpts from {c.authors} different people in {c.threads} threads. "
            f"Keywords: {', '.join(c.keywords)}.\n\nRepresentative excerpts:\n" + "\n".join(lines)
            + ("\n\nWhat already exists:\n" + "\n".join(context) if context else
               "\n\nNo existing products were mentioned."))
    out = llm.chat_json(model, SUMMARY_SYSTEM, user, SUMMARY_SCHEMA, temperature=0.2, max_tokens=2048)
    title = str(out.get("title") or "").strip().strip('"')
    out["title"] = title if len(title) <= 70 else title[:70].rsplit(" ", 1)[0] + "…"
    out["coherence"] = _clamp(out.get("coherence"), 1, 5, 3)
    verify = out.get("verify") if isinstance(out.get("verify"), list) else []
    out["verify"] = [str(v).strip() for v in verify if str(v).strip()][:4]
    return out


# ---------------------------------------------------------------- main entry
PER_SOURCE = 5  # besides the top list, describe the strongest needs of each smaller source


def per_source_picks(main: list[Cluster], rows: list[dict], exclude: set, k: int = PER_SOURCE) -> dict:
    """{source: [clusters]}: for each source except the biggest, the clusters it contributes most people to."""
    totals = Counter(r["source"] for r in rows)
    out = {}
    for src, _ in totals.most_common()[1:]:
        ranked = sorted((c for c in main if c.sources.get(src, 0) >= 2 and id(c) not in exclude),
                        key=lambda c: (-c.sources.get(src, 0), -c.score))
        if ranked:
            out[src] = ranked[:k]
    return out


def run_analyze(con, mode: str = "auto", embed: str = "auto", embed_model: str = llm.DEFAULT_EMBED,
                llm_model: str = llm.DEFAULT_LLM, min_cluster_size: int | None = None, top: int = 40,
                summary: bool = True, out_dir: str = "reports", seed: int = 42):
    rows, mode, total, checked, detailed = load_rows(con, mode)
    groups = Counter(r["group"] for r in rows)
    log(f"analyze: mode={mode} ({checked}/{total} excerpts checked by LLM), {len(rows)} excerpts used"
        + (f" ({groups['main']} software/service ideas, {groups['other']} other, "
           f"{groups['builder']} from people building a solution)" if detailed else ""))
    if groups["main"] < 30:
        log("not enough excerpts to cluster (need at least 30). Collect or classify more data first.")
        return None
    texts = [r["problem"] if mode == "llm" else neutralize(r["snippet"]) for r in rows]

    if embed == "auto":
        embed = "ollama" if llm.available(embed_model) else "tfidf"
        if embed == "tfidf":
            log(f"Ollama embedding model '{embed_model}' not available, using TF-IDF instead")
    if embed == "ollama":
        llm.ensure_model(embed_model)
        X = embed_ollama(con, texts, embed_model)
    else:
        X = embed_tfidf(texts, seed=seed)

    by_group = {g: [i for i, r in enumerate(rows) if r["group"] == g] for g in ("main", "other", "builder")}
    main, info = cluster_group(X, by_group["main"], "main", min_cluster_size, seed)
    if not main:
        log("clustering " + info)
        log("no clusters found; try a smaller --min-cluster-size or collect more data")
        return None
    back = attach_unassigned(X, main, by_group["main"])
    info += f"; {back} of them close enough to join a cluster afterwards"
    log("clustering " + info)
    other, info_other = cluster_group(X, by_group["other"], "other", min_cluster_size, seed)
    if other:
        info_other += f"; {attach_unassigned(X, other, by_group['other'])} joined afterwards"
    if by_group["other"]:
        log("clustering " + info_other)
    attach_builders(X, main, by_group["builder"])

    kw_texts = [neutralize(r["snippet"]) + (" " + r["problem"] if r["problem"] else "") for r in rows]

    def prepare(clusters: list[Cluster], everything: list[Cluster]) -> None:
        for c, kw in zip(everything, cluster_keywords(kw_texts, [c.members for c in everything])):
            c.keywords = kw
        score_clusters(rows, clusters, detailed)
        for c in clusters:
            c.reps = pick_representatives(X, rows, c.members)
            c.quotes = pick_quotes(X, rows, c)
            aud = Counter(rows[i]["audience"] for i in c.members if rows[i]["audience"])
            c.audiences = [a for a, _ in aud.most_common(3)]

    prepare(main + other, main + other)
    main.sort(key=lambda c: -c.score)
    other.sort(key=lambda c: -c.score)

    def describe(clusters: list[Cluster]) -> None:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(summarize, c, rows, llm_model, mode) for c in clusters]
            failed = []
            for c, f in zip(clusters, futs):
                try:
                    c.summary = f.result()
                except Exception as e:
                    log(f"  description failed for a cluster ({e}); will retry")
                    failed.append(c)
        for c in failed:  # one retry each, one at a time (failures are often Ollama running out of memory)
            time.sleep(3)
            try:
                c.summary = summarize(c, rows, llm_model, mode)
            except Exception as e:
                log(f"  description failed again: {e}")

    summarized = False
    extra: dict = {}
    if summary:
        if llm.available(llm_model):
            head = main[:top]
            extra = per_source_picks(main, rows, {id(c) for c in head})
            todo = head + [c for cs in extra.values() for c in cs]
            log(f"describing {len(todo)} clusters with {llm_model} (top {len(head)} + the strongest of each "
                f"smaller source)")
            describe(todo)
            # clusters whose excerpts don't share one problem: split in two and describe the parts
            for c in [c for c in head if c.mixed and c.n >= 10]:
                parts = split_cluster(X, c, seed)
                if len(parts) > 1:
                    log(f"  splitting a mixed cluster ({c.n} excerpts) and describing the parts")
                    head = [h for h in head if h is not c] + parts
                    main = [m for m in main if m is not c] + parts
                    prepare(parts, main + other)
                    describe(parts)
            summarized = True
        else:
            log(f"skipping descriptions: Ollama model '{llm_model}' not available")
    # ranking uses evidence only (people, threads, communities, would pay, growth); a cluster whose excerpts
    # don't share one problem (coherence 1-2) counts x0.6 because its numbers mix several needs
    for c in main:
        c.final = c.score * (0.6 if c.mixed else 1.0)
    main.sort(key=lambda c: -c.final)
    if summarized:
        top = len(head)
        main = sorted(head, key=lambda c: -c.final) + [c for c in main if all(c is not h for h in head)]
        extra = per_source_picks(main, rows, {id(c) for c in head})
        extra = {s: [c for c in cs if c.summary] for s, cs in extra.items()}
    link_related(X, main)

    meta = {"mode": mode, "detailed": detailed, "excerpts": len(rows), "excerpts_total": total,
            "sources": dict(Counter(r["source"] for r in rows)),
            "llm_checked": checked, "groups": dict(groups),
            "embedding": embed_model if embed == "ollama" else "tfidf", "clustering": info,
            "clustering_other": info_other if by_group["other"] else None,
            "summaries": llm_model if summarized else None, "top": top}
    path = write_report(out_dir, main, other, rows, meta, extra)
    log(f"report written to {path}/report.md")
    return path
