"""Turn texts into vectors: Ollama embedding model (best) or TF-IDF + SVD (no LLM needed)."""
from __future__ import annotations

import hashlib

import numpy as np

from . import llm
from .patterns import DOMAIN_STOP_WORDS
from .util import log


def _key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def embed_ollama(con, texts: list[str], model: str = llm.DEFAULT_EMBED, batch: int = 32) -> np.ndarray:
    prefix = "clustering: " if "nomic" in model else ""  # nomic-embed-text expects a task prefix
    inputs = [prefix + t for t in texts]
    keys = [_key(model, t) for t in inputs]
    vecs: list = [None] * len(texts)
    for i in range(0, len(keys), 500):  # cache lookup (SQLite variable limit)
        chunk = keys[i:i + 500]
        q = f"SELECT key, vec FROM emb_cache WHERE key IN ({','.join('?' * len(chunk))})"
        found = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in con.execute(q, chunk)}
        for j, k in enumerate(chunk):
            if k in found:
                vecs[i + j] = found[k]
    todo = [i for i, v in enumerate(vecs) if v is None]
    log(f"embeddings: {len(texts) - len(todo)} cached, {len(todo)} to compute with {model}")
    for b in range(0, len(todo), batch):
        idx = todo[b:b + batch]
        out = llm.embed(model, [inputs[i] for i in idx])
        for i, v in zip(idx, out):
            arr = np.asarray(v, dtype=np.float32)
            vecs[i] = arr
            con.execute("INSERT OR REPLACE INTO emb_cache (key, dim, vec) VALUES (?,?,?)",
                        (keys[i], int(arr.size), arr.tobytes()))
        con.commit()
        if (b // batch) % 20 == 19:
            log(f"  {b + len(idx)}/{len(todo)}")
    X = np.vstack(vecs).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


def embed_tfidf(texts: list[str], dims: int = 100, seed: int = 42) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS), ngram_range=(1, 2), min_df=2,
        max_df=0.2, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#]+\b",
    )
    X = vec.fit_transform(texts)
    k = max(2, min(dims, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    return normalize(Z).astype(np.float32)
