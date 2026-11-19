from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("data") / "ideafinder.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          TEXT PRIMARY KEY,   -- e.g. hn:12345
    source      TEXT NOT NULL,      -- hn (more sources later)
    kind        TEXT,               -- story | comment
    author      TEXT,
    created_at  INTEGER,            -- unix time
    title       TEXT,               -- own title (story) or the thread's title (comment)
    text        TEXT,
    url         TEXT,               -- link to the item itself
    thread_id   TEXT,
    points      INTEGER,
    query       TEXT,               -- search phrase that found it first
    fetched_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at);

CREATE TABLE IF NOT EXISTS processed (   -- items already scanned by the regex extractor
    item_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS pains (
    item_id    TEXT PRIMARY KEY,
    snippet    TEXT NOT NULL,       -- the sentences that express the need
    patterns   TEXT NOT NULL,       -- json list of matched pattern names
    money      INTEGER NOT NULL DEFAULT 0,  -- 1 if it mentions paying / prices
    llm_model  TEXT,                -- NULL = not checked by an LLM yet
    is_pain    INTEGER,             -- LLM verdict 1/0
    problem    TEXT,                -- LLM: generalized one-sentence problem
    audience   TEXT,
    category   TEXT
);

CREATE TABLE IF NOT EXISTS emb_cache (
    key TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# Columns added after the first version; existing databases get them on connect.
ADDED_COLUMNS = {
    "pains": {
        "schema_version": "INTEGER",   # version of the LLM fields (see classify.SCHEMA_VERSION)
        "solution_type": "TEXT",       # software / hardware / service / content / policy / other
        "buildable": "INTEGER",        # 1 = one developer could ship a first version in ~3 months
        "wtp": "TEXT",                 # willingness to pay: none / implied / explicit
        "speaker": "TEXT",             # has_problem / building_solution / observer
        "products": "TEXT",            # json list of products mentioned
    },
}


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    for table, cols in ADDED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, kind in cols.items():
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    con.commit()
    return con


def get_meta(con: sqlite3.Connection, key: str):
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def set_meta(con: sqlite3.Connection, key: str, value) -> None:
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value)))
