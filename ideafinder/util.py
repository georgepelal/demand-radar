from __future__ import annotations

import html
import re
import sys
import time

_TAG = re.compile(r"<[^>]+>")
_PARA = re.compile(r"(?i)<p>|<br\s*/?>")


def html_to_text(s: str | None) -> str:
    """Convert the small subset of HTML that Hacker News uses into plain text."""
    if not s:
        return ""
    s = _PARA.sub("\n", s)
    s = _TAG.sub("", s)
    s = html.unescape(s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[ \t ]+", " ", s)
    return s.strip()


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, file=sys.stderr, flush=True)
