"""`python -m ideafinder doctor --model X`: check that an Ollama model works with this tool, and how fast."""
from __future__ import annotations

import json
import platform
import subprocess
import time

import requests

from . import llm
from .classify import SCHEMA, SYSTEM

SAMPLE = (
    'Thread title: Ask HN: What tools do you wish existed?\n\nExcerpt:\n"""\n'
    "The whole space is so scammy that I'd pay for a good USB-C cable tester that tells me "
    'the real speed and wattage of a cable.\n"""'
)
TINY_SCHEMA = {"type": "object", "properties": {"greeting": {"type": "string"}}, "required": ["greeting"]}


def _stream(payload: dict, timeout: int) -> dict:
    """Send one streaming chat request and time it, even if the model never answers."""
    t0 = time.time()
    res = {"first": None, "content": "", "thinking": "", "final": {}, "error": None}
    try:
        with requests.post(f"{llm.OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(10, timeout)) as r:
            if r.status_code != 200:
                res["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    if "error" in d:
                        res["error"] = str(d["error"])[:200]
                        break
                    msg = d.get("message") or {}
                    if res["first"] is None and (msg.get("content") or msg.get("thinking")):
                        res["first"] = time.time() - t0
                    res["content"] += msg.get("content") or ""
                    res["thinking"] += msg.get("thinking") or ""
                    if d.get("done"):
                        res["final"] = d
                        break
                    if time.time() - t0 > timeout:
                        res["error"] = f"gave up after {timeout}s"
                        break
    except requests.RequestException as e:
        res["error"] = f"no answer ({e.__class__.__name__}) after {time.time() - t0:.0f}s"
    res["total"] = time.time() - t0
    return res


def _report(name: str, res: dict) -> None:
    f, ns = res["final"], 1e9
    parts = [f"total {res['total']:.1f}s"]
    if res["first"] is not None:
        parts.append(f"first token {res['first']:.1f}s")
    if f:
        n, ev = f.get("eval_count", 0), f.get("eval_duration", 0) / ns
        parts.append(f"load {f.get('load_duration', 0) / ns:.1f}s")
        parts.append(f"prompt {f.get('prompt_eval_duration', 0) / ns:.1f}s")
        parts.append(f"{n} tokens" + (f" at {n / ev:.1f} tok/s" if ev else ""))
    if res["thinking"]:
        parts.append(f"thinking {len(res['thinking'])} chars")
    if res["error"]:
        parts.append(f"ERROR: {res['error']}")
    print(f"  {name:<18} " + " · ".join(parts))
    if res["content"].strip():
        print("  " + " " * 18 + " -> " + res["content"].strip().replace("\n", " ")[:150])


def _mac_memory_gb():
    if platform.system() != "Darwin":
        return None
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    except (ValueError, OSError):
        return None


def _print_loaded(prefix: str) -> None:
    try:
        running = requests.get(f"{llm.OLLAMA_URL}/api/ps", timeout=5).json().get("models", [])
    except (requests.RequestException, ValueError):
        running = []
    for m in running:
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        gpu = f"{100 * vram / size:.0f}% GPU" if size else "?"
        print(f"{prefix}{m.get('name')} · runner {m.get('runner', '?')} · {size / 2**30:.1f} GB · {gpu}"
              f" · context {m.get('context_length', '?')}")


def run_doctor(model: str, timeout: int = 120) -> None:
    try:
        version = requests.get(f"{llm.OLLAMA_URL}/api/version", timeout=5).json().get("version")
    except (requests.RequestException, ValueError):
        raise llm.OllamaError(f"Cannot reach Ollama at {llm.OLLAMA_URL}") from None
    llm.ensure_model(model)
    caps = llm.capabilities(model)
    thinking = "thinking" in caps
    mem = _mac_memory_gb()
    print(f"Ollama {version} · {model} · capabilities: {', '.join(caps) or '?'}"
          + (f" · Mac memory {mem:.0f} GB" if mem else ""))

    def payload(messages, num_predict, fmt=None, think=None):
        p = {"model": model, "messages": messages, "stream": True,
             "options": {"temperature": 0, "num_ctx": llm.NUM_CTX, "num_predict": num_predict}}
        if fmt is not None:
            p["format"] = fmt
        if think is not None:
            p["think"] = think
        return p

    hello = [{"role": "user", "content": "Reply with just the word OK."}]
    tests = [("1. plain", payload(hello, 60))]
    if thinking:
        tests.append(("2. think=false", payload(hello, 60, think=False)))
    no_think = False if thinking else None
    tests.append(("3. JSON schema", payload([{"role": "user", "content": "Say hello."}], 200, TINY_SCHEMA, no_think)))
    tests.append(("4. real excerpt", payload([{"role": "system", "content": SYSTEM},
                                               {"role": "user", "content": SAMPLE}], 1024, SCHEMA, no_think)))
    _print_loaded("Already loaded: ")
    print("Loading the model (can take a few minutes the first time)...")
    t0 = time.time()
    try:  # an empty generate request just loads the model, with the same context size the tool uses
        r = requests.post(f"{llm.OLLAMA_URL}/api/generate",
                          json={"model": model, "keep_alive": "10m", "options": {"num_ctx": llm.NUM_CTX}},
                          timeout=300)
        load_error = None if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.RequestException as e:
        load_error = (f"{e.__class__.__name__}. Run `ollama ps`: if a model is stuck on 'Stopping...', "
                      "restart Ollama (quit it from the menu bar and open it again).")
    print(f"  {'0. load':<18} {time.time() - t0:.1f}s" + (f" · ERROR: {load_error}" if load_error else ""))
    if load_error:
        return
    print(f"Running {len(tests)} tests (each gives up after {timeout}s):")
    for name, p in tests:
        res = _stream(p, timeout)
        _report(name, res)
        if res["first"] is None and res["error"]:
            print("  stopping: the model produced nothing at all.")
            break

    _print_loaded("Loaded: ")
