"""Minimal client for a local Ollama server (https://ollama.com). Nothing leaves your machine."""
from __future__ import annotations

import json
import os
import re

import requests

OLLAMA_URL = os.environ.get("IDEAFINDER_OLLAMA_URL", "http://localhost:11434").rstrip("/")
DEFAULT_LLM = os.environ.get("IDEAFINDER_LLM", "qwen2.5:7b")
DEFAULT_EMBED = os.environ.get("IDEAFINDER_EMBED", "nomic-embed-text")
NUM_CTX = 8192  # keep the same for every call, otherwise Ollama reloads the model


class OllamaError(RuntimeError):
    pass


def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_URL} ({e.__class__.__name__}). "
            "Open the Ollama app or run `ollama serve`, or use --no-llm."
        ) from None
    return [m.get("name", "") for m in r.json().get("models", [])]


def ensure_model(name: str) -> None:
    models = list_models()
    if name not in models and f"{name}:latest" not in models:
        raise OllamaError(f"Ollama model '{name}' is not installed. Run:  ollama pull {name}")


def available(name: str) -> bool:
    try:
        ensure_model(name)
        return True
    except OllamaError:
        return False


_CAPS: dict = {}


def capabilities(model: str) -> list:
    """What the model supports according to Ollama, e.g. ["completion", "thinking"]."""
    if model not in _CAPS:
        try:
            r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=10)
            _CAPS[model] = r.json().get("capabilities", []) if r.ok else []
        except (requests.RequestException, ValueError):
            _CAPS[model] = []
    return _CAPS[model]


def chat_json(model: str, system: str, user: str, schema: dict, temperature: float = 0.0,
              max_tokens: int = 1024, timeout: int = 600, examples: list | None = None) -> dict:
    """examples: [(user_text, assistant_json_text), ...] shown before the real question (few-shot)."""
    messages = [{"role": "system", "content": system}]
    for u, a in examples or []:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "format": schema,          # structured output (Ollama >= 0.5)
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX, "num_predict": max_tokens},
    }
    if "thinking" in capabilities(model):
        # Reasoning models (gemma4, qwen3, deepseek-r1...) would otherwise "think" for
        # hundreds or thousands of tokens before every short JSON answer.
        payload["think"] = False
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from None
    if r.status_code != 200:
        raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    content = r.json().get("message", {}).get("content", "")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        raise OllamaError(f"model did not return JSON: {content[:200]!r}") from None


def embed(model: str, texts: list[str], timeout: int = 600) -> list[list[float]]:
    try:
        r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": model, "input": texts}, timeout=timeout)
    except requests.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from None
    if r.status_code != 200:
        raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    out = r.json().get("embeddings")
    if not out or len(out) != len(texts):
        raise OllamaError("unexpected embedding response from Ollama")
    return out
