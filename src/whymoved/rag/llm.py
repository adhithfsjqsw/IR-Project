"""Swappable LLM backends. Default: local Ollama (free, offline). Add a backend by registering a class in BACKENDS."""
from __future__ import annotations

import os

import requests


class LLMError(RuntimeError):
    pass


class OllamaBackend:
    def __init__(self, model: str = "qwen2.5:7b", host: str | None = None, temperature: float = 0.1,
                 timeout: int = 300, max_tokens: int | None = None) -> None:
        self.max_tokens = max_tokens
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, system: str, prompt: str) -> str:
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "options": {"temperature": self.temperature, "num_ctx": 8192,
                              **({"num_predict": self.max_tokens} if self.max_tokens else {})}}
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed ({self.host}, model {self.model}): {e}. "
                           "Is Ollama running and the model pulled?") from e
        return r.json()["message"]["content"].strip()


BACKENDS = {"ollama": OllamaBackend}


def make_backend(cfg: dict):
    name = cfg.get("backend", "ollama")
    if name not in BACKENDS:
        raise LLMError(f"unknown RAG backend {name!r}; available: {sorted(BACKENDS)}")
    kw = {k: cfg[k] for k in ("model", "host", "temperature", "timeout", "max_tokens") if k in cfg}
    return BACKENDS[name](**kw)
