"""P5 - LLM providers.

Three implementations behind one protocol, selected by `LLM_PROVIDER`:

* `stub`   - offline, no key, no network. The default for tests and the offline demo.
* `local`  - an OpenAI-compatible endpoint such as Ollama or llama.cpp.
* `hosted` - a hosted API; the key is read from the environment only, never from a file
             in the repo, and never logged.

`temperature=0` everywhere. This assistant states published facts; a non-zero
temperature would make the same question produce different numbers on different days,
which is the opposite of what a cited facts tool should do.
"""

from __future__ import annotations

import os
import time
from typing import Protocol, runtime_checkable

from src.config import Settings, get_settings
from src.llm.prompt_builder import parse_context_blocks

__all__ = [
    "HostedProvider",
    "LLMProvider",
    "LocalProvider",
    "ProviderError",
    "StubProvider",
    "get_provider",
]

MAX_TOKENS = 300
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _first_meaningful_line(text: str, section: str = "", scheme_name: str = "") -> str:
    """First line of a retrieved chunk that actually says something.

    Chunk text repeats its heading as line 1, so quoting verbatim would answer
    "HDFC Large Cap Fund - Direct - Growth - Exit load" and tell the user nothing. Only the
    *heading* line is skipped: content lines legitimately begin with the section name
    ("Exit load of 1% if redeemed within 1 year.") and must be kept.
    """
    section = (section or "").strip()
    scheme_name = (scheme_name or "").strip()

    for index, line in enumerate((text or "").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in {section, scheme_name}:
            continue
        if index == 0 and section and section in stripped:
            continue
        return stripped
    return ""


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        """Return the model's completion for `prompt`."""
        ...

    def classify(self, prompt: str) -> str:
        """Return a single intent label. Optional; used only for P6's `unknown` residue."""
        ...


class StubProvider:
    """Offline provider that echoes the highest-scoring context extract.

    It never fabricates. The output is explicitly labelled as an extract so a stubbed demo
    can never be mistaken for a generated answer, and it is the provider the tests use, so
    the whole pipeline is verifiable with no key and no network.
    """

    name = "stub"
    needs_network = False

    LABEL = "From the official source (no LLM available):"

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = MAX_TOKENS) -> str:
        blocks = parse_context_blocks(prompt)
        if not blocks:
            return f"{self.LABEL}\nno context was retrieved for this question."

        best = blocks[0]
        body = _first_meaningful_line(
            best["text"], best.get("section", ""), best.get("scheme_name", "")
        )
        if not body:
            body = best["text"].strip()
        # The label is on its own line so the extract stays separable from the provenance.
        # Sharing a line made the whole response read as a footer, which hid the answer
        # from the sentence cap and the novelty check.
        return f"{self.LABEL}\n{body}\n\nsource: {best['url']}"

    def classify(self, prompt: str) -> str:
        # A stub cannot reason. Returning "unknown" keeps P6's fallback honest instead of
        # inventing a verdict that would then decide whether to refuse.
        return "unknown"


class LocalProvider:
    """OpenAI-compatible local endpoint (Ollama, llama.cpp, vLLM)."""

    name = "local"
    needs_network = True

    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = MAX_TOKENS) -> str:
        return self._complete(prompt, temperature, max_tokens)

    def classify(self, prompt: str) -> str:
        return self._complete(prompt, 0.0, 8).strip().lower()

    def _complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import urllib.error
        import urllib.request

        payload = _post_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                import json

                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"local provider failed: {type(exc).__name__}") from exc

        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("local provider returned an unexpected payload") from exc


class HostedProvider:
    """Hosted chat-completion API. Key comes from the environment only."""

    name = "hosted"
    needs_network = True

    def __init__(self, api_key: str, model: str, timeout: int = 60, endpoint: str | None = None):
        if not api_key:
            raise ProviderError("hosted provider requires an API key")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.endpoint = endpoint or os.environ.get(
            "LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"
        )

    def __repr__(self) -> str:  # pragma: no cover - defensive
        # Never let a key reach a log line or a traceback via repr().
        return f"HostedProvider(model={self.model!r}, api_key='***')"

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = MAX_TOKENS) -> str:
        return self._complete(prompt, temperature, max_tokens)

    def classify(self, prompt: str) -> str:
        return self._complete(prompt, 0.0, 8).strip().lower()

    def _complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import json
        import urllib.error
        import urllib.request

        payload = _post_json(
            self.endpoint,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        last_error: Exception | None = None
        for attempt in range(2):  # one retry
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS or attempt == 1:
                    # 401/403 must not be retried, and the body may echo the key.
                    raise ProviderError(f"hosted provider returned HTTP {exc.code}") from None
                time.sleep(1.0)
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    raise ProviderError(f"hosted provider failed: {type(exc).__name__}") from None
                time.sleep(1.0)
        else:  # pragma: no cover - loop always breaks or raises
            raise ProviderError(f"hosted provider failed: {last_error}")

        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("hosted provider returned an unexpected payload") from exc


def _post_json(url: str, payload: dict) -> bytes:
    import json

    return json.dumps(payload).encode("utf-8")


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the configured provider. `stub` needs no key and makes no network call."""
    cfg = settings or get_settings()
    choice = (cfg.llm_provider or "stub").strip().lower()

    if choice == "stub":
        return StubProvider()
    if choice == "local":
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
        return LocalProvider(base_url, cfg.llm_model or "llama3.1")
    if choice == "hosted":
        key = cfg.llm_api_key or os.environ.get("LLM_API_KEY") or ""
        return HostedProvider(key, cfg.llm_model or "gpt-4o-mini", timeout=cfg.request_timeout)

    raise ProviderError(f"unknown LLM_PROVIDER {choice!r}; expected one of hosted|local|stub")
