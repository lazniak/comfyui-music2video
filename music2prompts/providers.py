"""LLM backends: LM Studio (local, default) plus OpenRouter, OpenAI and Anthropic.

Every client exposes the same ``chat_json`` signature as :class:`LMStudioClient`,
so :class:`~music2prompts.llm_stages.StageRunner` never knows which one it drives.
Only ``requests`` is used - no vendor SDK, nothing new to install inside ComfyUI.

The ``probe_*_raw`` functions do the actual network lookups; nothing calls them
directly on the schema-building path. They are driven by :mod:`.model_cache`, which
keeps one TTL cache shared by the node schema and the pack's HTTP route, so a
dropdown can refresh without restarting ComfyUI.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .lmstudio import DEFAULT_URL as LMSTUDIO_URL
from .lmstudio import FALLBACK_MODELS as LMSTUDIO_FALLBACK
from .lmstudio import LMStudioClient
from .util import PREFIX, clamp_seed, extract_json, log, warn

LLM_PROVIDERS = ["lmstudio", "openrouter", "openai", "anthropic"]

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

# Environment variables checked, in order, when a key widget is left empty.
# Both spellings of the OpenRouter variable are accepted, and fal's admin key
# works wherever FAL_KEY does.
KEY_ENV = {
    "openrouter": ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPENROUTER_KEY"),
    "openai": ("OPENAI_API_KEY", "OPEN_AI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "fal": ("FAL_KEY", "FAL_API_KEY", "FAL_ADMIN_API_KEY"),
}

# Used when a provider cannot be reached while the schema is built.
FALLBACK_OPENROUTER = [
    "anthropic/claude-sonnet-5",
    "google/gemini-3-pro",
    "openai/gpt-5.2",
    "qwen/qwen3-max",
]
FALLBACK_OPENAI = ["gpt-5.2", "gpt-5.1", "gpt-5-mini", "gpt-4.1"]
FALLBACK_ANTHROPIC = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "claude-opus-4-8",
]


class ProviderError(RuntimeError):
    pass


# --------------------------------------------------------------------------- keys


def resolve_key(provider: str, explicit: str = "") -> str:
    """Widget value first, then the provider's environment variables."""
    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    for name in KEY_ENV.get(provider, ()):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


# --------------------------------------------------------------------------- schema helpers


def strictify(schema: Any) -> Any:
    """Make a JSON schema acceptable to OpenAI strict structured outputs.

    OpenAI rejects object schemas that allow extra properties or leave a property
    out of ``required``. LM Studio does not care, so the same schemas are reused
    here with the missing bits filled in.
    """
    if not isinstance(schema, dict):
        return schema
    out = {key: strictify(value) if key != "properties" else value for key, value in schema.items()}
    if out.get("type") == "object":
        properties = out.get("properties")
        if isinstance(properties, dict):
            out["properties"] = {name: strictify(value) for name, value in properties.items()}
            out["required"] = list(properties.keys())
            out["additionalProperties"] = False
    items = out.get("items")
    if isinstance(items, dict):
        out["items"] = strictify(items)
    return out


def _split_data_uri(uri: str) -> tuple[str, str]:
    """``data:image/png;base64,AAA`` -> ``("image/png", "AAA")``."""
    if not uri.startswith("data:"):
        raise ProviderError(f"{PREFIX} expected a data: URI for an image, got {uri[:40]!r}")
    header, _, payload = uri.partition(",")
    media_type = header[5:].split(";", 1)[0] or "image/png"
    return media_type, payload


# --------------------------------------------------------------------------- clients


class _CloudClient:
    """Shared plumbing: retries, JSON parsing, no-op model lifecycle."""

    name = "cloud"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 300,
        retries: int = 2,
        verbose: bool = False,
        ledger=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout = max(10, int(timeout))
        self.retries = max(0, int(retries))
        self.verbose = bool(verbose)
        #: Where the cost of every reply is recorded. None = nobody is counting.
        self.ledger = ledger
        self._stage = ""
        self._attempt = 1
        if not self.api_key:
            raise ProviderError(
                f"{PREFIX} no API key for '{self.name}'. Paste one into the node or set "
                f"{'/'.join(KEY_ENV.get(self.name, ('the provider key',)))} in the environment."
            )

    # lifecycle hooks exist so the node can treat every provider the same way
    def ensure_model(self, *args, **kwargs) -> None:
        return None

    def unload(self, *args, **kwargs) -> None:
        return None

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _post(self, path: str, payload: dict) -> Any:
        import requests

        url = f"{self.base_url}{path}"
        try:
            response = requests.post(
                url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout
            )
        except requests.exceptions.Timeout as exc:
            raise ProviderError(f"{PREFIX} {self.name} timed out after {self.timeout}s on {path}.") from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"{PREFIX} {self.name} unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"{PREFIX} {self.name} returned HTTP {response.status_code} for {path}: {response.text[:400]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"{PREFIX} {self.name} returned no JSON: {response.text[:200]}") from exc
        # Recorded here rather than around chat_json because an attempt can answer HTTP 200
        # and still raise further down (bad JSON, a refusal, an empty block) - those tokens
        # were generated, so they were charged, and a recorder further out would miss them.
        self._record(payload, body)
        return body

    def _record(self, payload: dict, body: Any) -> None:
        """Best-effort: no accounting problem is worth losing a paid-for reply over."""
        if self.ledger is None:
            return
        try:
            usage = (body or {}).get("usage") if isinstance(body, dict) else None
            self.ledger.record_llm(
                self.name, str(payload.get("model") or ""), usage or {}, self._stage, self._attempt
            )
        except Exception as exc:  # pragma: no cover - defensive
            warn(f"could not record the cost of a {self.name} call: {exc}")

    def _once(self, system: str, user: str, schema: dict | None, images: list[str] | None, **kwargs) -> Any:
        raise NotImplementedError

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict | None = None,
        images: list[str] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        seed: int | None = None,
        stage: str = "stage",
        reasoning_effort: str = "none",
    ) -> Any:
        attempts = self.retries + 1
        last_error: Exception | None = None
        self._stage = stage
        for attempt in range(attempts):
            # the final retry drops the schema and parses the reply loosely
            use_schema = schema if (attempts == 1 or attempt < attempts - 1) else None
            self._attempt = attempt + 1
            try:
                return self._once(
                    system=system,
                    user=user,
                    schema=schema,
                    images=images,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                    structured=use_schema is not None,
                )
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    warn(f"{stage} failed on {self.name} ({exc}); retrying ({attempt + 1}/{attempts - 1})")
                    time.sleep(1.5 * (attempt + 1))
        raise ProviderError(f"{PREFIX} stage '{stage}' failed on {self.name} after {attempts} attempts: {last_error}")


class OpenAICompatClient(_CloudClient):
    """OpenAI and OpenRouter: both speak ``/chat/completions``."""

    def __init__(self, *args, name: str = "openai", **kwargs) -> None:
        self.name = name
        super().__init__(*args, **kwargs)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.name == "openrouter":
            headers["X-Title"] = "ComfyUI Music2Prompts"
            headers["HTTP-Referer"] = "https://github.com/lazniak/comfyui-ultimate-node"
        return headers

    def _once(self, system, user, schema, images, *, model, temperature, max_tokens, seed, reasoning_effort, structured):
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}]
            for uri in images:
                content.append({"type": "image_url", "image_url": {"url": uri}})

        instruction = system
        if schema is not None and not structured:
            instruction = system + "\n\nReply with raw JSON only. No prose, no code fences."

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if seed is not None:
            payload["seed"] = clamp_seed(seed)
        effort = (reasoning_effort or "").lower()
        if effort in {"low", "medium", "high"}:
            payload["reasoning_effort"] = effort
        if structured and schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": strictify(schema)},
            }

        response = self._post("/chat/completions", payload)
        text = self._first_message(response)
        if self.verbose:
            log(f"{self.name}: {len(text)} chars returned")
        try:
            return extract_json(text)
        except ValueError as exc:
            raise ProviderError(f"{exc}; reply began: {text[:200]!r}") from exc

    @staticmethod
    def _first_message(response: Any) -> str:
        if not isinstance(response, dict):
            raise ProviderError(f"{PREFIX} unexpected chat response: {response!r}")
        if response.get("error"):
            raise ProviderError(f"{PREFIX} provider error: {str(response['error'])[:300]}")
        choices = response.get("choices") or []
        if not choices:
            raise ProviderError(f"{PREFIX} chat response contained no choices: {str(response)[:300]}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        if not text:
            text = choices[0].get("text", "") or message.get("reasoning") or ""
        if not text:
            if str(choices[0].get("finish_reason", "")).lower() == "length":
                raise ProviderError(f"{PREFIX} the model hit the token limit before answering; raise lm_max_tokens.")
            raise ProviderError(f"{PREFIX} chat response was empty.")
        return str(text)


class AnthropicClient(_CloudClient):
    """Claude via the Messages API; structured stages use forced tool use."""

    name = "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _once(self, system, user, schema, images, *, model, temperature, max_tokens, seed, reasoning_effort, structured):
        blocks: list[dict[str, Any]] = []
        for uri in images or []:
            media_type, data = _split_data_uri(uri)
            blocks.append(
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
            )
        blocks.append({"type": "text", "text": user})

        instruction = system
        if schema is not None and not structured:
            instruction = system + "\n\nReply with raw JSON only. No prose, no code fences."

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens),
            "system": instruction,
            "messages": [{"role": "user", "content": blocks}],
        }
        # temperature is rejected by the current Claude models, so it is never sent
        if structured and schema is not None:
            payload["tools"] = [
                {
                    "name": "result",
                    "description": "Return the requested fields.",
                    "input_schema": schema,
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": "result"}

        response = self._post("/messages", payload)
        if not isinstance(response, dict):
            raise ProviderError(f"{PREFIX} unexpected Anthropic response: {response!r}")
        content = response.get("content") or []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                return block["input"]
        text = "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            if response.get("stop_reason") == "refusal":
                raise ProviderError(f"{PREFIX} Claude declined this request ({response.get('stop_details')}).")
            raise ProviderError(f"{PREFIX} Anthropic reply was empty: {str(response)[:300]}")
        if self.verbose:
            log(f"anthropic: {len(text)} chars returned")
        try:
            return extract_json(text)
        except ValueError as exc:
            raise ProviderError(f"{exc}; reply began: {text[:200]!r}") from exc


def make_llm_client(
    provider: str,
    *,
    lm_url: str = LMSTUDIO_URL,
    api_key: str = "",
    timeout: int = 300,
    retries: int = 2,
    verbose: bool = False,
    ledger=None,
):
    """Build the client for ``provider``; the key falls back to the environment."""
    provider = (provider or "lmstudio").strip().lower()
    if provider == "lmstudio":
        return LMStudioClient(
            lm_url, api_key, timeout=timeout, retries=retries, verbose=verbose, ledger=ledger
        )
    key = resolve_key(provider, api_key)
    if provider == "openrouter":
        return OpenAICompatClient(
            OPENROUTER_URL, key, timeout, retries, verbose, ledger=ledger, name="openrouter"
        )
    if provider == "openai":
        return OpenAICompatClient(OPENAI_URL, key, timeout, retries, verbose, ledger=ledger, name="openai")
    if provider == "anthropic":
        return AnthropicClient(ANTHROPIC_URL, key, timeout, retries, verbose, ledger=ledger)
    raise ProviderError(f"{PREFIX} unknown LLM provider '{provider}'. Pick one of {LLM_PROVIDERS}.")


# --------------------------------------------------------------------------- model probing


def _get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 2.5) -> Any:
    import requests

    response = requests.get(url, headers=headers or {}, timeout=timeout)
    if response.status_code >= 400:
        raise ProviderError(f"HTTP {response.status_code}")
    return response.json()


def probe_lmstudio_raw() -> list[str]:
    """Model keys served by LM Studio right now. Never raises."""
    return LMStudioClient.probe_model_keys(LMSTUDIO_URL, timeout=0.6)


def probe_openrouter_llms_raw() -> list[str]:
    """Text-only models on OpenRouter (public endpoint, no key needed)."""
    data = _get_json(f"{OPENROUTER_URL}/models")
    ids = []
    for model in data.get("data", []):
        architecture = model.get("architecture") or {}
        outputs = architecture.get("output_modalities") or ["text"]
        if "text" in outputs and "image" not in outputs:
            ids.append(model.get("id"))
    return sorted(item for item in ids if item)


def probe_openai_models_raw() -> list[str]:
    key = resolve_key("openai")
    if not key:
        return []
    data = _get_json(f"{OPENAI_URL}/models", {"Authorization": f"Bearer {key}"})
    skip = ("embedding", "tts", "transcribe", "whisper", "image", "audio", "realtime", "moderation", "dall-e")
    ids = [
        model.get("id")
        for model in data.get("data", [])
        if str(model.get("id", "")).startswith(("gpt", "o1", "o3", "o4", "chatgpt"))
        and not any(token in str(model.get("id", "")) for token in skip)
    ]
    return sorted(item for item in ids if item)


def probe_anthropic_models_raw() -> list[str]:
    key = resolve_key("anthropic")
    if not key:
        return []
    data = _get_json(
        f"{ANTHROPIC_URL}/models?limit=100",
        {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION},
    )
    return [model.get("id") for model in data.get("data", []) if model.get("id")]


LLM_KINDS = {
    "lmstudio": "lmstudio",
    "openrouter": "openrouter_llm",
    "openai": "openai_llm",
    "anthropic": "anthropic_llm",
}


def llm_model_options(provider: str) -> list[str]:
    """Cached model list for one provider (never touches the network itself)."""
    from . import model_cache

    kind = LLM_KINDS.get((provider or "").lower())
    return model_cache.snapshot(kind) if kind else []
