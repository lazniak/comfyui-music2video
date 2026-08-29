"""Minimal LM Studio client (REST v1 + OpenAI-compatible chat).

Only ``requests`` is used - no vendor SDK. Everything degrades gracefully:
the v1 native API is preferred, the older v0 API is the fallback, and the
OpenAI-compatible ``/v1/chat/completions`` endpoint does the inference.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .util import PREFIX, extract_json, log, warn

DEFAULT_URL = "http://127.0.0.1:1234"
FALLBACK_MODELS = ["google/gemma-4-e4b"]


@dataclass
class ModelInfo:
    key: str
    display_name: str = ""
    kind: str = "llm"
    loaded: bool = False
    max_context_length: int = 0
    loaded_context_length: int = 0
    vision: bool = False
    instance_ids: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return self.display_name or self.key


class LMStudioError(RuntimeError):
    pass


class LMStudioClient:
    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        api_key: str = "",
        timeout: int = 300,
        retries: int = 2,
        verbose: bool = False,
    ) -> None:
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout = max(10, int(timeout))
        self.retries = max(0, int(retries))
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------ plumbing

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: int | None = None) -> Any:
        import requests

        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method,
                url,
                headers=self._headers(),
                data=json.dumps(payload) if payload is not None else None,
                timeout=timeout or self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise LMStudioError(
                f"{PREFIX} LM Studio unreachable at {self.base_url} - start the server "
                "(LM Studio -> Developer -> Start Server) or change lm_url."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LMStudioError(f"{PREFIX} LM Studio timed out after {timeout or self.timeout}s on {path}.") from exc

        if response.status_code >= 400:
            raise LMStudioError(
                f"{PREFIX} LM Studio returned HTTP {response.status_code} for {path}: {response.text[:400]}"
            )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    # ------------------------------------------------------------------ models

    def list_models(self) -> list[ModelInfo]:
        try:
            data = self._request("GET", "/api/v1/models", timeout=min(30, self.timeout))
            models = data.get("models") if isinstance(data, dict) else None
            if isinstance(models, list):
                return [self._model_from_v1(item) for item in models if isinstance(item, dict)]
        except LMStudioError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            warn(f"v1 model listing failed ({exc}); falling back to /api/v0/models")

        data = self._request("GET", "/api/v0/models", timeout=min(30, self.timeout))
        entries = data.get("data") if isinstance(data, dict) else None
        return [self._model_from_v0(item) for item in entries or [] if isinstance(item, dict)]

    @staticmethod
    def _model_from_v1(item: dict) -> ModelInfo:
        capabilities = item.get("capabilities")
        vision = bool(capabilities.get("vision")) if isinstance(capabilities, dict) else False
        instances = item.get("loaded_instances")
        instances = instances if isinstance(instances, list) else []
        loaded_ctx = 0
        instance_ids: list[str] = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            if instance.get("id"):
                instance_ids.append(str(instance["id"]))
            config = instance.get("config") if isinstance(instance.get("config"), dict) else {}
            loaded_ctx = max(
                loaded_ctx,
                int(config.get("context_length") or instance.get("context_length") or 0),
            )
        return ModelInfo(
            key=str(item.get("key") or item.get("id") or ""),
            display_name=str(item.get("display_name") or ""),
            kind=str(item.get("type") or "llm"),
            loaded=bool(instances),
            max_context_length=int(item.get("max_context_length") or 0),
            loaded_context_length=loaded_ctx,
            vision=vision,
            instance_ids=tuple(instance_ids),
        )

    @staticmethod
    def _model_from_v0(item: dict) -> ModelInfo:
        capabilities = item.get("capabilities")
        vision = str(item.get("type", "")).lower() == "vlm"
        if isinstance(capabilities, list):
            vision = vision or "vision" in [str(c).lower() for c in capabilities]
        return ModelInfo(
            key=str(item.get("id") or ""),
            display_name=str(item.get("id") or ""),
            kind=str(item.get("type") or "llm"),
            loaded=str(item.get("state", "")).lower() == "loaded",
            max_context_length=int(item.get("max_context_length") or 0),
            loaded_context_length=int(item.get("loaded_context_length") or 0),
            vision=vision,
        )

    @staticmethod
    def probe_model_keys(base_url: str = DEFAULT_URL, timeout: float = 0.8) -> list[str]:
        """Best-effort model list. Never raises.

        Both API versions are tried at the same time: when LM Studio is not running
        each one costs a full connect timeout, and paying that twice in a row was the
        single slowest thing in the pack.
        """
        endpoints = (
            ("/api/v1/models", lambda d: [m.get("key") for m in d.get("models", [])]),
            ("/api/v0/models", lambda d: [m.get("id") for m in d.get("data", [])]),
        )

        def fetch(item) -> list[str]:
            path, extract = item
            try:
                import requests

                response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
                if response.status_code >= 400:
                    return []
                return [str(key) for key in extract(response.json()) if key]
            except Exception:
                return []

        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as pool:
                for keys in pool.map(fetch, endpoints):
                    if keys:
                        return keys
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------ lifecycle

    def ensure_model(
        self,
        model_key: str,
        auto_download: bool = True,
        auto_load: bool = True,
        context_length: int = 32768,
        progress: Callable[[str], None] | None = None,
    ) -> ModelInfo | None:
        """Make sure ``model_key`` exists locally and is loaded with enough context."""
        notify = progress or (lambda message: log(message))
        try:
            models = {model.key: model for model in self.list_models()}
        except LMStudioError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            warn(f"could not list models ({exc}); continuing without lifecycle management")
            return None

        info = models.get(model_key)
        if info is None:
            if not auto_download:
                warn(f"model '{model_key}' is not installed in LM Studio and lm_auto_download is off.")
                return None
            notify(f"downloading '{model_key}' in LM Studio...")
            self.download(model_key, progress=notify)
            models = {model.key: model for model in self.list_models()}
            info = models.get(model_key)

        if info is not None and auto_load:
            needs_load = not info.loaded
            needs_context = info.loaded and 0 < info.loaded_context_length < int(context_length)
            if needs_load or needs_context:
                if needs_context:
                    notify(
                        f"reloading '{model_key}' with context {context_length} "
                        f"(currently {info.loaded_context_length})"
                    )
                    self.unload(model_key, info.instance_ids)
                else:
                    notify(f"loading '{model_key}' (context {context_length})...")
                self.load(model_key, context_length)
        return info

    def download(self, model_key: str, progress: Callable[[str], None] | None = None) -> None:
        notify = progress or (lambda message: log(message))
        response = self._request("POST", "/api/v1/models/download", {"model": model_key}, timeout=60)
        job_id = None
        if isinstance(response, dict):
            for key in ("job_id", "jobId", "id", "download_id", "downloadId"):
                if response.get(key):
                    job_id = str(response[key])
                    break
        if not job_id:
            notify(f"download request accepted for '{model_key}' (no job id returned).")
            return

        deadline = time.time() + max(600, self.timeout * 4)
        last_report = 0.0
        while time.time() < deadline:
            status = self._request("GET", f"/api/v1/models/download/status/{job_id}", timeout=30)
            if not isinstance(status, dict):
                break
            state = str(status.get("status") or status.get("state") or "").lower()
            progress_value = status.get("progress")
            if isinstance(progress_value, (int, float)):
                percent = progress_value * 100.0 if progress_value <= 1.0 else float(progress_value)
                if time.time() - last_report > 5.0:
                    notify(f"downloading '{model_key}': {percent:.0f}%")
                    last_report = time.time()
            if state in {"completed", "complete", "finished", "success", "done"} or status.get("completed") is True:
                notify(f"download of '{model_key}' finished.")
                return
            if state in {"failed", "error", "cancelled", "canceled"}:
                raise LMStudioError(f"{PREFIX} LM Studio failed to download '{model_key}': {status}")
            time.sleep(2.0)
        warn(f"download of '{model_key}' did not report completion in time; continuing.")

    def load(self, model_key: str, context_length: int = 32768) -> None:
        payload = {"model": model_key, "config": {"context_length": int(context_length)}}
        try:
            self._request("POST", "/api/v1/models/load", payload, timeout=max(120, self.timeout))
            return
        except LMStudioError as exc:
            warn(f"load with context config failed ({exc}); retrying without config")
        try:
            self._request("POST", "/api/v1/models/load", {"model": model_key}, timeout=max(120, self.timeout))
        except LMStudioError as exc:
            # not fatal: LM Studio loads the model just-in-time on the first completion
            warn(f"could not preload '{model_key}' ({exc}); relying on just-in-time loading")

    def unload(self, model_key: str, instance_ids: tuple[str, ...] | None = None) -> None:
        """Unload every loaded instance of a model (the API keys this by instance id)."""
        targets = list(instance_ids or ())
        if not targets:
            try:
                targets = [
                    instance
                    for model in self.list_models()
                    if model.key == model_key
                    for instance in (model.instance_ids or ((model_key,) if model.loaded else ()))
                ]
            except Exception:
                targets = []
        if not targets:
            return
        for instance in targets:
            try:
                self._request("POST", "/api/v1/models/unload", {"instance_id": instance}, timeout=60)
            except LMStudioError as exc:
                warn(f"could not unload '{instance}': {exc}")

    # ------------------------------------------------------------------ inference

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
        """Chat completion that must return JSON. Returns the parsed object."""
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}]
            for uri in images:
                content.append({"type": "image_url", "image_url": {"url": uri}})

        base_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if seed is not None:
            base_payload["seed"] = int(seed)
        # Reasoning models (gemma-4, qwen3, ...) otherwise spend the whole token budget
        # on reasoning_content and return an empty message.
        if reasoning_effort and reasoning_effort.lower() not in {"", "default", "auto"}:
            base_payload["reasoning_effort"] = reasoning_effort.lower()

        attempts = self.retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            payload = dict(base_payload)
            # keep the JSON schema on every attempt but the last one; with retries disabled
            # there is no "last one" to fall back to, so keep it there too
            use_schema = schema is not None and (attempts == 1 or attempt < attempts - 1)
            if use_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "strict": True, "schema": schema},
                }
            elif schema is not None:
                payload["messages"] = [
                    {"role": "system", "content": system + "\n\nReply with raw JSON only. No prose, no code fences."},
                    {"role": "user", "content": content},
                ]
            try:
                try:
                    response = self._request("POST", "/v1/chat/completions", payload)
                except LMStudioError as exc:
                    if "reasoning" not in str(exc).lower() or "reasoning_effort" not in payload:
                        raise
                    warn("this model rejected 'reasoning_effort'; retrying without it")
                    base_payload.pop("reasoning_effort", None)
                    payload.pop("reasoning_effort", None)
                    response = self._request("POST", "/v1/chat/completions", payload)
                text = self._first_message(response)
                if self.verbose:
                    log(f"{stage}: {len(text)} chars returned")
                try:
                    return extract_json(text)
                except ValueError as exc:
                    raise LMStudioError(f"{exc}; reply began: {text[:200]!r}") from exc
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    warn(f"{stage} failed ({exc}); retrying ({attempt + 1}/{attempts - 1})")
                    time.sleep(1.5 * (attempt + 1))
        raise LMStudioError(f"{PREFIX} stage '{stage}' failed after {attempts} attempts: {last_error}")

    @staticmethod
    def _first_message(response: Any) -> str:
        if not isinstance(response, dict):
            raise LMStudioError(f"{PREFIX} unexpected chat response: {response!r}")
        choices = response.get("choices")
        if not choices:
            raise LMStudioError(f"{PREFIX} chat response contained no choices: {str(response)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if isinstance(text, list):  # some builds return content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        if not text:
            text = choice.get("text", "")
        if not text:
            # Reasoning models can spend the whole budget before writing an answer.
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning and "{" in reasoning:
                warn("model answered inside reasoning_content; parsing that instead")
                return str(reasoning)
            if str(choice.get("finish_reason", "")).lower() == "length":
                raise LMStudioError(
                    f"{PREFIX} the model hit the token limit before answering. Raise lm_max_tokens or "
                    "keep lm_reasoning_effort at 'none'."
                )
            raise LMStudioError(f"{PREFIX} chat response was empty.")
        return str(text)
