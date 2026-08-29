"""Optional rendering layer: turn the written prompts into images and video.

Two providers, both plain REST over ``requests``:

* **fal.ai** - queue API (``POST https://queue.fal.run/<model>``, then poll).
* **OpenRouter** - ``POST /api/v1/images`` (synchronous) and ``POST /api/v1/videos``
  (asynchronous, polled).

Nothing here runs unless the node's ``image_provider`` / ``video_provider`` is set
to something other than ``none`` - these are paid, per-call APIs.
"""

from __future__ import annotations

import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .providers import OPENROUTER_URL, ProviderError, resolve_key
from .util import PREFIX, log, warn

FAL_QUEUE_URL = "https://queue.fal.run"
FAL_MODEL_INDEX = "https://fal.ai/api/models"

MEDIA_PROVIDERS = ["none", "fal", "openrouter"]

FALLBACK_FAL_IMAGE = [
    "fal-ai/flux/dev",
    "fal-ai/flux-2-pro",
    "fal-ai/qwen-image",
    "fal-ai/z-image/turbo",
    "fal-ai/nano-banana-pro",
]
FALLBACK_FAL_VIDEO = [
    "minimax/h3/image-to-video",
    "minimax/h3/reference-to-video",
    "minimax/h3/text-to-video",
    "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
    "fal-ai/wan/v2.7/image-to-video",
]
FALLBACK_OPENROUTER_IMAGE = [
    "google/gemini-3.1-flash-image",
    "black-forest-labs/flux.2-pro",
    "openai/gpt-image-2",
]
FALLBACK_OPENROUTER_VIDEO = [
    "minimax/hailuo-3",
    "google/veo-3.1-fast",
    "alibaba/wan-3.0-prime",
]

# fal video endpoints that take subject references rather than a first frame
REFERENCE_HINTS = ("reference-to-video", "ref2v", "/reference")

_PROBE_CACHE: dict[str, list[str]] = {}


class RenderError(RuntimeError):
    pass


# --------------------------------------------------------------------------- requests


@dataclass
class ImageRequest:
    prompt: str
    negative: str = ""
    aspect_ratio: str = "16:9"
    seed: int | None = None
    references: list[str] = field(default_factory=list)  # data: URIs
    label: str = ""


@dataclass
class VideoRequest:
    prompt: str
    seconds: float = 6.0
    aspect_ratio: str = "16:9"
    seed: int | None = None
    first_frame: str = ""  # data: URI
    references: list[str] = field(default_factory=list)  # data: URIs
    label: str = ""


# --------------------------------------------------------------------------- helpers


def aspect_to_size(aspect_ratio: str, base: int = 1024) -> tuple[int, int]:
    """Pixel size for an aspect ratio, rounded to a multiple of 32."""
    try:
        left, right = str(aspect_ratio).split(":")
        ratio = float(left) / float(right)
    except Exception:
        ratio = 16.0 / 9.0
    width, height = base * ratio**0.5, base / ratio**0.5
    snap = lambda value: max(256, int(round(value / 32.0)) * 32)  # noqa: E731
    return snap(width), snap(height)


def data_uri(payload: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"


def image_bytes_to_tensor(payload: bytes):
    """PNG/JPEG bytes -> ComfyUI IMAGE tensor ``[1, H, W, 3]`` in 0..1."""
    import io as _io

    import numpy as np
    import torch
    from PIL import Image

    with Image.open(_io.BytesIO(payload)) as handle:
        image = handle.convert("RGB")
        array = np.asarray(image, dtype="float32") / 255.0
    return torch.from_numpy(array)[None, ...]


def output_directory(subfolder: str = "music2prompts") -> str:
    try:
        import folder_paths  # type: ignore

        base = folder_paths.get_output_directory()
    except Exception:
        base = os.getcwd()
    path = os.path.join(base, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def _in_parallel(jobs: list[Callable[[], Any]], concurrency: int, label: str) -> list[Any]:
    """Run jobs concurrently, keep order, turn failures into ``None``."""
    if not jobs:
        return []
    workers = max(1, min(int(concurrency or 1), len(jobs)))
    results: list[Any] = [None] * len(jobs)

    def guarded(index: int) -> None:
        try:
            results[index] = jobs[index]()
        except Exception as exc:
            warn(f"{label} {index + 1}/{len(jobs)} failed: {exc}")

    if workers == 1:
        for index in range(len(jobs)):
            guarded(index)
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(guarded, range(len(jobs))))
    return results


def _download(url: str, headers: dict[str, str] | None = None, timeout: int = 300) -> bytes:
    import requests

    response = requests.get(url, headers=headers or {}, timeout=timeout)
    if response.status_code >= 400:
        raise RenderError(f"{PREFIX} download failed ({response.status_code}) for {url[:120]}")
    return response.content


# --------------------------------------------------------------------------- fal.ai


class FalClient:
    """fal.ai queue API. Optional payload fields are dropped when a model rejects them."""

    name = "fal"

    def __init__(self, api_key: str = "", timeout: int = 600, poll_seconds: float = 2.0, verbose: bool = False) -> None:
        self.api_key = resolve_key("fal", api_key)
        self.timeout = max(30, int(timeout))
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.verbose = bool(verbose)
        if not self.api_key:
            raise RenderError(f"{PREFIX} no fal.ai key. Paste one into the node or set FAL_KEY in the environment.")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"}

    def run(self, model: str, payload: dict, optional_keys: tuple[str, ...] = ()) -> dict:
        """Submit, poll and return the model output."""
        import requests

        attempt_payload = dict(payload)
        droppable = [key for key in optional_keys if key in attempt_payload]
        while True:
            response = requests.post(
                f"{FAL_QUEUE_URL}/{model.strip('/')}",
                headers=self._headers(),
                data=json.dumps(attempt_payload),
                timeout=120,
            )
            if response.status_code in (400, 422) and droppable:
                dropped = droppable.pop(0)  # optional_keys is ordered most-droppable first
                attempt_payload.pop(dropped, None)
                warn(f"fal rejected '{dropped}' for {model}; retrying without it")
                continue
            if response.status_code >= 400:
                raise RenderError(f"{PREFIX} fal HTTP {response.status_code} for {model}: {response.text[:300]}")
            break

        queued = response.json()
        status_url = queued.get("status_url")
        response_url = queued.get("response_url")
        if not status_url or not response_url:
            raise RenderError(f"{PREFIX} fal did not queue the request: {str(queued)[:200]}")

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            status = requests.get(status_url, headers=self._headers(), timeout=60)
            if status.status_code >= 400:
                raise RenderError(f"{PREFIX} fal status HTTP {status.status_code}: {status.text[:200]}")
            state = str(status.json().get("status", "")).upper()
            if state == "COMPLETED":
                result = requests.get(response_url, headers=self._headers(), timeout=120)
                if result.status_code >= 400:
                    raise RenderError(f"{PREFIX} fal result HTTP {result.status_code}: {result.text[:200]}")
                return result.json()
            if state in {"FAILED", "ERROR", "CANCELLED"}:
                raise RenderError(f"{PREFIX} fal reported {state} for {model}: {status.text[:300]}")
            time.sleep(self.poll_seconds)
        raise RenderError(f"{PREFIX} fal did not finish {model} within {self.timeout}s.")

    # ---------------------------------------------------------------- media

    def image(self, model: str, request: ImageRequest) -> bytes:
        width, height = aspect_to_size(request.aspect_ratio)
        payload: dict[str, Any] = {
            "prompt": request.prompt,
            "num_images": 1,
            "image_size": {"width": width, "height": height},
            "output_format": "png",
            "enable_safety_checker": False,
        }
        if request.negative:
            payload["negative_prompt"] = request.negative
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.references:
            payload["image_urls"] = list(request.references)
        result = self.run(
            model,
            payload,
            optional_keys=(
                "negative_prompt",
                "enable_safety_checker",
                "output_format",
                "num_images",
                "image_urls",
                "image_size",
                "seed",
            ),
        )
        return _first_media(result, ("images", "image"), self._headers())

    def video(self, model: str, request: VideoRequest) -> bytes:
        payload: dict[str, Any] = {
            "prompt": request.prompt,
            "duration": max(1, int(round(request.seconds))),
        }
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        wants_references = any(hint in model for hint in REFERENCE_HINTS)
        if wants_references and request.references:
            payload["reference_image_urls"] = list(request.references)
            payload["aspect_ratio"] = request.aspect_ratio
        elif request.first_frame:
            payload["image_url"] = request.first_frame
        else:
            payload["aspect_ratio"] = request.aspect_ratio
        result = self.run(
            model,
            payload,
            optional_keys=("aspect_ratio", "seed", "duration", "reference_image_urls", "image_url"),
        )
        return _first_media(result, ("video", "videos"), self._headers())


def _first_media(result: Any, keys: tuple[str, ...], headers: dict[str, str]) -> bytes:
    """Pull the first file out of a fal result and return its bytes."""
    if not isinstance(result, dict):
        raise RenderError(f"{PREFIX} unexpected fal result: {str(result)[:200]}")
    candidate: Any = None
    for key in keys:
        value = result.get(key)
        if isinstance(value, list) and value:
            candidate = value[0]
            break
        if isinstance(value, dict):
            candidate = value
            break
    if candidate is None:
        raise RenderError(f"{PREFIX} fal result had none of {keys}: {str(result)[:200]}")
    url = candidate.get("url") if isinstance(candidate, dict) else str(candidate)
    if not url:
        raise RenderError(f"{PREFIX} fal result had no url: {str(candidate)[:200]}")
    if url.startswith("data:"):
        return base64.b64decode(url.partition(",")[2])
    return _download(url, headers)


# --------------------------------------------------------------------------- OpenRouter media


class OpenRouterMediaClient:
    name = "openrouter"

    def __init__(self, api_key: str = "", timeout: int = 600, poll_seconds: float = 3.0, verbose: bool = False) -> None:
        self.api_key = resolve_key("openrouter", api_key)
        self.timeout = max(30, int(timeout))
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.verbose = bool(verbose)
        if not self.api_key:
            raise RenderError(
                f"{PREFIX} no OpenRouter key. Paste one into the node or set OPENROUTER_API_KEY in the environment."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "ComfyUI Music2Prompts",
        }

    @staticmethod
    def _references(uris: list[str]) -> list[dict]:
        return [{"type": "image_url", "image_url": {"url": uri}} for uri in uris if uri]

    def image(self, model: str, request: ImageRequest) -> bytes:
        import requests

        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "n": 1,
            "aspect_ratio": request.aspect_ratio,
            "output_format": "png",
        }
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.references:
            payload["input_references"] = self._references(request.references)
        response = requests.post(
            f"{OPENROUTER_URL}/images", headers=self._headers(), data=json.dumps(payload), timeout=self.timeout
        )
        if response.status_code >= 400:
            raise RenderError(f"{PREFIX} OpenRouter image HTTP {response.status_code}: {response.text[:300]}")
        data = (response.json() or {}).get("data") or []
        if not data:
            raise RenderError(f"{PREFIX} OpenRouter returned no image: {response.text[:200]}")
        first = data[0]
        if first.get("b64_json"):
            return base64.b64decode(first["b64_json"])
        url = first.get("url") or (first.get("image_url") or {}).get("url")
        if not url:
            raise RenderError(f"{PREFIX} OpenRouter image had neither b64_json nor url: {str(first)[:200]}")
        return _download(url, self._headers())

    def video(self, model: str, request: VideoRequest) -> bytes:
        import requests

        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "duration": max(1, int(round(request.seconds))),
            "aspect_ratio": request.aspect_ratio,
        }
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.first_frame:
            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": request.first_frame},
                    "frame_type": "first_frame",
                }
            ]
        elif request.references:
            payload["input_references"] = self._references(request.references)

        response = requests.post(
            f"{OPENROUTER_URL}/videos", headers=self._headers(), data=json.dumps(payload), timeout=120
        )
        if response.status_code >= 400:
            raise RenderError(f"{PREFIX} OpenRouter video HTTP {response.status_code}: {response.text[:300]}")
        job = response.json() or {}
        poll_url = job.get("polling_url") or f"{OPENROUTER_URL}/videos/{job.get('id')}"

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            status = requests.get(poll_url, headers=self._headers(), timeout=60)
            if status.status_code >= 400:
                raise RenderError(f"{PREFIX} OpenRouter poll HTTP {status.status_code}: {status.text[:200]}")
            body = status.json() or {}
            state = str(body.get("status", "")).lower()
            if state == "completed":
                urls = body.get("unsigned_urls") or body.get("urls") or []
                if not urls:
                    raise RenderError(f"{PREFIX} OpenRouter finished without a video url: {str(body)[:200]}")
                return _download(urls[0], self._headers())
            if state in {"failed", "cancelled", "canceled", "error"}:
                raise RenderError(f"{PREFIX} OpenRouter video {state}: {str(body)[:300]}")
            time.sleep(self.poll_seconds)
        raise RenderError(f"{PREFIX} OpenRouter did not finish the video within {self.timeout}s.")


# --------------------------------------------------------------------------- front door


def make_media_client(provider: str, api_key: str = "", timeout: int = 600, verbose: bool = False):
    provider = (provider or "none").strip().lower()
    if provider in {"", "none", "off"}:
        return None
    if provider == "fal":
        return FalClient(api_key, timeout=timeout, verbose=verbose)
    if provider == "openrouter":
        return OpenRouterMediaClient(api_key, timeout=timeout, verbose=verbose)
    raise RenderError(f"{PREFIX} unknown media provider '{provider}'. Pick one of {MEDIA_PROVIDERS}.")


def render_images(client, model: str, requests_: list[ImageRequest], concurrency: int = 2) -> list[bytes | None]:
    if client is None or not requests_:
        return []
    log(f"rendering {len(requests_)} image(s) with {client.name}/{model} (concurrency {concurrency})")
    jobs = [lambda request=request: client.image(model, request) for request in requests_]
    return _in_parallel(jobs, concurrency, "image")


def render_videos(client, model: str, requests_: list[VideoRequest], concurrency: int = 2) -> list[bytes | None]:
    if client is None or not requests_:
        return []
    log(f"rendering {len(requests_)} video(s) with {client.name}/{model} (concurrency {concurrency})")
    jobs = [lambda request=request: client.video(model, request) for request in requests_]
    return _in_parallel(jobs, concurrency, "video")


def save_videos(payloads: list[bytes | None], prefix: str = "music2prompts") -> list[str]:
    """Write finished clips into ComfyUI's output folder; returns their paths."""
    directory = output_directory()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths: list[str] = []
    for index, payload in enumerate(payloads):
        if not payload:
            continue
        path = os.path.join(directory, f"{prefix}_{stamp}_shot{index + 1:03d}.mp4")
        try:
            with open(path, "wb") as handle:
                handle.write(payload)
            paths.append(path)
        except OSError as exc:
            warn(f"could not write {path}: {exc}")
    return paths


# --------------------------------------------------------------------------- model probing


def _cached(name: str, probe, fallback: list[str]) -> list[str]:
    if name in _PROBE_CACHE:
        return _PROBE_CACHE[name]
    try:
        found = [str(item) for item in probe() if item]
    except Exception:
        found = []
    result = found or list(fallback)
    _PROBE_CACHE[name] = result
    return result


def _fal_index(categories: tuple[str, ...], page_size: int = 60) -> list[str]:
    import requests

    ids: list[str] = []
    for category in categories:
        response = requests.get(
            FAL_MODEL_INDEX, params={"categories": category, "page_size": page_size}, timeout=3.0
        )
        if response.status_code >= 400:
            continue
        for item in (response.json() or {}).get("items", []):
            model_id = item.get("id")
            if model_id and model_id not in ids:
                ids.append(model_id)
    return ids


def probe_fal_images() -> list[str]:
    return _cached("fal_image", lambda: _fal_index(("text-to-image",)), FALLBACK_FAL_IMAGE)


def probe_fal_videos() -> list[str]:
    return _cached(
        "fal_video", lambda: _fal_index(("image-to-video", "text-to-video")), FALLBACK_FAL_VIDEO
    )


def _openrouter_media_models(kind: str) -> list[str]:
    import requests

    response = requests.get(f"{OPENROUTER_URL}/{kind}/models", timeout=3.0)
    if response.status_code >= 400:
        raise ProviderError(f"HTTP {response.status_code}")
    return [model.get("id") for model in (response.json() or {}).get("data", []) if model.get("id")]


def probe_openrouter_images() -> list[str]:
    return _cached("or_image", lambda: _openrouter_media_models("images"), FALLBACK_OPENROUTER_IMAGE)


def probe_openrouter_videos() -> list[str]:
    return _cached("or_video", lambda: _openrouter_media_models("videos"), FALLBACK_OPENROUTER_VIDEO)

