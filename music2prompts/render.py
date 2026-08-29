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

from . import cost as cost_module
from .providers import OPENROUTER_URL, ProviderError, resolve_key
from .util import PREFIX, clamp_seed, log, warn

FAL_QUEUE_URL = "https://queue.fal.run"
FAL_MODEL_INDEX = "https://fal.ai/api/models"
FAL_SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json"

MEDIA_PROVIDERS = ["none", "fal", "openrouter"]

#: How much of our prompt the endpoint is allowed to rewrite before generating.
EXPANSION_MODES = ["minimal", "model default", "rich"]

# Text-to-image endpoints: these render the first image of a run, and nothing else -
# not one of them declares an input for a reference image.
FALLBACK_FAL_IMAGE = [
    "fal-ai/flux/dev",
    "fal-ai/flux-2-pro",
    "fal-ai/qwen-image",
    "fal-ai/z-image/turbo",
    "fal-ai/nano-banana-pro",
]
# Edit endpoints: the only fal image models that can be handed an identity to keep.
# They *require* their image field, so they can never render the first frame of a run.
FALLBACK_FAL_IMAGE_EDIT = [
    "fal-ai/nano-banana-pro/edit",
    "fal-ai/flux-2-pro/edit",
    "fal-ai/qwen-image-edit-2511",
    "bytedance/seedream/v5/pro/edit",
    "fal-ai/flux-pro/kontext/multi",
]
FALLBACK_FAL_VIDEO = [
    "minimax/h3/image-to-video",
    "minimax/h3/reference-to-video",
    "minimax/h3/text-to-video",
    "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
    "fal-ai/wan/v2.7/image-to-video",  # the one default that takes driving audio
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

# Field names differ per endpoint - Wan wants `start_image_url` where MiniMax H3
# wants `image_url` - so the payload is built from the endpoint's own schema and
# these are only the candidates to look for in it.
FIRST_FRAME_FIELDS = (
    "image_url", "start_image_url", "first_frame_image", "start_image", "input_image_url", "image",
)
REFERENCE_FIELDS = (
    "reference_image_urls", "image_urls", "reference_images", "subject_reference_urls", "input_image_urls",
)
# Fields that take a *driving* audio track. `audio` is deliberately absent: on
# alibaba/wan-3.0-prime it is a boolean meaning "include generated audio", which is the
# opposite of handing the model a vocal to follow.
AUDIO_URL_FIELDS = ("audio_url", "audio", "speech_url", "voice_url")
AUDIO_LIST_FIELDS = ("reference_audio_urls", "audio_urls")


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
    #: The shot's own audio as a data: URI. Only sent to endpoints that declare an input
    #: for a driving audio track; on the rest there is nothing to send it to.
    audio: str = ""
    #: "minimal", "rich", or "" to leave the endpoint's own default alone.
    expansion: str = ""
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


def output_directory(subfolder: str = "music2prompts", temporary: bool = False) -> str:
    try:
        import folder_paths  # type: ignore

        base = folder_paths.get_temp_directory() if temporary else folder_paths.get_output_directory()
    except Exception:
        base = os.getcwd()
    path = os.path.join(base, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def _in_parallel(
    jobs: list[Callable[[], Any]],
    concurrency: int,
    label: str,
    errors: list[Exception] | None = None,
) -> list[Any]:
    """Run jobs concurrently, keep order, turn failures into ``None``.

    Failures are also appended to ``errors`` so the caller can tell "nothing rendered
    because the key is wrong" from "one shot tripped the safety checker".
    """
    if not jobs:
        return []
    workers = max(1, min(int(concurrency or 1), len(jobs)))
    results: list[Any] = [None] * len(jobs)

    def guarded(index: int) -> None:
        try:
            results[index] = jobs[index]()
        except Exception as exc:
            warn(f"{label} {index + 1}/{len(jobs)} failed: {exc}")
            if errors is not None:
                errors.append(exc)

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


# --------------------------------------------------------------------------- fal.ai schemas

_FAL_SCHEMAS: dict[str, dict] = {}


def fal_schema(model: str, timeout: float = 6.0) -> dict:
    """Input schema of one fal endpoint: ``{"properties": {...}, "required": [...]}``.

    fal publishes an OpenAPI document per endpoint, with no key required. Reading it
    is what lets the payload use the field names a given model actually has instead
    of guessing - and lets an impossible request fail before it is submitted.
    Returns an empty dict when the document cannot be read, which puts the caller
    back on the guess-and-drop path.
    """
    key = model.strip("/")
    if key in _FAL_SCHEMAS:
        return _FAL_SCHEMAS[key]
    schema: dict = {}
    try:
        import requests

        response = requests.get(FAL_SCHEMA_URL, params={"endpoint_id": key}, timeout=timeout)
        if response.status_code < 400:
            components = (response.json() or {}).get("components", {}).get("schemas", {})
            name = next((item for item in components if item.endswith("Input")), None)
            if name:
                schema = {
                    "properties": components[name].get("properties") or {},
                    "required": list(components[name].get("required") or []),
                }
    except Exception as exc:
        warn(f"could not read the fal schema for {key} ({exc}); falling back to default field names")
    _FAL_SCHEMAS[key] = schema
    return schema


def _branches(prop: Any) -> list[dict]:
    """A property plus every branch of its anyOf/oneOf, as plain dicts."""
    if not isinstance(prop, dict):
        return []
    found = [prop]
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in prop.get(key) or []:
            if isinstance(branch, dict):
                found.append(branch)
    return found


def _enum_of(prop: Any) -> list:
    for branch in _branches(prop):
        if branch.get("enum"):
            return list(branch["enum"])
    return []


def _bounds_of(prop: Any) -> tuple[float | None, float | None]:
    low = high = None
    for branch in _branches(prop):
        if branch.get("type") in {"integer", "number"}:
            low = branch.get("minimum", low)
            high = branch.get("maximum", high)
    return low, high


def _takes_object(prop: Any) -> bool:
    return any("$ref" in branch or branch.get("type") == "object" for branch in _branches(prop))


def _fit_enum(value: Any, options: list) -> Any:
    """Match our value against the endpoint's allowed values, ignoring case."""
    if not options:
        return value
    text = str(value).strip().lower()
    for option in options:
        if str(option).strip().lower() == text:
            return option
    return None


def _first_field(properties: dict, candidates: tuple[str, ...]) -> str:
    return next((name for name in candidates if name in properties), "")


def fal_image_reference_field(model: str) -> str:
    """The field this image endpoint takes reference images in, or "" if it takes none.

    Most fal image models are pure text-to-image and declare no image input at all: they
    physically cannot be told to keep a face or a look. Asking the schema is the only way
    to know before the money is spent.
    """
    properties = fal_schema(model).get("properties") or {}
    if not properties:
        return ""
    return _first_field(properties, REFERENCE_FIELDS) or _first_field(properties, FIRST_FRAME_FIELDS)


def fal_image_needs_image(model: str) -> str:
    """Name of the image field this endpoint *requires*, or "" when text alone is enough."""
    required = fal_schema(model).get("required") or []
    return next((name for name in required if name in REFERENCE_FIELDS + FIRST_FRAME_FIELDS), "")


def fal_audio_field(model: str) -> str:
    """The field this video endpoint takes a driving audio track in, or "".

    Only a real audio *input* counts. A boolean named ``audio`` means "generate a
    soundtrack" and is not one, so the schema's type is checked too.
    """
    properties = fal_schema(model).get("properties") or {}
    for name in AUDIO_LIST_FIELDS + AUDIO_URL_FIELDS:
        prop = properties.get(name)
        if prop is None:
            continue
        if any(branch.get("type") == "boolean" for branch in _branches(prop)):
            continue
        return name
    return ""


def fal_video_needs_image(model: str) -> str:
    """Name of the image field this endpoint requires, or "" if text alone is enough.

    Checked before the run starts, so an image-to-video model picked without an
    image provider fails immediately rather than after the LLM and the images
    have already been paid for.
    """
    required = fal_schema(model).get("required") or []
    return next((name for name in required if name in FIRST_FRAME_FIELDS + REFERENCE_FIELDS), "")


class _FalRejected(Exception):
    """fal refused the payload itself, at submit time or once the job ran."""

    def __init__(self, stage: str, status: int, text: str) -> None:
        super().__init__(text)
        self.stage, self.status, self.text = stage, status, text

    @property
    def missing(self) -> list[str]:
        """Field names the endpoint reported as required and absent."""
        try:
            detail = json.loads(self.text).get("detail")
        except Exception:
            return []
        names = []
        for item in detail or []:
            location = item.get("loc") if isinstance(item, dict) else None
            if isinstance(item, dict) and item.get("type") == "missing" and location:
                names.append(str(location[-1]))
        return names


def _repair_payload(payload: dict, missing: list[str], spare: dict, already: set[str]) -> list[str]:
    """Re-send an image we already hold under the field name this endpoint wants."""
    filled = []
    references = list(spare.get("references") or [])
    for name in missing:
        if name in already or name in payload:
            continue
        value: Any = None
        if name in FIRST_FRAME_FIELDS:
            value = spare.get("first_frame") or (references[0] if references else None)
        elif name in REFERENCE_FIELDS:
            value = references or ([spare["first_frame"]] if spare.get("first_frame") else None)
        if not value:
            continue
        payload[name] = value
        already.add(name)
        filled.append(name)
    return filled


class FalClient:
    """fal.ai queue API. Payloads are built from each endpoint's published schema."""

    name = "fal"

    def __init__(
        self,
        api_key: str = "",
        timeout: int = 600,
        poll_seconds: float = 2.0,
        verbose: bool = False,
        ledger=None,
    ) -> None:
        self.api_key = resolve_key("fal", api_key)
        self.timeout = max(30, int(timeout))
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.verbose = bool(verbose)
        #: Where each render's billed units are recorded. None = nobody is counting.
        self.ledger = ledger
        if not self.api_key:
            raise RenderError(f"{PREFIX} no fal.ai key. Paste one into the node or set FAL_KEY in the environment.")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"}

    def run(
        self,
        model: str,
        payload: dict,
        optional_keys: tuple[str, ...] = (),
        spare: dict[str, Any] | None = None,
        label: str = "",
        kind: str = "image",
    ) -> dict:
        """Submit, poll and return the model output.

        An endpoint can refuse the payload at submit time *or* only once the job
        runs, so both are funnelled into one retry loop: a field it asked for and
        we hold under another name is added, and anything it refuses is dropped.
        """
        attempt = dict(payload)
        droppable = [key for key in optional_keys if key in attempt]
        repaired: set[str] = set()
        for round_number in range(len(droppable) + 4):
            try:
                return self._attempt(model, attempt, label=label, kind=kind, attempt_number=round_number + 1)
            except _FalRejected as rejected:
                filled = _repair_payload(attempt, rejected.missing, spare or {}, repaired)
                if filled:
                    warn(f"{model} wanted {', '.join(filled)}; retrying under that name")
                    continue
                if droppable:
                    dropped = droppable.pop(0)  # optional_keys is ordered most-droppable first
                    attempt.pop(dropped, None)
                    warn(f"fal rejected '{dropped}' for {model}; retrying without it")
                    continue
                raise RenderError(
                    f"{PREFIX} fal refused the request for {model} "
                    f"(HTTP {rejected.status} at {rejected.stage}): {rejected.text[:300]}"
                ) from rejected
        raise RenderError(f"{PREFIX} fal kept refusing the request for {model}.")

    def _attempt(
        self, model: str, payload: dict, label: str = "", kind: str = "image", attempt_number: int = 1
    ) -> dict:
        """One submit-poll-collect round trip. Raises ``_FalRejected`` on 400/422."""
        import requests

        response = requests.post(
            f"{FAL_QUEUE_URL}/{model.strip('/')}",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=120,
        )
        if response.status_code in (400, 422):
            # refused by the gateway before a runner existed, so nothing ran and nothing
            # was billed - recorded anyway so a run that burned four attempts says so
            self._record(model, kind, None, label, attempt_number, outcome="rejected")
            raise _FalRejected("submit", response.status_code, response.text)
        if response.status_code >= 400:
            raise RenderError(f"{PREFIX} fal HTTP {response.status_code} for {model}: {response.text[:300]}")

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
                if result.status_code in (400, 422):
                    # the job ran before it was refused, so a runner spent GPU time on it
                    self._record(
                        model, kind, None, label, attempt_number, outcome="rejected", possibly_billed=True
                    )
                    raise _FalRejected("result", result.status_code, result.text)
                if result.status_code >= 400:
                    raise RenderError(f"{PREFIX} fal result HTTP {result.status_code}: {result.text[:200]}")
                # the billed quantity rides on this response and on no other: the status poll
                # carries no x-fal-* header, and its metrics.inference_time is runner wall clock
                self._record(
                    model, kind, result.headers.get("x-fal-billable-units"), label, attempt_number
                )
                return result.json()
            if state in {"FAILED", "ERROR", "CANCELLED"}:
                self._record(model, kind, None, label, attempt_number, outcome="failed")
                raise RenderError(f"{PREFIX} fal reported {state} for {model}: {status.text[:300]}")
            time.sleep(self.poll_seconds)
        raise RenderError(f"{PREFIX} fal did not finish {model} within {self.timeout}s.")

    def _record(
        self,
        model: str,
        kind: str,
        units,
        label: str,
        attempt: int,
        outcome: str = "ok",
        possibly_billed: bool = False,
    ) -> None:
        """Best-effort: an accounting failure must never cost a render that was paid for."""
        if self.ledger is None:
            return
        try:
            self.ledger.record_fal(
                model, kind, units, label, attempt, outcome=outcome, possibly_billed=possibly_billed
            )
        except Exception as exc:  # pragma: no cover - defensive
            warn(f"could not record the cost of a fal render: {exc}")

    # ---------------------------------------------------------------- media

    def image(self, model: str, request: ImageRequest) -> bytes:
        payload, optional = build_fal_image_payload(model, request, fal_schema(model))
        spare = {"references": list(request.references)}
        result = self.run(model, payload, optional_keys=optional, spare=spare, label=request.label, kind="image")
        return _first_media(result, ("images", "image"), self._headers())

    def video(self, model: str, request: VideoRequest) -> bytes:
        payload, optional = build_fal_video_payload(model, request, fal_schema(model))
        spare = {"first_frame": request.first_frame, "references": list(request.references)}
        result = self.run(model, payload, optional_keys=optional, spare=spare, label=request.label, kind="video")
        return _first_media(result, ("video", "videos"), self._headers())


def build_fal_image_payload(
    model: str, request: ImageRequest, schema: dict | None = None
) -> tuple[dict, tuple[str, ...]]:
    """Payload for one fal image endpoint, using only fields it declares."""
    properties = (schema or {}).get("properties") or {}
    known = bool(properties)
    payload: dict[str, Any] = {"prompt": request.prompt}
    optional: list[str] = []

    def offer(name: str, value: Any, droppable: bool = True) -> None:
        if value is None:
            return
        if known and name not in properties:
            return
        payload[name] = value
        if droppable:
            optional.append(name)

    width, height = aspect_to_size(request.aspect_ratio)
    if not known or "image_size" in properties:
        if known and not _takes_object(properties.get("image_size")):
            offer("image_size", _fit_enum(_size_preset(width, height), _enum_of(properties["image_size"])))
        else:
            offer("image_size", {"width": width, "height": height})
    if "aspect_ratio" in properties:
        offer("aspect_ratio", _fit_enum(request.aspect_ratio, _enum_of(properties["aspect_ratio"])))
    if "width" in properties and "height" in properties:
        offer("width", width)
        offer("height", height)

    offer("num_images", 1)
    offer("output_format", _fit_enum("png", _enum_of(properties.get("output_format"))) or "png")
    offer("enable_safety_checker", False)
    if request.negative:
        offer("negative_prompt", request.negative)
    if request.seed is not None:
        offer("seed", _fit_seed(request.seed, properties.get("seed")))
    if request.references:
        field = _first_field(properties, REFERENCE_FIELDS) if known else "image_urls"
        if field:
            offer(field, list(request.references))
        elif (single := _first_field(properties, FIRST_FRAME_FIELDS)):
            offer(single, request.references[0])

    # least essential first: whatever the endpoint still refuses gets dropped in that order
    order = ("negative_prompt", "enable_safety_checker", "output_format", "num_images")
    optional.sort(key=lambda name: order.index(name) if name in order else len(order))
    return payload, tuple(optional)


def build_fal_video_payload(
    model: str, request: VideoRequest, schema: dict | None = None
) -> tuple[dict, tuple[str, ...]]:
    """Payload for one fal video endpoint, using only fields it declares."""
    properties = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or ())
    known = bool(properties)
    payload: dict[str, Any] = {"prompt": request.prompt}
    optional: list[str] = []

    def offer(name: str, value: Any, droppable: bool = True) -> None:
        if value is None or (known and name not in properties):
            return
        payload[name] = value
        if droppable:
            optional.append(name)

    wants_references = any(hint in model for hint in REFERENCE_HINTS)
    reference_field = _first_field(properties, REFERENCE_FIELDS) if known else "reference_image_urls"
    frame_field = _first_field(properties, FIRST_FRAME_FIELDS) if known else "image_url"

    if wants_references and request.references and reference_field:
        payload[reference_field] = list(request.references)
    elif request.first_frame and frame_field:
        payload[frame_field] = request.first_frame
    elif request.references and reference_field:
        payload[reference_field] = list(request.references)
    elif request.first_frame and reference_field:
        # a reference-only endpoint still gets the frame we hold, as a reference
        payload[reference_field] = [request.first_frame]
    elif request.references and frame_field:
        payload[frame_field] = request.references[0]

    # an endpoint that requires an image cannot be run from text alone
    missing = [name for name in required if name in FIRST_FRAME_FIELDS + REFERENCE_FIELDS and name not in payload]
    if missing:
        raise RenderError(
            f"{PREFIX} '{model}' requires {missing[0]}: it cannot start from text. "
            "Turn on image_provider so every shot gets a first frame, render the subject "
            "sheets for a reference-to-video endpoint, or pick a text-to-video model."
        )

    if "duration" in properties or not known:
        low, high = _bounds_of(properties.get("duration"))
        seconds = max(1, int(round(request.seconds)))
        if low is not None:
            seconds = max(int(low), seconds)
        if high is not None:
            seconds = min(int(high), seconds)
        allowed = _enum_of(properties.get("duration"))
        if allowed:
            numeric = [value for value in allowed if str(value).strip().lstrip("-").isdigit()]
            if numeric:
                closest = min(numeric, key=lambda value: abs(int(str(value)) - seconds))
                offer("duration", closest)
        else:
            offer("duration", seconds)

    expansion = (request.expansion or "").strip().lower()
    if expansion in {"minimal", "rich"}:
        # Both H3 endpoints rewrite the prompt before generating (prompt_expansion_mode
        # defaults to "balanced"), and the wan endpoints do the same behind a boolean. A
        # rewrite is per-request, so it re-invents the look of every shot independently -
        # which is the last thing a film assembled from one art direction needs.
        if "prompt_expansion_mode" in properties:
            offer("prompt_expansion_mode", "fast" if expansion == "minimal" else "quality")
        if "enable_prompt_expansion" in properties:
            offer("enable_prompt_expansion", expansion == "rich")

    if request.audio:
        audio_field = _first_field(properties, AUDIO_LIST_FIELDS + AUDIO_URL_FIELDS) if known else ""
        if not known:
            audio_field = "audio_url"
        prop = properties.get(audio_field) if known else None
        if audio_field and not any(branch.get("type") == "boolean" for branch in _branches(prop)):
            wanted_list = audio_field in AUDIO_LIST_FIELDS
            offer(audio_field, [request.audio] if wanted_list else request.audio)

    if ("aspect_ratio" in properties or not known) and not payload.get(frame_field):
        offer("aspect_ratio", _fit_enum(request.aspect_ratio, _enum_of(properties.get("aspect_ratio"))))
    if request.seed is not None:
        offer("seed", _fit_seed(request.seed, properties.get("seed")))

    order = ("aspect_ratio", "seed", "duration")
    optional.sort(key=lambda name: order.index(name) if name in order else len(order))
    # the audio is the point of the request when it is there; drop it last
    optional = [name for name in optional if name not in AUDIO_LIST_FIELDS + AUDIO_URL_FIELDS]
    return payload, tuple(optional)


def _fit_seed(seed: int, prop: Any) -> int:
    """Clamp a seed into whatever range the endpoint declares (default 32-bit)."""
    value = clamp_seed(seed) or 0
    low, high = _bounds_of(prop)
    if high is not None:
        value = int(value) % (int(high) + 1)
    if low is not None:
        value = max(int(low), value)
    return int(value)


def _size_preset(width: int, height: int) -> str:
    """Nearest of fal's named image sizes, for endpoints that only take those."""
    ratio = width / max(1, height)
    if ratio > 1.6:
        return "landscape_16_9"
    if ratio > 1.1:
        return "landscape_4_3"
    if ratio > 0.9:
        return "square_hd"
    if ratio > 0.6:
        return "portrait_4_3"
    return "portrait_16_9"


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

    def __init__(
        self,
        api_key: str = "",
        timeout: int = 600,
        poll_seconds: float = 3.0,
        verbose: bool = False,
        ledger=None,
    ) -> None:
        self.api_key = resolve_key("openrouter", api_key)
        self.timeout = max(30, int(timeout))
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.verbose = bool(verbose)
        #: OpenRouter states the charge on the reply itself, so nothing has to be priced.
        self.ledger = ledger
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

    def _record(self, model: str, kind: str, body: dict, label: str, outcome: str = "ok") -> None:
        """Best-effort: an accounting failure must never cost a render that was paid for."""
        if self.ledger is None:
            return
        try:
            self.ledger.record_openrouter_media(
                model, kind, (body or {}).get("usage") or {}, label, outcome=outcome
            )
        except Exception as exc:  # pragma: no cover - defensive
            warn(f"could not record the cost of an OpenRouter render: {exc}")

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
            payload["seed"] = clamp_seed(request.seed)
        if request.references:
            payload["input_references"] = self._references(request.references)
        response = requests.post(
            f"{OPENROUTER_URL}/images", headers=self._headers(), data=json.dumps(payload), timeout=self.timeout
        )
        if response.status_code >= 400:
            raise RenderError(f"{PREFIX} OpenRouter image HTTP {response.status_code}: {response.text[:300]}")
        body = response.json() or {}
        # the image reply carries no id of any kind, so this is the only chance to see the
        # charge - it can never be looked up afterwards
        self._record(model, "image", body, request.label)
        data = body.get("data") or []
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
            payload["seed"] = clamp_seed(request.seed)
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
                # the usage block only exists once the job is done; the 202 submit has none
                self._record(model, "video", body, request.label)
                urls = body.get("unsigned_urls") or body.get("urls") or []
                if not urls:
                    raise RenderError(f"{PREFIX} OpenRouter finished without a video url: {str(body)[:200]}")
                return _download(urls[0], self._headers())
            if state in {"failed", "cancelled", "canceled", "error"}:
                self._record(model, "video", body, request.label, outcome="failed")
                raise RenderError(f"{PREFIX} OpenRouter video {state}: {str(body)[:300]}")
            time.sleep(self.poll_seconds)
        raise RenderError(f"{PREFIX} OpenRouter did not finish the video within {self.timeout}s.")


# --------------------------------------------------------------------------- front door


def make_media_client(
    provider: str, api_key: str = "", timeout: int = 600, verbose: bool = False, ledger=None
):
    provider = (provider or "none").strip().lower()
    if provider in {"", "none", "off"}:
        return None
    if provider == "fal":
        # fetched once here, on the calling thread, so no render worker ever waits on it
        cost_module.fal_prices()
        return FalClient(api_key, timeout=timeout, verbose=verbose, ledger=ledger)
    if provider == "openrouter":
        return OpenRouterMediaClient(api_key, timeout=timeout, verbose=verbose, ledger=ledger)
    raise RenderError(f"{PREFIX} unknown media provider '{provider}'. Pick one of {MEDIA_PROVIDERS}.")


def _announced(call: Callable[[], bytes], index: int, on_done: Callable[[int, bytes], None] | None):
    """Run one render and hand the result over as soon as it lands, not at the end.

    The callback is what feeds the node's live gallery, so a failure in it must cost
    the preview and nothing else - the bytes were already paid for.
    """

    def job() -> bytes:
        payload = call()
        if on_done and payload:
            try:
                on_done(index, payload)
            except Exception as exc:  # pragma: no cover - defensive
                warn(f"preview for item {index + 1} failed: {exc}")
        return payload

    return job


def render_images(
    client,
    model: str,
    requests_: list[ImageRequest],
    concurrency: int = 2,
    errors: list[Exception] | None = None,
    on_done: Callable[[int, bytes], None] | None = None,
) -> list[bytes | None]:
    if client is None or not requests_:
        return []
    log(f"rendering {len(requests_)} image(s) with {client.name}/{model} (concurrency {concurrency})")
    jobs = [
        _announced(lambda request=request: client.image(model, request), index, on_done)
        for index, request in enumerate(requests_)
    ]
    return _in_parallel(jobs, concurrency, "image", errors)


def render_videos(
    client,
    model: str,
    requests_: list[VideoRequest],
    concurrency: int = 2,
    errors: list[Exception] | None = None,
    on_done: Callable[[int, bytes], None] | None = None,
) -> list[bytes | None]:
    if client is None or not requests_:
        return []
    log(f"rendering {len(requests_)} video(s) with {client.name}/{model} (concurrency {concurrency})")
    jobs = [
        _announced(lambda request=request: client.video(model, request), index, on_done)
        for index, request in enumerate(requests_)
    ]
    return _in_parallel(jobs, concurrency, "video", errors)


def placeholder_image(aspect_ratio: str = "16:9", base: int = 512):
    """Black frame standing in for a shot whose render failed, so the IMAGE output
    stays aligned with the shot list instead of silently losing an entry."""
    import torch

    width, height = aspect_to_size(aspect_ratio, base=base)
    return torch.zeros(1, height, width, 3, dtype=torch.float32)


# what the provider actually handed back, so a JPEG is not saved as a .png
IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF8", "gif"),
)


def image_extension(payload: bytes) -> str:
    for signature, extension in IMAGE_SIGNATURES:
        if payload.startswith(signature):
            return extension
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "webp"
    return "png"


def run_stamp() -> str:
    """One timestamp per run, so every file it writes sorts together."""
    return time.strftime("%Y%m%d-%H%M%S")


def save_media(
    payloads: list[bytes | None],
    prefix: str,
    kind: str,
    extension: str = "",
    temporary: bool = False,
    reuse: dict[int, str] | None = None,
    stamp: str = "",
) -> list[str]:
    """Write finished renders to disk; returns their paths, skipping the ones that failed.

    ``temporary`` puts them in ComfyUI's temp folder instead of its output folder, and
    ``reuse`` names files the live preview already wrote there, so nothing is written to
    the temp folder twice. A shared ``stamp`` keeps one run's files next to each other.
    """
    directory = output_directory(temporary=temporary)
    stamp = stamp or run_stamp()
    paths: list[str] = []
    for index, payload in enumerate(payloads):
        if not payload:
            continue
        written = (reuse or {}).get(index) if temporary else None
        if written and os.path.exists(written):
            paths.append(written)
            continue
        suffix = extension or image_extension(payload)
        path = os.path.join(directory, f"{prefix}_{stamp}_{kind}{index + 1:03d}.{suffix}")
        try:
            with open(path, "wb") as handle:
                handle.write(payload)
            paths.append(path)
        except OSError as exc:
            warn(f"could not write {path}: {exc}")
    return paths


def save_videos(
    payloads: list[bytes | None],
    prefix: str = "music2prompts",
    temporary: bool = False,
    reuse: dict[int, str] | None = None,
    stamp: str = "",
) -> list[str]:
    """Write finished clips to disk. They always land somewhere - assembling the final
    film needs real files - and ``temporary`` only chooses which folder."""
    return save_media(payloads, prefix, "shot", "mp4", temporary, reuse, stamp)


def save_images(
    payloads: list[bytes | None],
    prefix: str = "music2prompts",
    kind: str = "image",
    stamp: str = "",
) -> list[str]:
    """Write rendered frames into ComfyUI's output folder, in the format they arrived in."""
    return save_media(payloads, prefix, kind, "", temporary=False, stamp=stamp)


# --------------------------------------------------------------------------- model probing


def _fal_index(categories: tuple[str, ...], pages: int = 2) -> list[str]:
    """Model ids from fal's public index, most popular first.

    The index ignores ``page_size`` and always answers with 40 items, so more than one
    page has to be asked for - a single call would have hidden every edit endpoint past
    the fortieth, which is most of them.
    """
    import requests

    ids: list[str] = []
    for category in categories:
        for page in range(1, max(1, pages) + 1):
            response = requests.get(
                FAL_MODEL_INDEX, params={"categories": category, "page": page}, timeout=4.0
            )
            if response.status_code >= 400:
                break
            data = response.json() or {}
            for item in data.get("items", []):
                model_id = item.get("id")
                if model_id and model_id not in ids:
                    ids.append(model_id)
            if page >= int(data.get("pages") or 1):
                break
    return ids


def probe_fal_images_raw() -> list[str]:
    # image-to-image is where every edit endpoint lives, and an edit endpoint is the only
    # kind that can be told which face to keep
    return _fal_index(("text-to-image", "image-to-image"))


def probe_fal_videos_raw() -> list[str]:
    # audio-to-video is the lip-sync / talking-head category
    return _fal_index(("image-to-video", "text-to-video", "audio-to-video"))


def _openrouter_media_models(kind: str) -> list[str]:
    import requests

    response = requests.get(f"{OPENROUTER_URL}/{kind}/models", timeout=3.0)
    if response.status_code >= 400:
        raise ProviderError(f"HTTP {response.status_code}")
    return [model.get("id") for model in (response.json() or {}).get("data", []) if model.get("id")]


def probe_openrouter_images_raw() -> list[str]:
    return _openrouter_media_models("images")


def probe_openrouter_videos_raw() -> list[str]:
    return _openrouter_media_models("videos")

