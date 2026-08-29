"""One process-wide cache of the model lists, shared by the schema and the HTTP route.

``define_schema`` is re-run by ComfyUI on every ``/object_info`` request and twice per
queued prompt, so it must never touch the network - it reads :func:`snapshot`, which
returns instantly. The refreshing happens elsewhere: the pack's own HTTP route calls
:func:`resolve` (in a worker thread), and a background warm-up fills the cache shortly
after ComfyUI starts.

Two rules keep saved workflows working:

* the cache is **monotonic** - a probe that fails, or a provider that went offline, can
  never remove a value that was offered before. ComfyUI validates the chosen combo value
  against the schema's options, so a shrinking list would reject a prompt the user never
  touched;
* a resolved list is kept for :data:`TTL_OK`, an unresolved one only for
  :data:`TTL_EMPTY`, so starting LM Studio *after* ComfyUI shows up quickly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .util import log, warn

TTL_OK = 300.0
TTL_EMPTY = 20.0

KINDS = (
    "lmstudio",
    "openrouter_llm",
    "openai_llm",
    "anthropic_llm",
    "fal_image",
    "fal_video",
    "openrouter_image",
    "openrouter_video",
)


@dataclass
class _Entry:
    values: list[str] = field(default_factory=list)
    fetched_at: float = 0.0
    resolved: bool = False

    def fresh(self) -> bool:
        ttl = TTL_OK if self.resolved else TTL_EMPTY
        return bool(self.values) and (time.time() - self.fetched_at) < ttl


_LOCK = threading.Lock()
_ENTRIES: dict[str, _Entry] = {}
_FLIGHT: dict[str, threading.Event] = {}


def _sources() -> dict[str, tuple[Callable[[], list[str]], list[str]]]:
    """Probe + static fallback per kind. Imported lazily to keep startup cheap."""
    from . import render
    from . import providers as prov

    return {
        "lmstudio": (prov.probe_lmstudio_raw, list(prov.LMSTUDIO_FALLBACK)),
        "openrouter_llm": (prov.probe_openrouter_llms_raw, list(prov.FALLBACK_OPENROUTER)),
        "openai_llm": (prov.probe_openai_models_raw, list(prov.FALLBACK_OPENAI)),
        "anthropic_llm": (prov.probe_anthropic_models_raw, list(prov.FALLBACK_ANTHROPIC)),
        "fal_image": (render.probe_fal_images_raw, list(render.FALLBACK_FAL_IMAGE)),
        "fal_video": (render.probe_fal_videos_raw, list(render.FALLBACK_FAL_VIDEO)),
        "openrouter_image": (render.probe_openrouter_images_raw, list(render.FALLBACK_OPENROUTER_IMAGE)),
        "openrouter_video": (render.probe_openrouter_videos_raw, list(render.FALLBACK_OPENROUTER_VIDEO)),
    }


def _fallback(kind: str) -> list[str]:
    try:
        return list(_sources()[kind][1])
    except Exception:
        return []


def _merge(fresh: list[str], previous: list[str], fallback: list[str]) -> list[str]:
    """Newest first, then everything ever offered - the list may never shrink."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in (fresh, previous, fallback):
        for value in group:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


def snapshot(kind: str) -> list[str]:
    """What the dropdown should show right now. Never blocks, never fetches."""
    with _LOCK:
        entry = _ENTRIES.get(kind)
        if entry and entry.values:
            return list(entry.values)
    return _fallback(kind)


def snapshot_all() -> dict[str, list[str]]:
    return {kind: snapshot(kind) for kind in KINDS}


def resolve(kind: str, force: bool = False) -> list[str]:
    """Return the list, probing when it is stale. One probe per kind at a time."""
    if kind not in KINDS:
        raise KeyError(kind)

    with _LOCK:
        entry = _ENTRIES.get(kind)
        if entry and entry.fresh() and not force:
            return list(entry.values)
        flight = _FLIGHT.get(kind)
        if flight is not None:  # somebody else is already probing this kind
            waiter = flight
        else:
            waiter = None
            _FLIGHT[kind] = threading.Event()

    if waiter is not None:
        waiter.wait(timeout=30.0)
        return snapshot(kind)

    probe, fallback = _sources()[kind]
    found: list[str] = []
    try:
        found = [str(value) for value in probe() if value]
    except Exception as exc:
        warn(f"model probe '{kind}' failed: {exc}")

    with _LOCK:
        previous = list(_ENTRIES[kind].values) if kind in _ENTRIES else []
        merged = _merge(found, previous, fallback)
        _ENTRIES[kind] = _Entry(values=merged, fetched_at=time.time(), resolved=bool(found))
        event = _FLIGHT.pop(kind, None)
    if event is not None:
        event.set()
    return merged


def warm(kinds: tuple[str, ...] = KINDS, background: bool = True) -> None:
    """Fill the cache without blocking ComfyUI's startup."""

    def run() -> None:
        started = time.time()
        threads = [threading.Thread(target=resolve, args=(kind,), daemon=True) for kind in kinds]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20.0)
        counts = ", ".join(f"{kind}:{len(snapshot(kind))}" for kind in kinds)
        log(f"model lists ready in {time.time() - started:.1f}s ({counts})")

    if background:
        threading.Thread(target=run, name="music2prompts-warm", daemon=True).start()
    else:
        run()


def reset() -> None:
    """Test helper: forget everything."""
    with _LOCK:
        _ENTRIES.clear()
        _FLIGHT.clear()
