"""The shared model-list cache: monotonic, TTL'd, one probe at a time."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import model_cache  # noqa: E402

KIND = "fal_image"
FALLBACK = ["fallback-model"]


@pytest.fixture(autouse=True)
def clean():
    model_cache.reset()
    yield
    model_cache.reset()


def stub(monkeypatch, probe, fallback=FALLBACK):
    monkeypatch.setattr(model_cache, "_sources", lambda: {KIND: (probe, list(fallback))})


def test_snapshot_is_the_fallback_while_cold(monkeypatch):
    stub(monkeypatch, lambda: ["never-called"])
    assert model_cache.snapshot(KIND) == FALLBACK


def test_resolve_probes_and_appends_the_fallback(monkeypatch):
    stub(monkeypatch, lambda: ["a", "b"])
    assert model_cache.resolve(KIND) == ["a", "b", "fallback-model"]
    assert model_cache.snapshot(KIND) == ["a", "b", "fallback-model"]


def test_the_list_never_shrinks(monkeypatch):
    """A model that was offered once stays selectable, or saved workflows break."""
    values = [["a", "b"], ["b"], []]
    stub(monkeypatch, lambda: values.pop(0))
    model_cache.resolve(KIND)
    assert model_cache.resolve(KIND, force=True) == ["b", "a", "fallback-model"]
    assert "a" in model_cache.resolve(KIND, force=True)


def test_a_failed_probe_keeps_what_was_there(monkeypatch):
    def boom():
        raise RuntimeError("provider down")

    stub(monkeypatch, lambda: ["a"])
    model_cache.resolve(KIND)
    stub(monkeypatch, boom)
    assert model_cache.resolve(KIND, force=True) == ["a", "fallback-model"]


def test_a_resolved_list_is_reused_inside_the_ttl(monkeypatch):
    calls = []
    stub(monkeypatch, lambda: calls.append(1) or ["a"])
    model_cache.resolve(KIND)
    model_cache.resolve(KIND)
    assert len(calls) == 1
    assert model_cache.resolve(KIND, force=True) is not None
    assert len(calls) == 2


def test_an_unresolved_list_is_retried_sooner(monkeypatch):
    """LM Studio started after ComfyUI must appear without a restart."""
    monkeypatch.setattr(model_cache, "TTL_EMPTY", 0.05)
    monkeypatch.setattr(model_cache, "TTL_OK", 60.0)
    calls = []
    stub(monkeypatch, lambda: calls.append(1) and [] or [])
    model_cache.resolve(KIND)
    time.sleep(0.06)
    model_cache.resolve(KIND)
    assert len(calls) == 2, "an empty probe must be retried after the short TTL"

    stub(monkeypatch, lambda: ["now-running"])
    time.sleep(0.06)  # the empty result is still inside its own short TTL until now
    assert model_cache.resolve(KIND)[0] == "now-running"


def test_concurrent_callers_share_one_probe(monkeypatch):
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.3)
        return ["a"]

    stub(monkeypatch, slow)
    threads = [threading.Thread(target=model_cache.resolve, args=(KIND,)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1, "six widgets must not each pay for their own probe"


def test_unknown_kind_is_rejected():
    with pytest.raises(KeyError):
        model_cache.resolve("not-a-kind")


def test_every_kind_has_a_source_and_a_fallback():
    sources = model_cache._sources()
    assert set(sources) == set(model_cache.KINDS)
    for kind, (probe, fallback) in sources.items():
        assert callable(probe), kind
        assert fallback, f"{kind} needs a static fallback"
