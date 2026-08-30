"""Giving the card back: what the node loads, it has to let go of.

A local LLM or a Whisper pipeline left resident is the usual reason the sampler that
runs next in the graph cannot find room, and the error it raises names neither.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import asr as asr_module  # noqa: E402
from music2prompts.lmstudio import LMStudioClient  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = {} if body is None else body
        self.content = b"x"
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def fake_requests(monkeypatch, handler):
    """Stand in for the lazily imported ``requests`` module, recording every call."""
    module = types.ModuleType("requests")
    module.request = handler
    module.exceptions = types.SimpleNamespace(
        Timeout=TimeoutError, RequestException=OSError, ConnectionError=ConnectionError
    )
    monkeypatch.setitem(sys.modules, "requests", module)
    return module


def node():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.node import Music2PromptsLM

    return Music2PromptsLM


# --------------------------------------------------------------------------- loading


def test_the_context_length_goes_where_lm_studio_looks_for_it(monkeypatch):
    """It is a top-level field. Nested under 'config' it is rejected with HTTP 400, the
    retry then loads the model at whatever context LM Studio defaults to, and prompts
    are silently truncated for the rest of the run."""
    seen = {}

    def handler(method, url, **kwargs):
        seen["path"] = url
        seen["body"] = json.loads(kwargs["data"])
        return FakeResponse(200, {"load_config": {"context_length": 32768}})

    fake_requests(monkeypatch, handler)
    LMStudioClient("http://lm").load("qwen/qwen3.8-27b", context_length=32768)

    assert seen["body"]["context_length"] == 32768, "top level, not under 'config'"
    assert "config" not in seen["body"]
    assert seen["body"]["echo_load_config"] is True, "so the log can report what was applied"


def test_a_smaller_context_than_asked_for_is_said_out_loud(monkeypatch, caplog):
    import logging

    fake_requests(
        monkeypatch,
        lambda *a, **k: FakeResponse(200, {"load_config": {"context_length": 4096}}),
    )
    with caplog.at_level(logging.WARNING):
        LMStudioClient("http://lm").load("m", context_length=32768)
    assert "4096" in caplog.text and "truncated" in caplog.text


def test_a_refused_load_still_falls_back_instead_of_failing_the_run(monkeypatch):
    calls = []

    def handler(method, url, **kwargs):
        calls.append(json.loads(kwargs["data"]))
        return FakeResponse(400, {"error": "nope"}) if len(calls) == 1 else FakeResponse(200, {})

    fake_requests(monkeypatch, handler)
    LMStudioClient("http://lm").load("m", context_length=8192)

    assert len(calls) == 2
    assert calls[1] == {"model": "m"}, "the retry asks for nothing but the model"


# --------------------------------------------------------------------------- unloading


def test_the_unload_is_waited_out_so_the_vram_is_really_back(monkeypatch):
    """The call returns as soon as it is accepted; the next node needs the memory now."""
    states = [
        {"data": [{"id": "m", "state": "loaded"}]},
        {"data": [{"id": "m", "state": "loaded"}]},
        {"data": [{"id": "m", "state": "not-loaded"}]},
    ]
    monkeypatch.setattr("time.sleep", lambda _: None)
    fake_requests(monkeypatch, lambda *a, **k: FakeResponse(200, states.pop(0) if states else {"data": []}))

    assert LMStudioClient("http://lm").wait_unloaded("m", timeout=5.0) is True
    assert not states, "it kept asking until the model was gone"


def test_a_model_that_will_not_go_is_reported_rather_than_hanging(monkeypatch, caplog):
    import logging

    monkeypatch.setattr("time.sleep", lambda _: None)
    fake_requests(
        monkeypatch,
        lambda *a, **k: FakeResponse(200, {"data": [{"id": "m", "state": "loaded"}]}),
    )
    with caplog.at_level(logging.WARNING):
        assert LMStudioClient("http://lm").wait_unloaded("m", timeout=0.0) is False
    assert "still reports" in caplog.text


def test_an_unreachable_server_does_not_hold_up_a_finished_run(monkeypatch):
    def handler(*a, **k):
        raise ConnectionError("down")

    fake_requests(monkeypatch, handler)
    assert LMStudioClient("http://lm").wait_unloaded("m", timeout=5.0) is True


# --------------------------------------------------------------------------- the cache


def test_releasing_the_cache_prefers_comfyui_s_own_call(monkeypatch):
    """ComfyUI tracks what it thinks is free, so tell it rather than going behind it."""
    called = []
    module = types.ModuleType("comfy.model_management")
    module.soft_empty_cache = lambda: called.append("comfy")
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
    monkeypatch.setitem(sys.modules, "comfy.model_management", module)

    asr_module.release_cache()
    assert called == ["comfy"]


def test_releasing_the_cache_outside_comfyui_is_harmless(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    asr_module.release_cache()  # must not raise


# --------------------------------------------------------------------------- defaults


def test_the_defaults_hand_the_card_back_rather_than_holding_it():
    """Both of these were the other way round, and cost a whole run to an OOM."""
    schema = node().define_schema()
    defaults = {item.id: getattr(item, "default", None) for item in schema.inputs}
    assert defaults["lm_unload_after"] is True
    assert defaults["whisper_keep_loaded"] is False
    assert defaults["free_lmstudio_vram"] is True
    assert defaults["free_comfy_vram"] is True
