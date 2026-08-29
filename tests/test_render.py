"""Rendering layer: payload shapes, concurrency and result parsing. No network."""

from __future__ import annotations

import base64
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.render import (  # noqa: E402
    FalClient,
    ImageRequest,
    OpenRouterMediaClient,
    RenderError,
    VideoRequest,
    _first_media,
    _in_parallel,
    aspect_to_size,
    data_uri,
    make_media_client,
    render_images,
)

PIXEL = data_uri(b"\x89PNG\r\n")


@pytest.fixture
def fal(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test-key")
    return FalClient(timeout=60)


@pytest.fixture
def openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return OpenRouterMediaClient(timeout=60)


def record(client, result):
    """Capture what ``run`` would submit instead of talking to fal."""
    seen: dict = {}

    def fake_run(model, payload, optional_keys=()):
        seen.update(model=model, payload=payload, optional=optional_keys)
        return result

    client.run = fake_run  # type: ignore[assignment]
    return seen


class FakeResponse:
    def __init__(self, status_code: int, body: dict | bytes) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)[:200]
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()

    def json(self):
        return self._body


def fake_requests(monkeypatch, post=None, get=None):
    """Inject a stand-in ``requests`` module (the real one is imported lazily)."""
    module = types.ModuleType("requests")
    module.post = post or (lambda *a, **k: FakeResponse(200, {}))
    module.get = get or (lambda *a, **k: FakeResponse(200, {}))
    module.exceptions = types.SimpleNamespace(
        Timeout=TimeoutError, RequestException=OSError, ConnectionError=ConnectionError
    )
    monkeypatch.setitem(sys.modules, "requests", module)
    return module


# --------------------------------------------------------------------------- helpers


def test_aspect_to_size_matches_the_ratio():
    width, height = aspect_to_size("16:9")
    assert width > height
    assert abs(width / height - 16 / 9) < 0.06
    assert width % 32 == 0 and height % 32 == 0
    assert aspect_to_size("1:1") == (1024, 1024)
    tall = aspect_to_size("9:16")
    assert tall[1] > tall[0]
    assert aspect_to_size("nonsense")[0] > aspect_to_size("nonsense")[1]  # falls back to 16:9


def test_parallel_keeps_order_and_survives_failures():
    def boom():
        raise RuntimeError("nope")

    results = _in_parallel([lambda: 1, boom, lambda: 3], concurrency=3, label="test")
    assert results == [1, None, 3]


def test_parallel_runs_single_threaded_when_asked():
    order: list[int] = []
    jobs = [lambda index=index: order.append(index) for index in range(4)]
    _in_parallel(jobs, concurrency=1, label="test")
    assert order == [0, 1, 2, 3]


def test_first_media_reads_images_videos_and_data_uris(monkeypatch):
    fake_requests(monkeypatch, get=lambda url, **kwargs: FakeResponse(200, b"binary"))
    assert _first_media({"images": [{"url": "https://x/a.png"}]}, ("images",), {}) == b"binary"
    assert _first_media({"video": {"url": "https://x/a.mp4"}}, ("video", "videos"), {}) == b"binary"
    inline = {"images": [{"url": data_uri(b"abc")}]}
    assert _first_media(inline, ("images",), {}) == b"abc"
    with pytest.raises(RenderError, match="none of"):
        _first_media({"nothing": 1}, ("images",), {})


def test_none_provider_renders_nothing():
    assert make_media_client("none") is None
    assert render_images(None, "whatever", [ImageRequest(prompt="x")], 4) == []


def test_unknown_provider_is_rejected():
    with pytest.raises(RenderError, match="unknown media provider"):
        make_media_client("midjourney-by-carrier-pigeon")


def test_missing_fal_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_API_KEY", raising=False)
    with pytest.raises(RenderError, match="no fal.ai key"):
        make_media_client("fal")


# --------------------------------------------------------------------------- fal payloads


def test_fal_image_payload(fal):
    seen = record(fal, {"images": [{"url": data_uri(b"png")}]})
    payload = ImageRequest(prompt="a night street", negative="daylight", aspect_ratio="9:16", seed=11)
    assert fal.image("fal-ai/flux/dev", payload) == b"png"
    sent = seen["payload"]
    assert sent["prompt"] == "a night street"
    assert sent["negative_prompt"] == "daylight"
    assert sent["seed"] == 11
    assert sent["image_size"]["height"] > sent["image_size"]["width"]
    assert "image_size" in seen["optional"], "size must be droppable for models that reject it"


def test_fal_video_uses_the_first_frame_for_image_to_video(fal):
    seen = record(fal, {"video": {"url": data_uri(b"mp4", "video/mp4")}})
    request = VideoRequest(prompt="[Shot 1] ...", seconds=6.4, first_frame=PIXEL, seed=3)
    assert fal.video("minimax/h3/image-to-video", request) == b"mp4"
    sent = seen["payload"]
    assert sent["image_url"] == PIXEL
    assert sent["duration"] == 6, "duration is rounded to whole seconds"
    assert "reference_image_urls" not in sent


def test_fal_video_uses_references_for_reference_endpoints(fal):
    seen = record(fal, {"video": {"url": data_uri(b"mp4", "video/mp4")}})
    request = VideoRequest(
        prompt="subject_definitions: ...", seconds=8.0, first_frame=PIXEL, references=[PIXEL, PIXEL]
    )
    fal.video("minimax/h3/reference-to-video", request)
    sent = seen["payload"]
    assert sent["reference_image_urls"] == [PIXEL, PIXEL]
    assert "image_url" not in sent
    assert sent["aspect_ratio"] == "16:9"


def test_fal_drops_rejected_fields_and_retries(fal, monkeypatch):
    calls: list[dict] = []

    def post(url, headers=None, data=None, timeout=None):
        payload = json.loads(data)
        calls.append(payload)
        if "negative_prompt" in payload:
            return FakeResponse(422, {"detail": "extra fields not permitted"})
        return FakeResponse(
            200,
            {
                "status_url": "https://queue/status",
                "response_url": "https://queue/response",
            },
        )

    def get(url, headers=None, timeout=None):
        if url.endswith("status"):
            return FakeResponse(200, {"status": "COMPLETED"})
        return FakeResponse(200, {"images": [{"url": data_uri(b"png")}]})

    fake_requests(monkeypatch, post=post, get=get)
    fal.poll_seconds = 0.5
    assert fal.image("fal-ai/flux/dev", ImageRequest(prompt="x", negative="blurry")) == b"png"
    assert len(calls) == 2 and "negative_prompt" not in calls[1]


def test_fal_reports_a_failed_job(fal, monkeypatch):
    fake_requests(
        monkeypatch,
        post=lambda *a, **k: FakeResponse(
            200, {"status_url": "https://queue/status", "response_url": "https://queue/response"}
        ),
        get=lambda *a, **k: FakeResponse(200, {"status": "FAILED"}),
    )
    with pytest.raises(RenderError, match="FAILED"):
        fal.run("fal-ai/flux/dev", {"prompt": "x"})


# --------------------------------------------------------------------------- OpenRouter payloads


def test_openrouter_image_returns_base64(monkeypatch, openrouter):
    seen: dict = {}

    def post(url, headers=None, data=None, timeout=None):
        seen.update(url=url, payload=json.loads(data))
        return FakeResponse(200, {"data": [{"b64_json": base64.b64encode(b"png").decode()}]})

    fake_requests(monkeypatch, post=post)
    request = ImageRequest(prompt="a face", aspect_ratio="4:3", seed=5, references=[PIXEL])
    assert openrouter.image("google/gemini-3.1-flash-image", request) == b"png"
    assert seen["url"].endswith("/images")
    assert seen["payload"]["aspect_ratio"] == "4:3"
    assert seen["payload"]["input_references"] == [{"type": "image_url", "image_url": {"url": PIXEL}}]


def test_openrouter_video_polls_until_completed(monkeypatch, openrouter):
    seen: dict = {}
    states = iter([{"status": "pending"}, {"status": "in_progress"}, None])

    def post(url, headers=None, data=None, timeout=None):
        seen["payload"] = json.loads(data)
        return FakeResponse(202, {"id": "job1", "polling_url": "https://openrouter/videos/job1"})

    def get(url, headers=None, timeout=None):
        if url.endswith("content"):
            return FakeResponse(200, b"mp4")
        state = next(states)
        if state is None:
            return FakeResponse(
                200, {"status": "completed", "unsigned_urls": ["https://openrouter/videos/job1/content"]}
            )
        return FakeResponse(200, state)

    fake_requests(monkeypatch, post=post, get=get)
    openrouter.poll_seconds = 1.0
    request = VideoRequest(prompt="[Shot 1] ...", seconds=7.2, first_frame=PIXEL)
    assert openrouter.video("minimax/hailuo-3", request) == b"mp4"
    frames = seen["payload"]["frame_images"]
    assert frames == [
        {"type": "image_url", "image_url": {"url": PIXEL}, "frame_type": "first_frame"}
    ]
    assert seen["payload"]["duration"] == 7


def test_openrouter_video_reports_failure(monkeypatch, openrouter):
    fake_requests(
        monkeypatch,
        post=lambda *a, **k: FakeResponse(202, {"id": "j", "polling_url": "https://openrouter/videos/j"}),
        get=lambda *a, **k: FakeResponse(200, {"status": "failed", "error": "nope"}),
    )
    with pytest.raises(RenderError, match="failed"):
        openrouter.video("minimax/hailuo-3", VideoRequest(prompt="x"))
