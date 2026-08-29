"""Rendering layer: payload shapes, concurrency and result parsing. No network."""

from __future__ import annotations

import base64
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import render as render_module  # noqa: E402
from music2prompts.providers import KEY_ENV  # noqa: E402
from music2prompts.render import (  # noqa: E402
    FalClient,
    _FalRejected,
    build_fal_image_payload,
    build_fal_video_payload,
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


@pytest.fixture(autouse=True)
def schemas(monkeypatch):
    """No test reaches fal for a schema; a test says what its endpoint declares."""
    known: dict[str, dict] = {}
    monkeypatch.setattr(render_module, "fal_schema", lambda model, timeout=6.0: known.get(model, {}))
    return known


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

    def fake_run(model, payload, optional_keys=(), spare=None, label="", kind="image"):
        seen.update(model=model, payload=payload, optional=optional_keys, spare=spare, label=label, kind=kind)
        return result

    client.run = fake_run  # type: ignore[assignment]
    return seen


class FakeResponse:
    def __init__(self, status_code: int, body: dict | bytes, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body
        # fal reports what it billed in a response header, so the stand-in needs one
        self.headers = headers or {}
        self.text = body.decode("utf-8", "replace") if isinstance(body, bytes) else json.dumps(body)
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
    for name in KEY_ENV["fal"]:  # every accepted variable must be gone
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RenderError, match="no fal.ai key"):
        make_media_client("fal")


# --------------------------------------------------------------------------- fal payloads


def test_media_seeds_are_clamped_too(fal):
    seen = record(fal, {"images": [{"url": data_uri(b"png")}]})
    fal.image("fal-ai/flux/dev", ImageRequest(prompt="x", seed=981447705429804))
    assert 0 <= seen["payload"]["seed"] <= 2**31 - 1


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


# --------------------------------------------------------------------------- fal schemas

WAN = {
    "required": ["start_image_url"],
    "properties": {
        "prompt": {"type": "string"},
        "start_image_url": {"type": "string"},
        "end_image_url": {"type": "string"},
        "duration": {"type": "integer", "minimum": 2, "maximum": 10},
        "resolution": {"enum": ["480p", "720p", "1080p"]},
        "aspect_ratio": {"enum": ["adaptive", "16:9", "9:16"]},
        "seed": {"type": "integer", "minimum": 0, "maximum": 999},
    },
}
KLING = {
    "required": ["prompt", "image_url"],
    "properties": {
        "prompt": {"type": "string"},
        "image_url": {"type": "string"},
        "duration": {"enum": ["5", "10"]},
    },
}
BANANA = {
    "required": ["prompt"],
    "properties": {
        "prompt": {"type": "string"},
        "aspect_ratio": {"enum": ["1:1", "16:9", "9:16"]},
        "num_images": {"type": "integer"},
        "output_format": {"enum": ["jpeg", "png"]},
        "image_urls": {"type": "array"},
    },
}


def test_the_image_field_comes_from_the_endpoint_not_a_guess(schemas, fal):
    """The bug this fixes: Wan wants start_image_url, so image_url was silently missing."""
    schemas["alibaba/wan-3.0-prime/image-to-video"] = WAN
    seen = record(fal, {"video": {"url": data_uri(b"mp4", "video/mp4")}})
    request = VideoRequest(prompt="push in", seconds=6.4, first_frame=PIXEL, seed=5)
    fal.video("alibaba/wan-3.0-prime/image-to-video", request)
    assert seen["payload"]["start_image_url"] == PIXEL
    assert "image_url" not in seen["payload"]


def test_fields_the_endpoint_does_not_declare_are_never_sent(schemas):
    payload, _ = build_fal_image_payload(
        "fal-ai/nano-banana-pro", ImageRequest(prompt="x", negative="blurry", seed=3), BANANA
    )
    assert set(payload) <= set(BANANA["properties"])
    assert "enable_safety_checker" not in payload and "negative_prompt" not in payload
    assert payload["aspect_ratio"] == "16:9", "an aspect enum replaces the image_size object"


def test_a_model_that_must_have_an_image_says_so_instead_of_failing_at_fal():
    with pytest.raises(RenderError, match="start_image_url"):
        build_fal_video_payload("alibaba/wan-3.0-prime/image-to-video", VideoRequest(prompt="x"), WAN)


def test_duration_is_clamped_and_matched_to_what_the_endpoint_allows():
    long_shot = VideoRequest(prompt="x", seconds=14.0, first_frame=PIXEL)
    assert build_fal_video_payload("m", long_shot, WAN)[0]["duration"] == 10
    assert build_fal_video_payload("m", long_shot, KLING)[0]["duration"] == "10", "enum durations are strings"
    short = VideoRequest(prompt="x", seconds=6.0, first_frame=PIXEL)
    assert build_fal_video_payload("m", short, KLING)[0]["duration"] == "5", "nearest allowed value"


def test_the_seed_respects_the_endpoints_own_range():
    payload, _ = build_fal_video_payload("m", VideoRequest(prompt="x", first_frame=PIXEL, seed=12345), WAN)
    assert 0 <= payload["seed"] <= 999


def test_an_unreadable_schema_still_produces_the_old_payload(schemas, fal):
    """fal aliases (bytedance-seed/...) have no schema document; those runs must still work."""
    seen = record(fal, {"images": [{"url": data_uri(b"png")}]})
    fal.image("bytedance-seed/seedream-5-0-lite", ImageRequest(prompt="x", negative="blurry"))
    assert seen["payload"]["negative_prompt"] == "blurry"
    assert "image_size" in seen["payload"]


def test_a_missing_field_reported_only_when_the_job_runs_is_retried(fal, monkeypatch):
    """fal validates some endpoints late: the 422 arrives from the result URL, not the POST."""
    calls: list[dict] = []

    def post(url, headers=None, data=None, timeout=None):
        calls.append(json.loads(data))
        return FakeResponse(200, {"status_url": "https://queue/status", "response_url": "https://queue/response"})

    def get(url, headers=None, timeout=None):
        if url.endswith("status"):
            return FakeResponse(200, {"status": "COMPLETED"})
        if "start_image_url" not in calls[-1]:
            return FakeResponse(
                422, {"detail": [{"type": "missing", "loc": ["body", "start_image_url"], "msg": "Field required"}]}
            )
        return FakeResponse(200, {"video": {"url": data_uri(b"mp4", "video/mp4")}})

    fake_requests(monkeypatch, post=post, get=get)
    request = VideoRequest(prompt="x", first_frame=PIXEL, seconds=6)
    assert fal.video("alibaba/wan-3.0-prime/image-to-video", request) == b"mp4"
    assert len(calls) == 2 and calls[1]["start_image_url"] == PIXEL


def test_a_refusal_we_cannot_repair_reports_what_fal_said(fal, monkeypatch):
    fake_requests(
        monkeypatch,
        post=lambda *a, **k: FakeResponse(422, {"detail": [{"type": "value_error", "msg": "prompt too long"}]}),
        get=lambda *a, **k: FakeResponse(200, {}),
    )
    with pytest.raises(RenderError, match="prompt too long"):
        fal.run("fal-ai/flux/dev", {"prompt": "x"})


def test_missing_fields_are_read_out_of_the_error_body():
    body = json.dumps(
        {"detail": [{"type": "missing", "loc": ["body", "image_url"]}, {"type": "value_error", "loc": ["body", "x"]}]}
    )
    assert _FalRejected("result", 422, body).missing == ["image_url"]
    assert _FalRejected("submit", 400, "not json").missing == []


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
