"""Live previews: what reaches the browser while a render is still running."""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import preview as preview_module  # noqa: E402
from music2prompts import render as render_module  # noqa: E402
from music2prompts.preview import EVENT, PreviewFeed  # noqa: E402

PNG = b"\x89PNG\r\n" + b"0" * 32


@pytest.fixture
def sent(monkeypatch):
    """Capture what the node would push over ComfyUI's websocket."""
    events: list[tuple[str, dict]] = []

    class FakeServer:
        instance = None

        def send_sync(self, event, data, sid=None):
            events.append((event, data))

    server = types.ModuleType("server")
    server.PromptServer = FakeServer
    FakeServer.instance = FakeServer()
    monkeypatch.setitem(sys.modules, "server", server)
    return events


@pytest.fixture
def feed(tmp_path, sent, monkeypatch):
    monkeypatch.setattr(
        render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path)
    )
    return PreviewFeed("42", "take", enabled=True)


def test_a_finished_image_is_written_and_announced(feed, sent, tmp_path):
    path = feed.publish("image", 0, PNG, label="shot 1", total=4)
    assert os.path.exists(path) and open(path, "rb").read() == PNG
    event, data = sent[-1]
    assert event == EVENT
    assert data["node"] == "42" and data["kind"] == "image" and data["total"] == 4
    assert data["type"] == "temp" and data["subfolder"] == preview_module.SUBFOLDER
    assert data["filename"] == os.path.basename(path)


def test_clips_are_written_as_mp4(feed):
    assert feed.publish("video", 2, b"mp4 bytes").endswith("video003.mp4")


def test_nothing_is_written_when_the_preview_is_switched_off(tmp_path, sent, monkeypatch):
    monkeypatch.setattr(render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path))
    off = PreviewFeed("42", "take", enabled=False)
    assert off.publish("image", 0, PNG) == ""
    assert not sent and not os.listdir(tmp_path)


def test_a_node_without_an_id_stays_quiet(tmp_path, sent, monkeypatch):
    """UNIQUE_ID is missing outside a real queue run - the render must still go ahead."""
    monkeypatch.setattr(render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path))
    assert PreviewFeed(None, "take").publish("image", 0, PNG) == ""
    assert not sent


def test_reset_empties_the_gallery_before_a_new_take(feed, sent):
    feed.publish("image", 0, PNG)
    feed.reset(total=3)
    assert feed.items == []
    assert sent[-1][1] == {"node": "42", "reset": True, "total": 3}


def test_written_clips_are_offered_back_so_nothing_is_written_twice(feed):
    feed.publish("video", 0, b"a")
    feed.publish("video", 1, b"b")
    feed.publish("image", 0, PNG)
    paths = feed.paths("video")
    assert sorted(paths) == [0, 1]
    assert all(path.endswith(".mp4") for path in paths.values())


def test_the_ui_payload_carries_every_item(feed):
    assert feed.ui() == {}
    feed.publish("image", 0, PNG)
    items = feed.ui()["m2p_preview"]
    assert len(items) == 1 and items[0]["filename"].endswith(".png")


def test_a_disk_failure_costs_the_preview_not_the_render(tmp_path, sent, monkeypatch):
    monkeypatch.setattr(render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path / "gone"))
    broken = PreviewFeed("7", "take")
    monkeypatch.setattr(broken, "directory", lambda: str(tmp_path / "gone"))
    assert broken.publish("image", 0, PNG) == ""  # no exception


def test_a_closed_socket_costs_the_preview_not_the_render(tmp_path, monkeypatch):
    monkeypatch.setattr(render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path))
    monkeypatch.delitem(sys.modules, "server", raising=False)
    monkeypatch.setattr(preview_module, "_send", lambda payload: False)
    assert PreviewFeed("7", "take").publish("image", 0, PNG)


# --------------------------------------------------------------------------- render hookup


class StubClient:
    name = "stub"

    def image(self, model, request):
        return f"image:{request.prompt}".encode()

    def video(self, model, request):
        return f"video:{request.prompt}".encode()


def test_each_result_is_handed_over_as_it_lands(monkeypatch):
    seen: list[tuple[int, bytes]] = []
    requests = [render_module.ImageRequest(prompt=str(index)) for index in range(3)]
    payloads = render_module.render_images(
        StubClient(), "m", requests, concurrency=1, on_done=lambda index, data: seen.append((index, data))
    )
    assert payloads == [b"image:0", b"image:1", b"image:2"]
    assert seen == [(0, b"image:0"), (1, b"image:1"), (2, b"image:2")]


def test_a_broken_preview_does_not_lose_the_render():
    def boom(index, data):
        raise RuntimeError("socket closed")

    requests = [render_module.VideoRequest(prompt="x")]
    assert render_module.render_videos(StubClient(), "m", requests, 1, on_done=boom) == [b"video:x"]


def test_clips_already_written_for_the_preview_are_not_written_again(tmp_path, monkeypatch):
    monkeypatch.setattr(render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path))
    existing = tmp_path / "already.mp4"
    existing.write_bytes(b"clip")
    paths = render_module.save_videos([b"clip", b"other"], "take", temporary=True, reuse={0: str(existing)})
    assert paths[0] == str(existing)
    assert paths[1] != str(existing) and os.path.exists(paths[1])


def test_the_output_folder_is_always_written_fresh(tmp_path, monkeypatch):
    """Saving for real must not hand back a temp file that ComfyUI will delete."""
    monkeypatch.setattr(render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path))
    existing = tmp_path / "already.mp4"
    existing.write_bytes(b"clip")
    paths = render_module.save_videos([b"clip"], "take", temporary=False, reuse={0: str(existing)})
    assert paths[0] != str(existing)
