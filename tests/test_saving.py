"""What a run leaves on disk: the images that were paid for, the transcript, the analysis."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import render as render_module  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"0" * 32


@pytest.fixture(autouse=True)
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        render_module, "output_directory", lambda subfolder="", temporary=False: str(tmp_path)
    )
    return tmp_path


def test_rendered_frames_land_in_the_output_folder(output):
    paths = render_module.save_images([PNG, PNG], "take", "frame", stamp="STAMP")
    assert [os.path.basename(path) for path in paths] == [
        "take_STAMP_frame001.png",
        "take_STAMP_frame002.png",
    ]
    assert open(paths[0], "rb").read() == PNG


def test_the_extension_follows_what_the_provider_returned(output):
    paths = render_module.save_images([JPEG, WEBP], "take", "frame", stamp="S")
    assert [os.path.splitext(path)[1] for path in paths] == [".jpg", ".webp"]


def test_a_failed_shot_is_skipped_without_shifting_the_others(output):
    paths = render_module.save_images([PNG, None, PNG], "take", "frame", stamp="S")
    names = [os.path.basename(path) for path in paths]
    assert names == ["take_S_frame001.png", "take_S_frame003.png"], "shot numbers must stay honest"


def test_images_and_clips_of_one_run_share_a_timestamp(output):
    stamp = render_module.run_stamp()
    image = render_module.save_images([PNG], "take", "frame", stamp=stamp)[0]
    clip = render_module.save_videos([b"mp4"], "take", temporary=False, stamp=stamp)[0]
    assert stamp in os.path.basename(image) and stamp in os.path.basename(clip)


def test_a_run_that_rendered_nothing_writes_nothing(output):
    assert render_module.save_images([None, None], "take") == []
    assert os.listdir(output) == []


# --------------------------------------------------------------------------- sidecars

from music2prompts.shots import ShotSlot  # noqa: E402


def node():
    """The node class, or a skip: importing it needs ComfyUI's V3 API on sys.path."""
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.node import Music2PromptsLM

    return Music2PromptsLM


def slot(index, start, end, section=""):
    return ShotSlot(index=index, start=start, end=end, section=section)


def test_the_transcript_is_written_next_to_the_run(output):
    path = node()._save_text("hello", "take", "transcript", "txt", "STAMP")
    assert os.path.basename(path) == "take_STAMP_transcript.txt"
    assert open(path, encoding="utf-8").read() == "hello"


def test_the_transcript_document_lines_the_lyrics_up_with_the_shots():
    transcription = {
        "language": "Polish",
        "text": "raz dwa trzy",
        "words": [
            {"word": "raz", "start": 0.2, "end": 0.5},
            {"word": "dwa", "start": 1.1, "end": 1.4},
            {"word": "trzy", "start": 7.0, "end": 7.3},
        ],
    }
    slots = [slot(1, 0.0, 6.0, "verse"), slot(2, 6.0, 12.0, "chorus")]
    document = node()._transcript_document(
        transcription, slots, ["[Shot 1] push in", "[Shot 2] pull out"], ["a lit room", "a dark street"]
    )
    assert "language: Polish" in document
    assert "raz dwa" in document.split("[shot 2]")[0]
    assert "trzy" in document.split("[shot 2]")[1]
    assert "a dark street" in document and "[Shot 2] pull out" in document


def test_a_silent_shot_says_so_rather_than_looking_empty():
    slots = [slot(1, 0.0, 6.0)]
    document = node()._transcript_document({"text": ""}, slots, [], [])
    assert "(instrumental)" in document and "(no speech detected)" in document
