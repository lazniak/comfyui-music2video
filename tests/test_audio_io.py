"""Encoding a shot's audio slice into something a media API will take."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import audio_io  # noqa: E402

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
av = pytest.importorskip("av", reason="PyAV lives in the ComfyUI environment")

RATE = 44100


def tone(seconds: float = 6.0, channels: int = 2, rate: int = RATE) -> dict:
    steps = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * 220 * steps)
    stacked = np.stack([wave * (1.0 - 0.3 * index) for index in range(channels)])
    return {"waveform": torch.tensor(stacked[None, ...], dtype=torch.float32), "sample_rate": rate}


def test_a_clip_becomes_an_mp3_that_decodes_back_to_its_own_length():
    payload = audio_io.encode(tone(6.0), "mp3")
    assert payload[:3] in (b"ID3", b"\xff\xfb") or payload[:2] == b"\xff\xf3"
    import io

    with av.open(io.BytesIO(payload)) as container:
        assert container.streams.audio[0].rate == RATE
        assert abs(float(container.duration or 0) / 1_000_000 - 6.0) < 0.1


def test_mp3_is_an_order_of_magnitude_smaller_than_wav():
    """The clip travels in the same JSON body as a full-resolution start frame."""
    clip = tone(6.0)
    assert len(audio_io.encode(clip, "wav")) > 8 * len(audio_io.encode(clip, "mp3"))


def test_a_mono_clip_is_written_with_a_mono_layout():
    payload = audio_io.encode(tone(3.0, channels=1, rate=22050), "mp3")
    assert payload, "a mono waveform must not need an explicit layout from the caller"


def test_a_non_contiguous_slice_is_accepted():
    """A torch stride-slice is not C-contiguous, and PyAV refuses those outright."""
    clip = tone(4.0)
    clip["waveform"] = clip["waveform"][:, :, ::2]
    clip["sample_rate"] = RATE // 2
    assert audio_io.encode(clip, "mp3")


def test_more_than_stereo_is_folded_down():
    steps = np.zeros((1, 6, RATE), dtype="float32")
    assert audio_io.planes({"waveform": torch.tensor(steps), "sample_rate": RATE}).shape[0] == 2


def test_the_duration_matches_the_slice():
    assert abs(audio_io.duration(tone(2.5)) - 2.5) < 0.001


def test_the_data_uri_declares_the_right_media_type():
    assert audio_io.data_uri(tone(2.0), "mp3").startswith("data:audio/mpeg;base64,")


def test_an_empty_or_missing_waveform_is_a_clear_error():
    with pytest.raises(audio_io.AudioError, match="no waveform"):
        audio_io.encode({}, "mp3")
    with pytest.raises(audio_io.AudioError, match="empty"):
        audio_io.encode({"waveform": torch.zeros(1, 2, 0), "sample_rate": RATE}, "mp3")


def test_an_unknown_format_names_the_ones_that_work():
    with pytest.raises(audio_io.AudioError, match="mp3"):
        audio_io.encode(tone(1.0), "ogg")
