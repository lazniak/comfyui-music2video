"""Tests for the per-shot audio cuts that feed lipsync (numpy stands in for torch)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

from music2prompts.util import slice_audio  # noqa: E402

SAMPLE_RATE = 48000


def ramp_audio(seconds: float = 10.0, channels: int = 2) -> dict:
    """Waveform whose value equals the sample index, so cuts are verifiable."""
    total = int(seconds * SAMPLE_RATE)
    waveform = np.tile(np.arange(total, dtype="float32"), (1, channels, 1))
    return {"waveform": waveform, "sample_rate": SAMPLE_RATE}


def test_cut_is_sample_accurate():
    clip = slice_audio(ramp_audio(), 2.0, 5.0)
    assert clip["sample_rate"] == SAMPLE_RATE
    assert clip["waveform"].shape[-1] == 3 * SAMPLE_RATE
    assert clip["waveform"][0, 0, 0] == 2.0 * SAMPLE_RATE
    assert clip["waveform"][0, 0, -1] == 5.0 * SAMPLE_RATE - 1


def test_channels_and_rate_are_preserved():
    clip = slice_audio(ramp_audio(channels=2), 0.0, 1.0)
    assert clip["waveform"].shape[:2] == (1, 2)
    mono = slice_audio(ramp_audio(channels=1), 0.0, 1.0)
    assert mono["waveform"].shape[:2] == (1, 1)


def test_shape_is_normalised_to_batch_channel_samples():
    flat = {"waveform": np.arange(SAMPLE_RATE, dtype="float32"), "sample_rate": SAMPLE_RATE}
    assert slice_audio(flat, 0.0, 0.5)["waveform"].ndim == 3


def test_padding_widens_both_sides_and_clamps_to_the_track():
    clip = slice_audio(ramp_audio(), 2.0, 5.0, padding=0.5)
    assert clip["waveform"].shape[-1] == 4 * SAMPLE_RATE
    assert clip["waveform"][0, 0, 0] == 1.5 * SAMPLE_RATE

    head = slice_audio(ramp_audio(), 0.0, 1.0, padding=0.5)
    assert head["waveform"][0, 0, 0] == 0.0
    assert head["waveform"].shape[-1] == int(1.5 * SAMPLE_RATE)

    tail = slice_audio(ramp_audio(seconds=10.0), 9.0, 10.0, padding=0.5)
    assert tail["waveform"].shape[-1] == int(1.5 * SAMPLE_RATE)


def test_clips_tile_the_track_without_gaps_or_overlap():
    audio = ramp_audio(seconds=12.0)
    boundaries = [(0.0, 5.0), (5.0, 9.5), (9.5, 12.0)]
    total = 0
    for start, end in boundaries:
        clip = slice_audio(audio, start, end)
        assert clip["waveform"][0, 0, 0] == round(start * SAMPLE_RATE)
        total += clip["waveform"].shape[-1]
    assert total == 12 * SAMPLE_RATE


def test_degenerate_range_still_returns_one_sample():
    clip = slice_audio(ramp_audio(), 3.0, 3.0)
    assert clip["waveform"].shape[-1] == 1


def test_rejects_non_audio_input():
    with pytest.raises(ValueError):
        slice_audio({"not": "audio"}, 0.0, 1.0)
