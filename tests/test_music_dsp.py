"""Tests for the built-in numpy DSP backend (skipped when numpy is missing)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

from music2prompts import music_dsp  # noqa: E402


def click_track(bpm: float, seconds: float = 30.0, sample_rate: int = 22050):
    """A percussive click every beat plus a quiet tone, i.e. an unambiguous tempo."""
    total = int(seconds * sample_rate)
    audio = np.random.randn(total).astype("float32") * 0.001
    period = 60.0 / bpm
    click = np.exp(-np.linspace(0, 12, int(0.03 * sample_rate))).astype("float32")
    time = 0.0
    while time < seconds:
        start = int(time * sample_rate)
        end = min(total, start + len(click))
        audio[start:end] += click[: end - start]
        time += period
    return audio, sample_rate


@pytest.mark.parametrize("bpm", [90.0, 120.0, 140.0])
def test_numpy_backend_finds_the_tempo(bpm):
    audio, sample_rate = click_track(bpm)
    result = music_dsp._analyze_numpy(audio, sample_rate, duration=len(audio) / sample_rate)
    assert result["backend"] == "numpy"
    assert abs(result["bpm"] - bpm) < 4.0, f"detected {result['bpm']} for a {bpm} BPM track"


def test_beats_land_on_the_clicks():
    bpm = 120.0
    audio, sample_rate = click_track(bpm, seconds=20.0)
    result = music_dsp._analyze_numpy(audio, sample_rate, duration=len(audio) / sample_rate)
    period = 60.0 / bpm
    assert len(result["beats"]) > 10
    for beat in result["beats"][:20]:
        offset = beat % period
        assert min(offset, period - offset) < 0.12, f"beat at {beat}s is off the grid"


def test_analysis_shape_is_complete():
    audio, sample_rate = click_track(100.0, seconds=45.0)
    result = music_dsp.analyze(audio, sample_rate, enabled=True)
    assert set(result) >= {"duration", "bpm", "beats", "sections", "energy_curve", "backend"}
    assert result["sections"][0]["start"] == 0.0
    assert abs(result["sections"][-1]["end"] - result["duration"]) < 0.01
    for previous, current in zip(result["sections"], result["sections"][1:]):
        assert current["start"] == previous["end"]
    assert all(0.0 <= point["e"] <= 1.0 for point in result["energy_curve"])


def test_disabled_analysis_returns_a_single_section():
    audio, sample_rate = click_track(100.0, seconds=20.0)
    result = music_dsp.analyze(audio, sample_rate, enabled=False)
    assert result["bpm"] == 0.0
    assert len(result["sections"]) == 1


def test_compact_for_llm_shrinks_long_lists():
    analysis = {
        "duration": 300.0,
        "bpm": 120.0,
        "beats": [round(i * 0.5, 2) for i in range(600)],
        "sections": [{"name": "Part 1", "start": 0.0, "end": 300.0, "energy": 0.5}],
        "energy_curve": [{"t": float(i), "e": 0.5} for i in range(300)],
    }
    compact = music_dsp.compact_for_llm(analysis, max_beats=48, max_energy_points=40)
    assert len(compact["beats_sample"]) <= 48
    assert len(compact["energy_curve"]) <= 40
    assert compact["bpm"] == 120.0
