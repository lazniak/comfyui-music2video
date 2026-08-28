"""Timing tests for the shot planner (no torch, no network)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.shots import attach_lyrics, plan_shots, target_shot_length  # noqa: E402


def test_shots_cover_the_whole_track_without_gaps():
    shots = plan_shots(duration=61.0, clip_seconds=6.0, snap_to_beats=False)
    assert shots[0].start == 0.0
    assert abs(shots[-1].end - 61.0) < 0.01
    for previous, current in zip(shots, shots[1:]):
        assert abs(current.start - previous.end) < 0.01
        assert current.index == previous.index + 1


def test_shot_lengths_respect_the_h3_window():
    shots = plan_shots(duration=120.0, clip_seconds=6.0, min_seconds=5.0, max_seconds=15.0, snap_to_beats=False)
    for shot in shots:
        assert 5.0 - 0.01 <= shot.duration <= 15.0 + 0.01


def test_explicit_shot_count_is_honoured():
    shots = plan_shots(duration=60.0, num_shots=6, snap_to_beats=False)
    assert len(shots) == 6
    assert abs(shots[0].duration - 10.0) < 0.01


def test_short_track_returns_single_shot():
    shots = plan_shots(duration=3.2, clip_seconds=6.0)
    assert len(shots) == 1
    assert shots[0].duration == 3.2


def test_boundaries_snap_to_beats():
    beats = [round(0.5 * i, 3) for i in range(1, 120)]
    unsnapped = plan_shots(duration=60.0, clip_seconds=6.0, snap_to_beats=False)
    snapped = plan_shots(duration=60.0, clip_seconds=6.0, beats=beats, snap_to_beats=True)
    assert len(snapped) == len(unsnapped)
    for shot in snapped[1:]:
        assert abs(shot.start * 2 - round(shot.start * 2)) < 0.01  # lands on a 0.5 s beat


def test_higher_dynamicity_shortens_shots():
    calm = target_shot_length(6.0, dynamicity=0.0, min_s=5.0, max_s=15.0)
    kinetic = target_shot_length(6.0, dynamicity=1.0, min_s=5.0, max_s=15.0)
    assert kinetic < calm


def test_lyrics_land_in_the_right_shot():
    shots = plan_shots(duration=30.0, num_shots=3, snap_to_beats=False)
    words = [
        {"start": 1.0, "end": 1.4, "text": "first"},
        {"start": 12.0, "end": 12.5, "text": "second"},
        {"start": 25.0, "end": 25.6, "text": "third"},
    ]
    shots = attach_lyrics(shots, words)
    assert shots[0].lyrics == "first"
    assert shots[1].lyrics == "second"
    assert shots[2].lyrics == "third"
