"""Assembling the finished film. The real encoding tests need PyAV (ComfyUI ships it)."""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import video as video_module  # noqa: E402
from music2prompts.video import _snap_rate, _target_geometry, _target_rate  # noqa: E402


def info(width, height, rate=None, audio=False, rotation=0):
    return {
        "path": "x.mp4",
        "width": width,
        "height": height,
        "rate": Fraction(rate) if rate else None,
        "rotation": rotation,
        "has_audio": audio,
    }


# --------------------------------------------------------------------------- helpers


def test_geometry_follows_the_most_common_clip():
    infos = [info(1280, 720), info(1280, 720), info(640, 360)]
    assert _target_geometry(infos, None, None) == (1280, 720)


def test_a_tie_goes_to_the_larger_frame():
    """Half-landscape/half-portrait must not depend on which clip came back first."""
    infos = [info(640, 360), info(1080, 1920)]
    assert _target_geometry(infos, None, None) == (1080, 1920)
    assert _target_geometry(list(reversed(infos)), None, None) == (1080, 1920)


def test_explicit_geometry_wins_and_is_made_even():
    assert _target_geometry([info(640, 360)], 855, 481) == (854, 480)


def test_rate_defaults_to_the_fastest_clip():
    assert _target_rate([info(0, 0, 24), info(0, 0, 30)], None) == Fraction(30)
    assert _target_rate([info(0, 0)], None) == Fraction(24)


def test_typed_ntsc_rates_snap_to_their_broadcast_fractions():
    assert _target_rate([], 23.976) == Fraction(24000, 1001)
    assert _target_rate([], 29.97) == Fraction(30000, 1001)
    assert _target_rate([], 25) == Fraction(25)


def test_audio_rates_snap_to_something_aac_can_open():
    assert _snap_rate(44100) == 44100
    assert _snap_rate(33000) == 32000  # ComfyUI's own writer crashes on this one


# --------------------------------------------------------------------------- encoding

av = pytest.importorskip("av", reason="PyAV lives in the ComfyUI environment")
np = pytest.importorskip("numpy")


def make_clip(path, width, height, fps, seconds, audio=False):
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=Fraction(fps))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    track = container.add_stream("aac", rate=48000, layout="stereo") if audio else None
    for index in range(int(fps * seconds)):
        array = np.full((height, width, 3), 40 + (index * 7) % 200, dtype="uint8")
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame.pts, frame.time_base = index, Fraction(1, int(fps))
        for packet in stream.encode(frame.reformat(format="yuv420p")):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    if track is not None:
        samples = np.zeros((2, int(48000 * seconds)), dtype="float32")
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
        frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
        for packet in track.encode(frame):
            container.mux(packet)
        for packet in track.encode(None):
            container.mux(packet)
    container.close()
    return str(path)


def probe(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
    with av.open(path) as container:
        stream = container.streams.video[0]
        audio = container.streams.audio[0] if container.streams.audio else None
        return {
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate),
            "frames": frames,
            "audio_rate": audio.rate if audio else None,
        }


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    directory = tmp_path_factory.mktemp("clips")
    return [
        make_clip(directory / "a.mp4", 640, 360, 24, 1.0, audio=True),
        make_clip(directory / "b.mp4", 1280, 720, 30, 1.0),
        make_clip(directory / "c.mp4", 360, 640, 25, 1.0, audio=True),  # portrait
    ]


def test_clips_of_different_size_and_rate_become_one_film(clips, tmp_path):
    out = str(tmp_path / "film.mp4")
    result = video_module.concat_clips(clips, out, audio_mode="none")
    assert result["width"], result["height"] == (1280, 720)
    measured = probe(out)
    assert measured["frames"] == result["frames"]
    assert measured["width"] == 1280 and measured["height"] == 720


def test_every_clip_lasts_exactly_as_long_as_its_shot(clips, tmp_path):
    out = str(tmp_path / "timed.mp4")
    result = video_module.concat_clips(
        clips, out, audio_mode="none", clip_durations=[2.0, 0.5, 1.0], fps=24
    )
    assert result["frames_per_clip"] == [48, 12, 24], result["frames_per_clip"]
    assert abs(result["duration"] - 3.5) < 0.01


def test_the_music_track_is_muxed_and_trimmed_to_the_film(clips, tmp_path):
    torch = pytest.importorskip("torch")
    out = str(tmp_path / "music.mp4")
    audio = {"waveform": torch.zeros(1, 2, 44100 * 30), "sample_rate": 44100}
    result = video_module.concat_clips(clips, out, audio=audio, audio_mode="music")
    assert result["audio"] == "music"
    assert probe(out)["audio_rate"] == 44100


def test_an_unusual_sample_rate_is_snapped_instead_of_crashing(clips, tmp_path):
    torch = pytest.importorskip("torch")
    out = str(tmp_path / "odd.mp4")
    audio = {"waveform": torch.zeros(1, 1, 33000 * 3), "sample_rate": 33000}
    video_module.concat_clips(clips[:1], out, audio=audio, audio_mode="music")
    assert probe(out)["audio_rate"] == 32000


def test_a_portrait_clip_is_letterboxed_not_stretched(clips, tmp_path):
    out = str(tmp_path / "pad.mp4")
    video_module.concat_clips([clips[2]], out, audio_mode="none", width=1280, height=720, fit="pad")
    with av.open(out) as container:
        frame = next(container.decode(container.streams.video[0]))
        columns = frame.to_ndarray(format="rgb24").mean(axis=(0, 2))
    filled = np.nonzero(columns > 8)[0]
    assert columns[:100].mean() == 0, "left bar must be black"
    assert 300 < filled.min() < 500 and 800 < filled.max() < 1000


def test_clip_audio_can_be_kept_instead_of_the_music(clips, tmp_path):
    out = str(tmp_path / "keep.mp4")
    result = video_module.concat_clips(clips, out, audio_mode="clips")
    assert result["audio"] == "clips"
    assert probe(out)["audio_rate"] == 48000


def test_asking_for_music_without_a_track_falls_back(clips, tmp_path):
    out = str(tmp_path / "fallback.mp4")
    result = video_module.concat_clips(clips, out, audio=None, audio_mode="music")
    assert result["audio"] == "clips"


def test_missing_files_are_reported(tmp_path):
    with pytest.raises(video_module.VideoError, match="nothing to concatenate"):
        video_module.concat_clips(["nope.mp4"], str(tmp_path / "x.mp4"))


# --------------------------------------------------------------------------- assembly precision


def test_the_cuts_are_measured_from_the_running_boundary_not_shot_by_shot():
    """Rounding each shot on its own lets one shot's error push the next cut off the beat."""
    from fractions import Fraction

    from music2prompts.video import frame_budgets

    durations = [6.084, 6.037, 5.480, 6.083, 6.293, 5.805, 5.859]  # a real beat-snapped plan
    budgets = frame_budgets(durations, Fraction(30))
    assert sum(budgets) == round(sum(durations) * 30), "the film is exactly as long as the plan"
    boundary = 0
    for index, budget in enumerate(budgets):
        boundary += budget
        wanted = sum(durations[: index + 1]) * 30
        assert abs(boundary - wanted) <= 0.5, "every cut sits within half a frame of its beat"


def test_no_shot_is_left_without_a_single_frame():
    from fractions import Fraction

    from music2prompts.video import frame_budgets

    assert frame_budgets([0.001, 0.001], Fraction(30)) == [1, 1]
