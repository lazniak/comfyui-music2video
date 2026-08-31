"""Joining clips that came from anywhere in the graph, and the soundtrack under them.

These encode and decode real files: the point of the node is what PyAV writes, and a
mocked muxer would prove nothing about it.
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import video as video_module  # noqa: E402

av = pytest.importorskip("av", reason="the muxing tests need PyAV")
np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------- fixtures


def clip(path: str, frames: int = 12, rate: int = 24, size=(64, 64), tone: float = 0.0) -> str:
    """A tiny mp4. ``tone`` above 0 writes a sine at that amplitude, else no audio track."""
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
    stream.options = {"crf": "30", "preset": "ultrafast"}

    audio_stream = None
    if tone > 0:
        audio_stream = container.add_stream("aac", rate=48000, layout="stereo")

    for index in range(frames):
        picture = np.full((size[1], size[0], 3), (index * 7) % 256, dtype="uint8")
        frame = av.VideoFrame.from_ndarray(picture, format="rgb24")
        frame.pts = index
        frame.time_base = Fraction(1, rate)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)

    if audio_stream is not None:
        samples = int(48000 * frames / rate)
        time = np.arange(samples, dtype="float32") / 48000.0
        wave = (np.sin(2 * math.pi * 440 * time) * tone).astype("float32")
        planes = np.ascontiguousarray(np.stack([wave, wave]))
        frame = av.AudioFrame.from_ndarray(planes, format="fltp", layout="stereo")
        frame.sample_rate = 48000
        frame.pts = 0
        frame.time_base = Fraction(1, 48000)
        for packet in audio_stream.encode(frame):
            container.mux(packet)
        for packet in audio_stream.encode(None):
            container.mux(packet)
    container.close()
    return path


def track(seconds: float = 1.0, rate: int = 48000, amplitude: float = 0.8) -> dict:
    """A ComfyUI AUDIO dict holding a steady tone."""
    time = np.arange(int(seconds * rate), dtype="float32") / rate
    wave = (np.sin(2 * math.pi * 220 * time) * amplitude).astype("float32")
    return {"waveform": np.stack([wave, wave])[None, ...], "sample_rate": rate}


def read_audio(path: str):
    """The muxed audio back as planar float32, or None when the file carries none."""
    return video_module._decode_audio(path, 48000, "stereo")


def peak(planes) -> float:
    return 0.0 if planes is None or planes.size == 0 else float(np.abs(planes).max())


# --------------------------------------------------------------------------- the muxer


def test_the_clips_end_up_in_one_file_on_one_grid(tmp_path):
    paths = [clip(str(tmp_path / f"{i}.mp4"), frames=12, rate=24, size=(64, 64)) for i in range(3)]
    paths.append(clip(str(tmp_path / "odd.mp4"), frames=12, rate=24, size=(48, 80)))
    out = str(tmp_path / "film.mp4")

    info = video_module.concat_clips(paths, out, audio_mode="none")

    assert os.path.exists(out)
    assert info["frames"] == 48, "every clip's frames, none dropped"
    assert info["fps"] == 24.0
    assert (info["width"], info["height"]) == (64, 64), "the odd one out is fitted, not followed"
    assert info["duration"] == pytest.approx(2.0, abs=0.05)


def test_the_shot_durations_decide_how_long_each_clip_lasts(tmp_path):
    """A clip longer than its shot is trimmed; a short one is held on its last frame."""
    paths = [clip(str(tmp_path / f"{i}.mp4"), frames=24, rate=24) for i in range(2)]
    out = str(tmp_path / "film.mp4")

    info = video_module.concat_clips(paths, out, audio_mode="none", clip_durations=[0.5, 1.5])

    assert info["frames_per_clip"] == [12, 36]
    assert info["trimmed_frames"] == 12, "the first clip's surplus is cut off"
    assert info["held_frames"] == 12, "the second is short of its shot and holds its last frame"


def test_source_audio_puts_the_track_under_the_film_and_nothing_else(tmp_path):
    paths = [clip(str(tmp_path / "a.mp4"), frames=24, rate=24, tone=0.9)]
    out = str(tmp_path / "film.mp4")

    info = video_module.concat_clips(paths, out, audio=track(1.0), audio_mode="music")

    assert info["audio"] == "music"
    assert peak(read_audio(out)) > 0.2, "the track is there"


def test_video_audio_keeps_what_the_clips_came_with(tmp_path):
    paths = [
        clip(str(tmp_path / "loud.mp4"), frames=24, rate=24, tone=0.9),
        clip(str(tmp_path / "silent.mp4"), frames=24, rate=24),
    ]
    out = str(tmp_path / "film.mp4")

    info = video_module.concat_clips(paths, out, audio=track(2.0), audio_mode="clips")

    assert info["audio"] == "clips"
    planes = read_audio(out)
    assert peak(planes[:, : 48000 // 2]) > 0.2, "the first clip's own audio survives"
    assert peak(planes[:, int(48000 * 1.2) :]) < 0.05, "the silent clip stays silent"


def test_mix_sums_the_track_and_the_clips(tmp_path):
    """The clip is silent for its second half, so only the track can be heard there."""
    paths = [
        clip(str(tmp_path / "loud.mp4"), frames=24, rate=24, tone=0.9),
        clip(str(tmp_path / "silent.mp4"), frames=24, rate=24),
    ]
    out = str(tmp_path / "film.mp4")

    info = video_module.concat_clips(
        paths, out, audio=track(2.0), audio_mode="mix", music_gain=1.0, clip_gain=0.5
    )

    assert info["audio"] == "mix"
    planes = read_audio(out)
    assert peak(planes[:, int(48000 * 1.2) :]) > 0.2, "the track plays on under the silent clip"
    assert peak(planes) <= 1.0001, "the sum is clipped to full scale, never wrapped"


def test_the_two_gains_actually_change_the_mix(tmp_path):
    paths = [clip(str(tmp_path / "a.mp4"), frames=24, rate=24, tone=0.5)]
    quiet, loud = str(tmp_path / "quiet.mp4"), str(tmp_path / "loud.mp4")

    video_module.concat_clips(paths, quiet, audio=track(1.0, amplitude=0.2),
                              audio_mode="mix", music_gain=0.1, clip_gain=0.1)
    video_module.concat_clips(paths, loud, audio=track(1.0, amplitude=0.2),
                              audio_mode="mix", music_gain=1.0, clip_gain=1.0)

    assert peak(read_audio(quiet)) < peak(read_audio(loud))


def test_mix_without_a_track_falls_back_rather_than_failing(tmp_path):
    paths = [clip(str(tmp_path / "a.mp4"), frames=24, rate=24, tone=0.9)]
    out = str(tmp_path / "film.mp4")

    info = video_module.concat_clips(paths, out, audio=None, audio_mode="mix")

    assert info["audio"] == "clips", "no track to mix with, so the clips' own audio is kept"


def test_mix_with_silent_clips_falls_back_to_the_track(tmp_path):
    paths = [clip(str(tmp_path / "a.mp4"), frames=24, rate=24)]
    out = str(tmp_path / "film.mp4")

    info = video_module.concat_clips(paths, out, audio=track(1.0), audio_mode="mix")

    assert info["audio"] == "music", "nothing to mix in, so the track goes under on its own"


def test_mix_is_one_of_the_offered_modes():
    assert "mix" in video_module.AUDIO_MODES


# --------------------------------------------------------------------------- the node


def node():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.concat import Music2VideoConcat

    return Music2VideoConcat


class FakeVideo:
    """A VIDEO that is already a file on disk, the way VideoFromFile behaves."""

    def __init__(self, path, trimmed=False):
        self.path = path
        self.trimmed = trimmed
        self.saved_to = None

    def get_stream_source(self):
        return self.path

    def get_active_trim_window(self):
        return (0.5, 1.0) if self.trimmed else (0.0, 0.0)

    def save_to(self, target, **_):
        self.saved_to = target
        with open(self.path, "rb") as source, open(target, "wb") as sink:
            sink.write(source.read())


def test_every_widget_arrives_wrapped_in_a_list_and_is_unwrapped_once():
    """is_input_list hands widgets in as one-element lists; reading [0] blindly is a trap."""
    concat = pytest.importorskip("music2prompts.concat")
    assert concat._first(["mix"]) == "mix"
    assert concat._first([], "source audio") == "source audio"
    assert concat._first(None, 1.0) == 1.0
    assert concat._first("mix") == "mix", "a bare value must survive too"


def test_the_widget_names_map_onto_what_the_muxer_calls_them():
    concat = pytest.importorskip("music2prompts.concat")
    assert set(concat.AUDIO_SOURCES.values()) <= set(video_module.AUDIO_MODES)
    assert concat.AUDIO_SOURCES["source audio"] == "music"
    assert concat.AUDIO_SOURCES["video audio"] == "clips"


def test_a_clip_already_on_disk_is_read_where_it_lies(tmp_path):
    concat = pytest.importorskip("music2prompts.concat")
    path = clip(str(tmp_path / "a.mp4"))
    paths, temporary = concat._materialise([FakeVideo(path)])
    assert paths == [path] and temporary == [], "no copy, no temp file to clean up"


def test_a_trimmed_clip_is_written_out_first(tmp_path):
    """The trim window lives in the VIDEO object; the muxer only sees files."""
    concat = pytest.importorskip("music2prompts.concat")
    path = clip(str(tmp_path / "a.mp4"))
    paths, temporary = concat._materialise([FakeVideo(path, trimmed=True)])
    assert paths != [path] and paths == temporary, "a copy was made and is ours to delete"
    assert os.path.exists(paths[0])
    os.unlink(paths[0])


def test_the_node_joins_the_clips_and_hands_back_where_it_wrote_them(tmp_path, monkeypatch):
    cls = node()
    from music2prompts import render as render_module

    monkeypatch.setattr(render_module, "output_directory", lambda *a, **k: str(tmp_path))
    clips = [FakeVideo(clip(str(tmp_path / f"{i}.mp4"), frames=24, rate=24)) for i in range(2)]

    result = cls.execute(videos=clips, audio_mode=["silent"], filename_prefix=["test"])

    path, duration = result.args[1], result.args[2]
    assert os.path.exists(path) and path.endswith("_concat.mp4")
    assert os.path.basename(path).startswith("test_")
    assert duration == pytest.approx(2.0, abs=0.05)


def test_the_pipe_supplies_the_shot_lengths(tmp_path, monkeypatch):
    cls = node()
    from music2prompts import pipe as pipe_module
    from music2prompts import render as render_module

    monkeypatch.setattr(render_module, "output_directory", lambda *a, **k: str(tmp_path))
    clips = [FakeVideo(clip(str(tmp_path / f"{i}.mp4"), frames=24, rate=24)) for i in range(2)]
    packed = pipe_module.pack(durations=[0.5, 0.5])

    result = cls.execute(videos=clips, pipe=[packed], audio_mode=["silent"])

    assert result.args[2] == pytest.approx(1.0, abs=0.05), "each clip cut to its half-second shot"


def test_a_pipe_that_does_not_match_the_clips_is_ignored_rather_than_trusted(tmp_path, monkeypatch, caplog):
    """A failed render leaves one clip fewer, and the lists no longer mean the same thing."""
    import logging

    cls = node()
    from music2prompts import pipe as pipe_module
    from music2prompts import render as render_module

    monkeypatch.setattr(render_module, "output_directory", lambda *a, **k: str(tmp_path))
    clips = [FakeVideo(clip(str(tmp_path / "0.mp4"), frames=24, rate=24))]
    packed = pipe_module.pack(durations=[0.5, 0.5, 0.5])

    with caplog.at_level(logging.WARNING):
        result = cls.execute(videos=clips, pipe=[packed], audio_mode=["silent"])

    assert "3 shot(s) but 1 clip(s)" in caplog.text
    assert result.args[2] == pytest.approx(1.0, abs=0.05), "the clip keeps its own length"


def test_nothing_to_join_says_so_instead_of_writing_an_empty_file():
    cls = node()
    with pytest.raises(ValueError, match="no clips to join"):
        cls.execute(videos=[])


# --------------------------------------------------------------------------- where it lands


def directory_like_comfyui(tmp_path):
    """The real output_directory: base + subfolder, created on the way."""
    def directory(subfolder="", temporary=False):
        path = os.path.join(str(tmp_path), subfolder)
        os.makedirs(path, exist_ok=True)
        return path

    return directory


def test_a_film_can_be_written_into_a_folder_that_does_not_exist_yet(tmp_path):
    """The bug this covers cost half an hour: PyAV opens the file at the first packet, so
    a missing folder surfaced as a bare '[Errno 2] No such file or directory' from inside
    the muxer, naming no path, after every clip had already been decoded and scaled."""
    target = tmp_path / "films" / "tour" / "night.mp4"
    info = video_module.concat_clips(
        [clip(str(tmp_path / "a.mp4"), frames=12, rate=24)], str(target), audio_mode="none"
    )
    assert os.path.exists(info["path"])


def test_the_pipe_names_the_film_when_the_widget_is_left_empty(tmp_path, monkeypatch):
    """So the film lands in the run's project folder, beside the clips it was cut from."""
    cls = node()
    from music2prompts import pipe as pipe_module
    from music2prompts import render as render_module

    monkeypatch.setattr(render_module, "output_directory", directory_like_comfyui(tmp_path))
    clips = [FakeVideo(clip(str(tmp_path / "0.mp4"), frames=24, rate=24))]
    packed = pipe_module.pack(final_video_name="music2prompts/song_v003/music2video_STAMP_final")

    path = cls.execute(videos=clips, pipe=[packed], audio_mode=["silent"]).args[1]

    assert os.path.basename(path) == "music2video_STAMP_final.mp4"
    assert os.path.basename(os.path.dirname(path)) == "song_v003"


def test_a_prefix_typed_into_the_widget_wins_over_the_pipe(tmp_path, monkeypatch):
    cls = node()
    from music2prompts import pipe as pipe_module
    from music2prompts import render as render_module

    monkeypatch.setattr(render_module, "output_directory", directory_like_comfyui(tmp_path))
    clips = [FakeVideo(clip(str(tmp_path / "0.mp4"), frames=24, rate=24))]
    packed = pipe_module.pack(final_video_name="music2prompts/song_v003/music2video_STAMP_final")

    path = cls.execute(
        videos=clips, pipe=[packed], audio_mode=["silent"], filename_prefix=["mine"]
    ).args[1]

    assert os.path.basename(path).startswith("mine_") and path.endswith("_concat.mp4")


def test_without_a_pipe_or_a_prefix_it_still_names_itself(tmp_path, monkeypatch):
    cls = node()
    from music2prompts import render as render_module

    monkeypatch.setattr(render_module, "output_directory", directory_like_comfyui(tmp_path))
    clips = [FakeVideo(clip(str(tmp_path / "0.mp4"), frames=24, rate=24))]

    path = cls.execute(videos=clips, audio_mode=["silent"]).args[1]

    assert os.path.basename(path).startswith("music2video_")
    assert os.path.basename(os.path.dirname(path)) == render_module.SUBFOLDER


def test_a_prefix_with_a_subfolder_writes_into_it(tmp_path, monkeypatch):
    """ComfyUI's own save nodes take 'video/ComfyUI'; typing that here used to be errno 2."""
    cls = node()
    from music2prompts import render as render_module

    monkeypatch.setattr(render_module, "output_directory", directory_like_comfyui(tmp_path))
    clips = [FakeVideo(clip(str(tmp_path / "0.mp4"), frames=24, rate=24))]

    path = cls.execute(videos=clips, audio_mode=["silent"], filename_prefix=["video/ComfyUI"]).args[1]

    assert os.path.basename(os.path.dirname(path)) == "video"
    assert os.path.exists(path)
