"""Glue the rendered clips into one finished film.

PyAV only - no ffmpeg binary, nothing new to install (ComfyUI already ships ``av``).
Every clip is decoded, placed on one fixed output grid (size, frame rate) and
re-encoded into a single H.264/mp4, with the music muxed in as AAC.

Why not the cheap paths:

* stream copy is only legal when every clip shares codec, geometry, timing **and**
  codec parameter sets - clips from different vendors do not, and a mismatch decodes
  to garbage without raising, so this module never copies;
* decoding everything into one big tensor (ComfyUI's ``VideoComponents``) needs the
  whole film in RAM - 30 shots of 6 s at 1080p is gigabytes - so this module streams.

Geometry is taken from the *display* size (a clip carrying a rotation matrix is
rotated first), and clips whose aspect differs from the output are letterboxed
rather than stretched.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from fractions import Fraction
from typing import Any, Callable, Sequence

from .util import PREFIX, log, warn

# av.codec.Codec("aac", "w").audio_rates - anything else makes the encoder fail to open
AAC_RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)
LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1"}
FIT_MODES = ["pad", "stretch", "crop"]
AUDIO_MODES = ["music", "clips", "none"]
MAX_RATE = 120  # a clip that mis-reports its frame rate must not set the grid for the film


class VideoError(RuntimeError):
    pass


def _even(value: float) -> int:
    """H.264 with yuv420p rejects odd dimensions."""
    return max(2, int(value) - (int(value) % 2))


def _snap_rate(rate: int) -> int:
    return rate if rate in AAC_RATES else min(AAC_RATES, key=lambda candidate: abs(candidate - rate))


def _probe(path: str) -> dict:
    """Metadata of one clip, in display geometry. Raises on a truncated download."""
    import av

    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise VideoError(f"{PREFIX} no video stream in {path}")
        stream = container.streams.video[0]
        width, height = int(stream.width), int(stream.height)
        rotation = 0
        try:  # the rotation lives on the frame, not on the stream
            for frame in container.decode(stream):
                rotation = int(round((frame.rotation or 0) / 90)) % 4
                break
        except Exception:
            rotation = 0
        if rotation % 2:
            width, height = height, width
        rate = Fraction(stream.average_rate) if stream.average_rate else None
        if rate is not None and (rate <= 0 or rate > MAX_RATE):
            rate = None
        return {
            "path": path,
            "width": width,
            "height": height,
            "rotation": rotation,
            "rate": rate,
            "has_audio": bool(container.streams.audio),
        }


def _target_geometry(infos: list[dict], width: int | None, height: int | None) -> tuple[int, int]:
    if width and height:
        return _even(width), _even(height)
    sizes = Counter((info["width"], info["height"]) for info in infos)
    best_count = max(sizes.values())
    # ties go to the largest frame, so one odd clip never shrinks the film
    candidates = [size for size, count in sizes.items() if count == best_count]
    chosen = max(candidates, key=lambda size: size[0] * size[1])
    if len({(info["width"] / max(1, info["height"])) for info in infos}) > 1:
        warn("clips have different aspect ratios; the odd ones are letterboxed")
    return _even(width or chosen[0]), _even(height or chosen[1])


def _target_rate(infos: list[dict], fps: float | None) -> Fraction:
    if fps and fps > 0:
        # snap the rates a user types to their broadcast rationals
        for numerator, denominator in ((24000, 1001), (30000, 1001), (60000, 1001)):
            if abs(fps - numerator / denominator) < 0.01:
                return Fraction(numerator, denominator)
        return Fraction(fps).limit_denominator(1000)
    rates = [info["rate"] for info in infos if info["rate"]]
    return max(rates) if rates else Fraction(24)


def _oriented(frame, rotation: int):
    """Apply the clip's display matrix so a portrait clip really is portrait."""
    if not rotation:
        return frame
    import av
    import numpy as np

    array = np.rot90(frame.to_ndarray(format="rgb24"), k=rotation, axes=(0, 1)).copy()
    return av.VideoFrame.from_ndarray(array, format="rgb24")


def _place(frame, width: int, height: int, fit: str, interpolation: str):
    """Resize one frame onto the output grid, letterboxing unless told otherwise."""
    import av
    import numpy as np

    kwargs: dict[str, Any] = {"interpolation": interpolation}
    if frame.color_range == 2:  # ColorRange.JPEG - full-range source, else it washes out
        kwargs.update(src_color_range="JPEG", dst_color_range="MPEG")

    source_aspect = frame.width / max(1, frame.height)
    target_aspect = width / max(1, height)
    if fit == "stretch" or abs(source_aspect - target_aspect) < 0.005:
        return frame.reformat(width=width, height=height, format="yuv420p", **kwargs)

    if fit == "crop":
        scale = max(width / frame.width, height / frame.height)
    else:  # pad
        scale = min(width / frame.width, height / frame.height)
    inner_w, inner_h = _even(frame.width * scale), _even(frame.height * scale)
    resized = frame.reformat(width=inner_w, height=inner_h, format="rgb24", **kwargs)
    array = resized.to_ndarray(format="rgb24")

    if fit == "crop":
        left = max(0, (inner_w - width) // 2)
        top = max(0, (inner_h - height) // 2)
        canvas = array[top : top + height, left : left + width]
        if canvas.shape[0] != height or canvas.shape[1] != width:  # rounding guard
            padded = np.zeros((height, width, 3), dtype=array.dtype)
            padded[: canvas.shape[0], : canvas.shape[1]] = canvas
            canvas = padded
    else:
        canvas = np.zeros((height, width, 3), dtype=array.dtype)
        left = max(0, (width - inner_w) // 2)
        top = max(0, (height - inner_h) // 2)
        canvas[top : top + inner_h, left : left + inner_w] = array[
            : height - top, : width - left
        ]
    placed = av.VideoFrame.from_ndarray(np.ascontiguousarray(canvas), format="rgb24")
    return placed.reformat(format="yuv420p")


def _audio_planes(audio: dict):
    """ComfyUI AUDIO dict -> (planar float32 (channels, samples), rate, layout)."""
    import numpy as np

    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().float().numpy()
    array = np.asarray(waveform, dtype="float32")
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    channels = int(array.shape[0])
    if channels not in LAYOUTS:
        array = array[:2] if channels >= 2 else np.repeat(array, 2, axis=0)
    layout = LAYOUTS.get(int(array.shape[0]), "stereo")
    return np.ascontiguousarray(array), int(audio.get("sample_rate") or 44100), layout


def concat_clips(  # noqa: PLR0912, PLR0915 - one linear muxing pass, kept in one place
    clip_paths: Sequence[str],
    output_path: str,
    *,
    audio: dict | None = None,
    audio_mode: str = "music",
    clip_durations: Sequence[float] | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    fit: str = "pad",
    crf: int = 20,
    preset: str = "medium",
    interpolation: str = "BICUBIC",  # uppercase; "bicubic" raises KeyError
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Concatenate ``clip_paths`` into ``output_path``. Returns what was written."""
    import av
    import numpy as np

    paths = [path for path in clip_paths if path and os.path.exists(path)]
    if not paths:
        raise VideoError(f"{PREFIX} nothing to concatenate - no clip files exist")

    infos = [_probe(path) for path in paths]
    width, height = _target_geometry(infos, width, height)
    rate = _target_rate(infos, fps)
    step = Fraction(1) / rate
    time_base = Fraction(1) / rate  # exact for every rational rate, including 1001-based

    mode = (audio_mode or "music").lower()
    if mode not in AUDIO_MODES:
        mode = "music"
    if mode == "music" and audio is None:
        warn("no music track supplied; keeping the clips' own audio instead")
        mode = "clips"
    if mode == "clips" and not any(info["has_audio"] for info in infos):
        mode = "none"

    log(
        f"concatenating {len(paths)} clip(s) -> {width}x{height} @ {float(rate):.3f} fps, "
        f"audio: {mode}"
    )

    container = av.open(output_path, mode="w", format="mp4", options={"movflags": "faststart"})
    video = container.add_stream("libx264", rate=rate)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    video.codec_context.time_base = time_base
    video.codec_context.max_b_frames = 0  # keeps mp4 frame durations in display order
    video.options = {"crf": str(int(crf)), "preset": preset}

    audio_stream = None
    audio_rate = 48000
    layout = "stereo"
    planes = None
    source_rate = 48000
    if mode == "music":
        planes, source_rate, layout = _audio_planes(audio or {})
        audio_rate = _snap_rate(source_rate)
        audio_stream = container.add_stream("aac", rate=audio_rate, layout=layout)
    elif mode == "clips":
        audio_stream = container.add_stream("aac", rate=audio_rate, layout=layout)

    frames_per_clip: list[int] = []
    written = 0
    try:
        for index, info in enumerate(infos):
            target = None
            if clip_durations is not None and index < len(clip_durations):
                target = max(1, int(round(float(clip_durations[index]) * float(rate))))
            emitted = _write_clip(
                container, video, info, width, height, fit, interpolation, rate, step,
                time_base, written, target,
            )
            frames_per_clip.append(emitted)
            written += emitted
            if progress_cb:
                progress_cb(index + 1, len(infos))

        for packet in video.encode(None):
            container.mux(packet)
        duration = float(written * step)

        if audio_stream is not None and mode == "music" and planes is not None:
            _write_music(container, audio_stream, planes, source_rate, audio_rate, layout, duration)
        elif audio_stream is not None and mode == "clips":
            _write_clip_audio(
                container, audio_stream, infos, frames_per_clip, audio_rate, layout, float(step)
            )
        if audio_stream is not None:
            for packet in audio_stream.encode(None):
                container.mux(packet)
    finally:
        container.close()

    log(f"wrote {output_path}: {written} frames, {duration:.2f}s")
    return {
        "path": output_path,
        "frames": written,
        "duration": duration,
        "width": width,
        "height": height,
        "fps": float(rate),
        "frames_per_clip": frames_per_clip,
        "audio": mode,
    }


def _write_clip(
    container, video, info: dict, width: int, height: int, fit: str, interpolation: str,
    rate: Fraction, step: Fraction, time_base: Fraction, offset: int, target: int | None,
) -> int:
    """Re-time one clip onto the output grid. Returns the number of frames emitted."""
    import av

    emitted = 0
    last_placed = None
    with av.open(info["path"], mode="r") as source:
        stream = source.streams.video[0]
        stream.thread_type = "AUTO"
        source_rate = Fraction(stream.average_rate) if stream.average_rate else rate
        if source_rate <= 0 or source_rate > MAX_RATE:
            source_rate = rate
        next_slot = Fraction(0)
        first_pts = None
        for packet in source.demux(stream):
            try:
                frames = packet.decode()
            except av.error.FFmpegError as exc:
                warn(f"decode error in {os.path.basename(info['path'])}: {exc}")
                continue
            for frame in frames:
                if frame.pts is None:
                    continue
                if first_pts is None:
                    first_pts = frame.pts
                start = Fraction(int(frame.pts - first_pts)) * Fraction(stream.time_base)
                length = (
                    Fraction(int(frame.duration)) * Fraction(stream.time_base)
                    if frame.duration
                    else Fraction(1) / source_rate
                )
                placed = None
                # one output frame per grid slot this source frame covers:
                # 24->30 duplicates, 30->24 drops, variable frame rate handled the same way
                while next_slot < start + length:
                    if target is not None and emitted >= target:
                        break
                    if placed is None:  # convert once, reuse for duplicated slots
                        placed = _place(_oriented(frame, info["rotation"]), width, height, fit, interpolation)
                        last_placed = placed
                    _mux_frame(container, video, placed, offset + emitted, time_base)
                    emitted += 1
                    next_slot += step
                if target is not None and emitted >= target:
                    break
            if target is not None and emitted >= target:
                break

    if last_placed is None:
        raise VideoError(f"{PREFIX} no decodable frames in {info['path']}")
    # a clip that came back shorter than its shot holds its last frame, so the film
    # stays in sync with the music instead of drifting earlier with every shot
    while target is not None and emitted < target:
        _mux_frame(container, video, last_placed, offset + emitted, time_base)
        emitted += 1
    return emitted


def _mux_frame(container, video, frame, index: int, time_base: Fraction) -> None:
    frame.pts = index
    frame.time_base = time_base
    frame.pict_type = 0  # let x264 choose frame types instead of copying the source's
    for packet in video.encode(frame):
        container.mux(packet)


def _write_music(container, stream, planes, source_rate: int, audio_rate: int, layout: str, duration: float) -> None:
    """Mux the track, trimmed or padded to the length of the film."""
    import av
    import numpy as np

    need = int(math.ceil(duration * source_rate))
    if planes.shape[1] < need:
        planes = np.pad(planes, ((0, 0), (0, need - planes.shape[1])))
    else:
        planes = planes[:, :need]

    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(planes), format="fltp", layout=layout)
    frame.sample_rate = source_rate
    frame.pts = 0
    frame.time_base = Fraction(1, source_rate)

    written = 0
    resampler = (
        av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=audio_rate)
        if source_rate != audio_rate
        else None
    )
    batch = resampler.resample(frame) if resampler else [frame]
    if resampler:
        batch = list(batch) + list(resampler.resample(None))
    for part in batch:  # the encoder re-cuts these into 1024-sample AAC frames itself
        part.pts = written
        part.time_base = Fraction(1, audio_rate)
        written += part.samples
        for packet in stream.encode(part):
            container.mux(packet)


def _write_clip_audio(
    container, stream, infos: list[dict], frames_per_clip: list[int], audio_rate: int,
    layout: str, step: float,
) -> None:
    """Keep the audio the video model generated, padded so the cuts stay aligned."""
    import av
    import numpy as np

    channels = 1 if layout == "mono" else 2
    written = 0
    elapsed = 0.0
    for info, frames in zip(infos, frames_per_clip):
        elapsed += frames * step
        target = int(round(elapsed * audio_rate))
        if info["has_audio"]:
            chunks = []
            with av.open(info["path"], mode="r") as source:
                track = source.streams.audio[0]
                resampler = av.audio.resampler.AudioResampler(
                    format="fltp", layout=layout, rate=audio_rate
                )
                for packet in source.demux(track):
                    try:
                        decoded = packet.decode()
                    except av.error.FFmpegError:
                        continue
                    for part in decoded:
                        chunks.extend(piece.to_ndarray() for piece in resampler.resample(part))
                chunks.extend(piece.to_ndarray() for piece in resampler.resample(None))
            if chunks:
                segment = np.concatenate(chunks, axis=1)
                keep = max(0, min(segment.shape[1], target - written))
                if keep:
                    written += _mux_audio(container, stream, segment[:, :keep], audio_rate, layout, written)
        if written < target:  # a silent clip must still take its slot
            silence = np.zeros((channels, target - written), dtype="float32")
            written += _mux_audio(container, stream, silence, audio_rate, layout, written)


def _mux_audio(container, stream, planes, audio_rate: int, layout: str, offset: int) -> int:
    import av
    import numpy as np

    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(planes), format="fltp", layout=layout)
    frame.sample_rate = audio_rate
    frame.pts = offset
    frame.time_base = Fraction(1, audio_rate)
    for packet in stream.encode(frame):
        container.mux(packet)
    return int(planes.shape[1])
