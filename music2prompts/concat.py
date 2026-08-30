"""Cut a list of clips into one film, wherever those clips came from.

The main node assembles what it rendered itself. This one takes VIDEO from anywhere in
the graph - an LTX or Wan subgraph, a sampler, a load-from-disk node - and cuts it
together the same way: one size and one frame rate for the whole film, each clip trimmed
to the shot it belongs to, and the soundtrack laid back underneath.

It reads the *pipe* only for the shot durations. Without a pipe the clips are used at
their own lengths; with one, every cut lands where the beat grid put it - the difference
between a film that drifts out of sync by the last chorus and one that does not.
"""

from __future__ import annotations

import os
import tempfile
import time

from comfy_api.latest import io

from . import pipe as pipe_module
from . import render as render_module
from . import video as video_module
from .util import PREFIX, log, warn

#: what the widget says -> what the muxer calls it
AUDIO_SOURCES = {
    "source audio": "music",
    "mix": "mix",
    "video audio": "clips",
    "silent": "none",
}


def _first(value, default=None):
    """One value out of what an is_input_list node receives (every input is a list)."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _materialise(videos: list) -> tuple[list[str], list[str]]:
    """File paths for each VIDEO, plus the ones written here that need deleting after.

    A VIDEO that is already a file on disk is used where it lies. Anything else - frames
    still in memory, a trimmed view of a longer file - is written out first, because the
    muxer reads clips with PyAV rather than pulling whole decoded videos into RAM.
    """
    paths: list[str] = []
    temporary: list[str] = []
    for index, item in enumerate(videos):
        if item is None:
            continue
        source = None
        try:
            if hasattr(item, "get_stream_source") and _whole(item):
                source = item.get_stream_source()
        except Exception as exc:  # pragma: no cover - defensive
            warn(f"could not read clip {index + 1} directly ({exc}); writing it out instead")
        if isinstance(source, str) and os.path.exists(source):
            paths.append(source)
            continue
        handle, target = tempfile.mkstemp(prefix="m2v_concat_", suffix=".mp4")
        os.close(handle)
        try:
            item.save_to(target)
        except Exception as exc:
            os.unlink(target)
            warn(f"clip {index + 1} could not be written out ({exc}); it is left out of the film")
            continue
        paths.append(target)
        temporary.append(target)
    return paths, temporary


def _whole(item) -> bool:
    """True when the VIDEO is the whole file, not a trimmed window onto it."""
    try:
        start, duration = item.get_active_trim_window()
    except Exception:
        return True  # no trim window to honour
    return start <= 0 and duration <= 0


class Music2VideoConcat(io.ComfyNode):
    """Join a list of clips into one film, with the soundtrack of your choice."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Music2VideoConcat",
            display_name="🎵 Music2Video Concat",
            category="Music2Video",
            description=(
                "Cuts a list of clips into one film. Every clip is placed on a single grid - "
                "one size, one frame rate - and the soundtrack is laid underneath: the track "
                "you wired in, the audio the video models generated, both mixed, or none.\n\n"
                "Wire the 'pipe' from the Music2Video node and each clip is trimmed to the "
                "shot it was written for, with the cut boundaries pinned to within half a "
                "frame of the beat grid; without it the clips are used at whatever length "
                "they came back at. Nothing is generated here and nothing is billed - it is "
                "one muxing pass with PyAV, no ffmpeg binary needed."
            ),
            is_input_list=True,  # the clips arrive together, not one execution per clip
            inputs=[
                io.Video.Input(
                    "videos",
                    tooltip=(
                        "The clips, in the order they should appear. Wire the 'videos' output "
                        "of the Music2Video node, or the VIDEO output of whatever rendered "
                        "them - this node takes a whole list at once rather than running per "
                        "clip. A clip that is already a file on disk is read where it lies; "
                        "anything still in memory is written to a temporary file first and "
                        "removed afterwards."
                    ),
                ),
                io.Custom(pipe_module.PIPE_TYPE).Input(
                    "pipe",
                    optional=True,
                    tooltip=(
                        "The pipe from the Music2Video node, read for one thing only: the "
                        "per-shot durations. Each clip is then trimmed, or held on its last "
                        "frame, to last exactly as long as its shot, and every cut lands "
                        "within half a frame of where the beat grid put it. Leave it "
                        "unwired and the clips keep their own lengths - fine for a rough "
                        "assembly, but the film drifts out of sync with the track as the "
                        "errors accumulate."
                    ),
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "The source track, normally the same AUDIO you fed the Music2Video "
                        "node. Used when 'audio_mode' is 'source audio' or 'mix'; it is "
                        "trimmed or padded with silence to the length of the finished film. "
                        "Without it those two modes fall back to the clips' own audio and say "
                        "so in the log."
                    ),
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=list(AUDIO_SOURCES),
                    default="source audio",
                    tooltip=(
                        "What the film's soundtrack is.\n"
                        "'source audio' - the track from the 'audio' input, and nothing else. "
                        "The usual choice for a music video.\n"
                        "'mix' - the track and the clips' own audio summed, balanced by "
                        "'music_gain' and 'clip_gain'. Use it when the clips carry dialogue "
                        "or effects worth keeping under the music.\n"
                        "'video audio' - only what the video models generated, each clip's "
                        "audio in its own slot, silence where a clip has none.\n"
                        "'silent' - no audio track at all."
                    ),
                ),
                io.Float.Input(
                    "music_gain",
                    default=1.0,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "How loud the source track sits in the mix, as a linear multiplier - "
                        "1.0 is untouched, 0.5 is half amplitude (about -6 dB). Only used by "
                        "'mix'. The sum of both is clipped to full scale, so pushing both "
                        "above 1.0 buys distortion, not loudness."
                    ),
                ),
                io.Float.Input(
                    "clip_gain",
                    default=0.5,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "How loud the clips' own audio sits in the mix, as a linear "
                        "multiplier. The default of 0.5 puts generated dialogue and effects "
                        "under the music rather than over it. Only used by 'mix'."
                    ),
                ),
                io.Combo.Input(
                    "fit",
                    options=list(video_module.FIT_MODES),
                    default="pad",
                    tooltip=(
                        "What happens to a clip whose aspect ratio differs from the film's. "
                        "'pad' letterboxes it, keeping the whole frame; 'crop' fills the frame "
                        "and loses the edges; 'stretch' distorts it to fit. The film's own "
                        "size is the most common size among the clips unless you override it."
                    ),
                ),
                io.Float.Input(
                    "fps",
                    default=0.0,
                    min=0.0,
                    max=120.0,
                    step=1.0,
                    tooltip=(
                        "Frame rate of the finished film. 0 takes it from the clips (the "
                        "highest sane rate among them), which is what you want unless a "
                        "delivery spec says otherwise. Every clip is resampled onto this one "
                        "grid, so a mixed-rate set of clips still cuts cleanly."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=8192,
                    step=8,
                    tooltip=(
                        "Width of the finished film, or 0 to take it from the clips. Set both "
                        "this and 'height' to force a delivery size; odd numbers are rounded "
                        "up, because H.264 cannot encode them."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=8192,
                    step=8,
                    tooltip="Height of the finished film, or 0 to take it from the clips.",
                ),
                io.Int.Input(
                    "crf",
                    default=20,
                    min=0,
                    max=51,
                    tooltip=(
                        "H.264 quality: lower is better and larger. 18 is visually lossless "
                        "for most material, 20 is a good delivery default, above 28 shows. "
                        "This is the only re-encode the clips get here."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="music2video",
                    tooltip=(
                        "Start of the filename. The film is written to "
                        "ComfyUI/output/music2prompts as "
                        "<prefix>_<date>-<time>_concat.mp4, so a re-run never overwrites the "
                        "last one."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The finished film, ready for SaveVideo or a preview. It is already "
                        "written to disk - saving it again only copies it somewhere else."
                    ),
                ),
                io.String.Output(
                    display_name="path",
                    tooltip="Where the file was written, absolute.",
                ),
                io.Float.Output(
                    display_name="duration",
                    tooltip="Length of the finished film in seconds.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        videos=None,
        pipe=None,
        audio=None,
        audio_mode="source audio",
        music_gain=1.0,
        clip_gain=0.5,
        fit="pad",
        fps=0.0,
        width=0,
        height=0,
        crf=20,
        filename_prefix="music2video",
    ) -> io.NodeOutput:
        # is_input_list hands every input in as a list, widgets included
        clips = [item for item in (videos or []) if item is not None]
        pipe = _first(pipe)
        audio = _first(audio)
        mode = AUDIO_SOURCES.get(str(_first(audio_mode, "source audio")), "music")
        prefix = str(_first(filename_prefix, "music2video") or "music2video")

        if not clips:
            raise ValueError(
                f"{PREFIX} no clips to join. Wire the 'videos' output of the Music2Video node, "
                "or any VIDEO list, into 'videos'."
            )

        durations = None
        if isinstance(pipe, dict):
            durations = [float(value) for value in (pipe.get("durations") or [])]
            if len(durations) != len(clips):
                warn(
                    f"the pipe describes {len(durations)} shot(s) but {len(clips)} clip(s) came "
                    "in - the clips are used at their own lengths. This happens when a render "
                    "failed: the failed shot leaves no clip, so the two lists no longer line up."
                )
                durations = None
        if durations is None and pipe is not None and not isinstance(pipe, dict):
            warn("'pipe' is not a Music2Video pipe; the shot durations are ignored")

        paths, temporary = _materialise(clips)
        if not paths:
            raise ValueError(f"{PREFIX} none of the {len(clips)} clip(s) could be read as a file.")

        target = os.path.join(
            render_module.output_directory(),
            f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_concat.mp4",
        )
        log(f"joining {len(paths)} clip(s), audio: {mode}")
        try:
            info = video_module.concat_clips(
                paths,
                target,
                audio=audio,
                audio_mode=mode,
                clip_durations=durations,
                width=int(_first(width, 0)) or None,
                height=int(_first(height, 0)) or None,
                fps=float(_first(fps, 0.0)) or None,
                fit=str(_first(fit, "pad")),
                music_gain=float(_first(music_gain, 1.0)),
                clip_gain=float(_first(clip_gain, 0.5)),
                crf=int(_first(crf, 20)),
            )
        finally:
            for path in temporary:
                try:
                    os.unlink(path)
                except OSError:  # pragma: no cover - the film is written either way
                    pass

        video = cls._as_video(info["path"])
        return io.NodeOutput(video, info["path"], float(info["duration"]))

    @staticmethod
    def _as_video(path: str):
        try:
            from comfy_api.input_impl import VideoFromFile  # type: ignore
        except Exception as exc:  # pragma: no cover - older ComfyUI
            warn(f"this ComfyUI has no VIDEO type ({exc}); the film is on disk at {path}")
            return None
        return VideoFromFile(path)
