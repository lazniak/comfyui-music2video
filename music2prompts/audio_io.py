"""Encoding a ComfyUI AUDIO dict into bytes a media API will accept.

The node already slices the track into one clip per shot for lip sync; this is what
turns such a slice into a file. MP3 is the wire format: a 6-second stereo clip is about
95 KB as MP3 against 1.0 MB as WAV, and it travels inside the same JSON request body as
the start frame, so eleven times the bytes buy nothing.

PyAV does the encoding - ComfyUI declares it (``av>=17.0.0``) and it is the only audio
library that is reliably importable here; ``torchaudio`` is listed by ComfyUI but is not
actually installed in every environment, and ComfyUI's own SaveAudio cannot return bytes.
"""

from __future__ import annotations

import io

from .util import PREFIX

#: container and codec per format we can write
FORMATS = {
    "mp3": ("mp3", "libmp3lame"),
    "wav": ("wav", "pcm_s16le"),
    "m4a": ("mp4", "aac"),
}

MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4"}


class AudioError(RuntimeError):
    pass


def planes(audio: dict):
    """A ComfyUI AUDIO dict as a contiguous float32 ``(channels, samples)`` array.

    ``AudioFrame.from_ndarray`` is strict: float32 only for ``fltp``, C-contiguous only,
    exactly two dimensions, and channels first. A torch slice satisfies none of those by
    accident, so the conversion is explicit.
    """
    import numpy as np

    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    if waveform is None:
        raise AudioError(f"{PREFIX} no waveform in the audio clip")
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().float().numpy()
    array = np.asarray(waveform, dtype="float32")
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] > 2:  # more than stereo confuses every encoder we target
        array = array[:2]
    if array.size == 0:
        raise AudioError(f"{PREFIX} the audio clip is empty")
    return np.ascontiguousarray(array, dtype="float32")


def encode(audio: dict, kind: str = "mp3", bit_rate: int = 128000) -> bytes:
    """One AUDIO dict as encoded bytes."""
    import av

    if kind not in FORMATS:
        raise AudioError(f"{PREFIX} cannot write '{kind}' audio; try one of {sorted(FORMATS)}")
    array = planes(audio)
    layout = "mono" if array.shape[0] == 1 else "stereo"
    rate = int((audio.get("sample_rate") if isinstance(audio, dict) else 0) or 44100)
    container_format, codec = FORMATS[kind]

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format=container_format)
    try:
        stream = container.add_stream(codec, rate=rate, layout=layout)
        if codec != "pcm_s16le":
            stream.bit_rate = int(bit_rate)
        frame = av.AudioFrame.from_ndarray(array, format="fltp", layout=layout)
        frame.sample_rate = rate
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):  # without the flush the tail is lost
            container.mux(packet)
    finally:
        container.close()  # writes the trailer
    return buffer.getvalue()


def duration(audio: dict) -> float:
    array = planes(audio)
    rate = int((audio.get("sample_rate") if isinstance(audio, dict) else 0) or 44100)
    return array.shape[1] / float(max(1, rate))


def data_uri(audio: dict, kind: str = "mp3") -> str:
    """The clip as a ``data:`` URI, which is what fal accepts inline."""
    import base64

    payload = encode(audio, kind)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{MEDIA_TYPES.get(kind, 'application/octet-stream')};base64,{encoded}"
