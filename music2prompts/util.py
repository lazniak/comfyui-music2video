"""Small helpers shared across the Music2Prompts node.

Heavy third-party imports (numpy, torch, librosa) stay inside functions so that
importing this package never breaks ComfyUI startup and so the pure-python
modules can be unit tested without them.
"""

from __future__ import annotations

import base64
import io as _io
import json
import logging
import re
from typing import Any, Iterable, Iterator, Sequence

LOGGER = logging.getLogger("music2prompts")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)

PREFIX = "[Music2Prompts]"


def log(message: str) -> None:
    LOGGER.info("%s %s", PREFIX, message)


def warn(message: str) -> None:
    LOGGER.warning("%s %s", PREFIX, message)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def chunked(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """Yield consecutive chunks of at most ``size`` elements."""
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# --------------------------------------------------------------------------- audio


def audio_to_mono(audio: dict, target_sr: int | None = None) -> tuple[Any, int]:
    """Convert a ComfyUI AUDIO dict to a mono float32 numpy array.

    ``audio`` is ``{"waveform": torch.Tensor[B, C, N], "sample_rate": int}``.
    Returns ``(samples, sample_rate)``.
    """
    import numpy as np

    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError(f"{PREFIX} expected a ComfyUI AUDIO input (dict with 'waveform').")

    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate") or 44100)

    array = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
    array = np.asarray(array, dtype=np.float32)

    while array.ndim > 2:  # [B, C, N] -> [C, N]
        array = array[0]
    if array.ndim == 2:  # [C, N] -> [N]
        array = array.mean(axis=0)
    array = np.ascontiguousarray(array, dtype=np.float32)

    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak

    if target_sr and target_sr != sample_rate:
        array = resample(array, sample_rate, target_sr)
        sample_rate = target_sr
    return array, sample_rate


def resample(samples: Any, source_sr: int, target_sr: int) -> Any:
    """Resample mono audio: librosa -> scipy polyphase -> linear interpolation."""
    import numpy as np

    if source_sr == target_sr:
        return samples
    audio = np.asarray(samples, dtype=np.float32)

    try:
        import librosa

        return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"librosa.resample failed ({exc}); using scipy instead.")

    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(source_sr), int(target_sr))
        return resample_poly(audio, int(target_sr) // divisor, int(source_sr) // divisor).astype("float32")
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"scipy resampling failed ({exc}); using linear interpolation.")

    duration = len(audio) / float(source_sr)
    target_len = max(1, int(round(duration * target_sr)))
    source_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype("float32")


def image_tensor_to_data_uri(image: Any, index: int = 0, max_side: int = 768) -> str | None:
    """Convert one image of a ComfyUI IMAGE batch ([B, H, W, C], 0..1) to a PNG data URI."""
    try:
        import numpy as np
        from PIL import Image

        array = image[index].detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image[index])
        array = np.clip(array * 255.0, 0, 255).astype("uint8")
        pil = Image.fromarray(array)
        if max(pil.size) > max_side:
            scale = max_side / float(max(pil.size))
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))))
        buffer = _io.BytesIO()
        pil.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as exc:
        warn(f"could not encode reference image #{index}: {exc}")
        return None


# --------------------------------------------------------------------------- json


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON out of a model reply that may contain fences or prose."""
    if text is None:
        raise ValueError("empty model reply")
    candidate = text.strip()
    if not candidate:
        raise ValueError("empty model reply")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            candidate = fenced.group(1).strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for pos in range(start, len(candidate)):
            char = candidate[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : pos + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("model reply did not contain valid JSON")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def first_str(mapping: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def join_sentences(parts: Iterable[str]) -> str:
    """Join sentence fragments into one paragraph with single spaces."""
    cleaned = [re.sub(r"\s+", " ", str(part).strip()) for part in parts if part and str(part).strip()]
    out = " ".join(cleaned)
    return re.sub(r"\s+([,.;:!?])", r"\1", out).strip()


def sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "--", repo_id.strip())
