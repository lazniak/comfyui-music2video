"""Deterministic music analysis: tempo, beat grid, sections, energy.

Two backends:

* **librosa** when it is installed (best quality),
* a **numpy/scipy fallback** otherwise, so the node works on a stock ComfyUI
  install without adding dependencies.

The language model is never asked to "listen" - it receives these measured
numbers as facts. Every step degrades to an empty result instead of raising.
"""

from __future__ import annotations

from typing import Any

from .util import log, warn

_ANALYSIS_SR = 22050
_HOP = 512
_WINDOW = 2048
_MIN_BPM = 60.0
_MAX_BPM = 190.0


def empty_analysis(duration: float) -> dict:
    return {
        "duration": round(float(duration), 3),
        "bpm": 0.0,
        "beats": [],
        "sections": [{"name": "Part 1", "start": 0.0, "end": round(float(duration), 3), "energy": 0.5}],
        "energy_curve": [],
        "backend": "none",
    }


def analyze(samples: Any, sample_rate: int, enabled: bool = True) -> dict:
    duration = float(len(samples)) / float(max(1, sample_rate))
    if not enabled:
        return empty_analysis(duration)

    try:
        import numpy as np  # noqa: F401
    except Exception as exc:  # pragma: no cover - numpy is always present in ComfyUI
        warn(f"numpy unavailable ({exc}); skipping music analysis.")
        return empty_analysis(duration)

    try:
        import librosa  # noqa: F401

        result = _analyze_librosa(samples, sample_rate, duration)
    except ImportError:
        result = _analyze_numpy(samples, sample_rate, duration)
    except Exception as exc:
        warn(f"librosa analysis failed ({exc}); falling back to the numpy backend.")
        result = _analyze_numpy(samples, sample_rate, duration)

    log(
        f"music analysis [{result['backend']}]: {result['bpm']:.1f} BPM, {len(result['beats'])} beats, "
        f"{len(result['sections'])} sections, {duration:.1f}s"
    )
    return result


# --------------------------------------------------------------------------- librosa backend


def _analyze_librosa(samples: Any, sample_rate: int, duration: float) -> dict:
    import librosa
    import numpy as np

    result = empty_analysis(duration)
    result["backend"] = "librosa"
    audio = np.asarray(samples, dtype="float32")

    try:
        tempo, beat_times = librosa.beat.beat_track(y=audio, sr=sample_rate, units="time")
        result["bpm"] = round(float(np.atleast_1d(tempo)[0]), 2)
        result["beats"] = [round(float(t), 3) for t in beat_times]
    except Exception as exc:
        warn(f"beat tracking failed: {exc}")

    try:
        rms = librosa.feature.rms(y=audio, hop_length=_HOP)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sample_rate, hop_length=_HOP)
        result["energy_curve"] = _energy_points(rms, times, duration)
    except Exception as exc:
        warn(f"energy curve failed: {exc}")

    try:
        features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        count = _section_count(duration)
        frames = librosa.segment.agglomerative(features, count)
        times = [float(t) for t in librosa.frames_to_time(frames, sr=sample_rate)]
        result["sections"] = _sections_from_boundaries(times, duration, result["energy_curve"])
    except Exception as exc:
        warn(f"section detection failed: {exc}")
    return result


# --------------------------------------------------------------------------- numpy backend


def _analyze_numpy(samples: Any, sample_rate: int, duration: float) -> dict:
    import numpy as np

    result = empty_analysis(duration)
    result["backend"] = "numpy"

    audio, sr = _downmix(np.asarray(samples, dtype="float32"), sample_rate)
    spectrogram, frame_times = _spectrogram(audio, sr)
    if spectrogram.size == 0:
        return result

    onset = _onset_envelope(spectrogram)
    bpm, beats = _tempo_and_beats(onset, frame_times, duration)
    result["bpm"] = round(float(bpm), 2)
    result["beats"] = [round(float(t), 3) for t in beats]

    rms = np.sqrt(np.mean(spectrogram**2, axis=0) + 1e-12)
    result["energy_curve"] = _energy_points(rms, frame_times, duration)
    boundaries = _novelty_boundaries(spectrogram, frame_times, duration)
    result["sections"] = _sections_from_boundaries(boundaries, duration, result["energy_curve"])
    return result


def _downmix(audio: Any, sample_rate: int) -> tuple[Any, int]:
    import numpy as np

    if sample_rate <= _ANALYSIS_SR:
        return audio, sample_rate
    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(sample_rate), _ANALYSIS_SR)
        return (
            resample_poly(audio, _ANALYSIS_SR // divisor, int(sample_rate) // divisor).astype("float32"),
            _ANALYSIS_SR,
        )
    except Exception:
        step = max(1, int(round(sample_rate / _ANALYSIS_SR)))
        return np.ascontiguousarray(audio[::step]), int(sample_rate / step)


def _spectrogram(audio: Any, sample_rate: int) -> tuple[Any, Any]:
    import numpy as np

    if len(audio) < _WINDOW:
        return np.zeros((0, 0), dtype="float32"), np.zeros((0,), dtype="float32")
    frame_count = 1 + (len(audio) - _WINDOW) // _HOP
    window = np.hanning(_WINDOW).astype("float32")
    indices = np.arange(_WINDOW)[None, :] + _HOP * np.arange(frame_count)[:, None]
    frames = audio[indices] * window
    spectrum = np.abs(np.fft.rfft(frames, axis=1)).T.astype("float32")
    spectrum = np.log1p(spectrum)
    times = (np.arange(frame_count) * _HOP + _WINDOW / 2.0) / float(sample_rate)
    return spectrum, times.astype("float32")


def _onset_envelope(spectrogram: Any) -> Any:
    import numpy as np

    flux = np.diff(spectrogram, axis=1, prepend=spectrogram[:, :1])
    envelope = np.sum(np.maximum(flux, 0.0), axis=0)
    if envelope.size > 8:  # remove slow drift
        kernel = np.ones(9, dtype="float32") / 9.0
        envelope = envelope - np.convolve(envelope, kernel, mode="same")
    envelope = np.maximum(envelope, 0.0)
    peak = float(np.max(envelope)) or 1.0
    return (envelope / peak).astype("float32")


def _tempo_and_beats(onset: Any, frame_times: Any, duration: float) -> tuple[float, list[float]]:
    import numpy as np

    if onset.size < 8 or frame_times.size < 2:
        return 0.0, []
    frame_rate = 1.0 / float(np.mean(np.diff(frame_times)))
    centered = onset - float(np.mean(onset))
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    # unbiased: without this the longer lags are penalised only by overlap count and
    # the search happily settles on half the real tempo
    overlap = np.maximum(1.0, len(centered) - np.arange(len(correlation)))
    correlation = correlation / overlap

    min_lag = max(1, int(round(frame_rate * 60.0 / _MAX_BPM)))
    max_lag = min(len(correlation) - 1, int(round(frame_rate * 60.0 / _MIN_BPM)))
    if max_lag <= min_lag:
        return 0.0, []

    lag_axis = np.arange(min_lag, max_lag + 1)
    candidate_bpms = 60.0 * frame_rate / lag_axis
    # log-normal prior around 120 BPM, the same idea librosa uses
    prior = np.exp(-0.5 * (np.log2(candidate_bpms / 120.0) / 0.9) ** 2)
    lag = int(lag_axis[int(np.argmax(correlation[min_lag : max_lag + 1] * prior))])

    # octave correction: a periodic signal correlates just as well at twice the period
    half = lag // 2
    if half >= min_lag and correlation[half] > 0.8 * correlation[lag]:
        lag = half

    # sub-frame period: an integer lag drifts by seconds over a full track
    period_frames = float(lag)
    if 0 < lag < len(correlation) - 1:
        left, centre, right = correlation[lag - 1], correlation[lag], correlation[lag + 1]
        denominator = left - 2.0 * centre + right
        if denominator != 0:
            period_frames = lag + float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    period = period_frames / frame_rate
    bpm = 60.0 / period if period > 0 else 0.0

    # best phase: align a pulse train with the onset envelope
    beat_count = int(len(onset) / period_frames)
    best_offset, best_score = 0.0, -1.0
    for offset in range(max(1, int(round(period_frames)))):
        positions = np.round(offset + period_frames * np.arange(beat_count)).astype(int)
        positions = positions[positions < len(onset)]
        score = float(np.sum(onset[positions])) if positions.size else 0.0
        if score > best_score:
            best_score, best_offset = score, float(offset)

    beats: list[float] = []
    search = max(1, int(period_frames // 6))
    for index in range(beat_count + 1):
        centre = int(round(best_offset + period_frames * index))
        if centre >= len(onset):
            break
        low = max(0, centre - search)
        high = min(len(onset), centre + search + 1)
        local = int(np.argmax(onset[low:high])) + low
        time = float(frame_times[local])
        if 0.0 <= time <= duration and (not beats or time - beats[-1] > period * 0.5):
            beats.append(round(time, 3))
    return bpm, beats


def _novelty_boundaries(spectrogram: Any, frame_times: Any, duration: float) -> list[float]:
    import numpy as np

    count = _section_count(duration)
    if spectrogram.shape[1] < 16 or count <= 1:
        return [0.0, duration]

    # coarse feature: log-band energies smoothed over ~2 s
    bands = np.array_split(spectrogram, 12, axis=0)
    features = np.stack([np.mean(band, axis=0) for band in bands])
    smooth = max(1, int(2.0 / max(1e-6, float(np.mean(np.diff(frame_times))))))
    kernel = np.ones(smooth, dtype="float32") / smooth
    features = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, features)

    normalized = features / (np.linalg.norm(features, axis=0, keepdims=True) + 1e-9)
    novelty = 1.0 - np.sum(normalized[:, 1:] * normalized[:, :-1], axis=0)
    novelty = np.concatenate([[0.0], novelty])

    minimum_gap = max(1, int(round(len(novelty) * (8.0 / max(8.0, duration)))))
    candidates = list(np.argsort(novelty)[::-1])
    picked: list[int] = []
    for index in candidates:
        if len(picked) >= count - 1:
            break
        if all(abs(int(index) - other) >= minimum_gap for other in picked):
            picked.append(int(index))
    times = sorted(float(frame_times[index]) for index in picked)
    return [0.0] + [t for t in times if 1.0 < t < duration - 1.0] + [duration]


# --------------------------------------------------------------------------- shared helpers


def _section_count(duration: float) -> int:
    return int(min(12, max(3, round(duration / 20.0))))


def _energy_points(values: Any, times: Any, duration: float) -> list[dict]:
    import numpy as np

    if len(values) == 0:
        return []
    peak = float(np.max(values)) or 1.0
    step = max(1, int(len(values) / max(1, int(duration))))
    return [
        {"t": round(float(times[i]), 2), "e": round(float(values[i] / peak), 3)}
        for i in range(0, len(values), step)
    ]


def _sections_from_boundaries(boundaries: list[float], duration: float, energy_curve: list[dict]) -> list[dict]:
    import numpy as np

    ordered = sorted({0.0} | {float(t) for t in boundaries if 0.0 < float(t) < duration}) + [duration]
    sections: list[dict] = []
    for index in range(len(ordered) - 1):
        start, end = ordered[index], ordered[index + 1]
        if end - start < 4.0 and sections:
            sections[-1]["end"] = round(end, 3)
            continue
        window = [point["e"] for point in energy_curve if start <= point["t"] < end]
        sections.append(
            {
                "name": f"Part {len(sections) + 1}",
                "start": round(start, 3),
                "end": round(end, 3),
                "energy": round(float(np.mean(window)), 3) if window else 0.5,
            }
        )
    if not sections:
        return [{"name": "Part 1", "start": 0.0, "end": round(duration, 3), "energy": 0.5}]
    sections[-1]["end"] = round(duration, 3)
    return sections


def compact_for_llm(analysis: dict, max_beats: int = 48, max_energy_points: int = 40) -> dict:
    """Shrink the analysis so it fits comfortably in a local model's context."""
    beats = analysis.get("beats") or []
    if len(beats) > max_beats:
        stride = max(1, len(beats) // max_beats)
        beats = beats[::stride][:max_beats]
    energy = analysis.get("energy_curve") or []
    if len(energy) > max_energy_points:
        stride = max(1, len(energy) // max_energy_points)
        energy = energy[::stride][:max_energy_points]
    return {
        "duration": analysis.get("duration"),
        "bpm": analysis.get("bpm"),
        "beats_sample": beats,
        "sections": analysis.get("sections"),
        "energy_curve": energy,
    }
