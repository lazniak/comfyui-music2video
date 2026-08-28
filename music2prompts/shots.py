"""Shot segmentation: turn a track duration + beat grid into clip boundaries.

Pure python (stdlib only) so it stays unit-testable. The language model never
does timing arithmetic - it only receives the boundaries computed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShotSlot:
    index: int
    start: float
    end: float
    section: str = ""
    lyrics: str = ""
    words: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def target_shot_length(clip_seconds: float, dynamicity: float, min_s: float, max_s: float) -> float:
    """Higher dynamicity -> shorter shots (faster cutting)."""
    factor = 1.3 - 0.6 * _clamp(float(dynamicity), 0.0, 1.0)
    return _clamp(float(clip_seconds) * factor, min_s, max_s)


def _snap(value: float, candidates: list[float], tolerance: float) -> float:
    if not candidates or tolerance <= 0:
        return value
    best = min(candidates, key=lambda c: abs(c - value))
    return best if abs(best - value) <= tolerance else value


def plan_shots(
    duration: float,
    clip_seconds: float = 6.0,
    min_seconds: float = 5.0,
    max_seconds: float = 15.0,
    dynamicity: float = 0.6,
    num_shots: int = 0,
    beats: list[float] | None = None,
    sections: list[dict] | None = None,
    snap_to_beats: bool = True,
) -> list[ShotSlot]:
    """Split ``0..duration`` into consecutive shots with no gaps or overlaps."""
    duration = max(0.1, float(duration))
    min_seconds = max(0.5, float(min_seconds))
    max_seconds = max(min_seconds, float(max_seconds))

    if duration <= min_seconds:
        return [ShotSlot(index=1, start=0.0, end=round(duration, 3), section=_section_at(sections, 0.0))]

    if int(num_shots) > 0:
        count = int(num_shots)
    else:
        target = target_shot_length(clip_seconds, dynamicity, min_seconds, max_seconds)
        count = max(1, round(duration / target))

    # keep every shot inside [min_seconds, max_seconds] where the duration allows it
    count = max(count, 1)
    count = min(count, max(1, int(duration // min_seconds)))
    while count > 1 and duration / count > max_seconds:
        count += 1
    count = max(1, count)

    step = duration / count
    boundaries = [i * step for i in range(count + 1)]
    boundaries[-1] = duration

    if snap_to_beats:
        candidates: list[float] = []
        for section in sections or []:
            for key in ("start", "end"):
                value = section.get(key)
                if isinstance(value, (int, float)):
                    candidates.append(float(value))
        beat_list = [float(b) for b in (beats or []) if 0.0 < float(b) < duration]
        tolerance = min(step * 0.35, max(0.35, step * 0.35))
        for position in range(1, count):
            wanted = boundaries[position]
            snapped = _snap(wanted, candidates, tolerance * 1.2)
            if snapped == wanted:
                snapped = _snap(wanted, beat_list, tolerance)
            lower = boundaries[position - 1] + min_seconds
            upper = duration - (count - position) * min_seconds
            if lower <= snapped <= upper:
                boundaries[position] = snapped

    boundaries = _enforce_monotonic(boundaries, duration, min_seconds, max_seconds)

    shots: list[ShotSlot] = []
    for position in range(len(boundaries) - 1):
        start = round(boundaries[position], 3)
        end = round(boundaries[position + 1], 3)
        if end - start <= 0.05:
            continue
        shots.append(
            ShotSlot(
                index=len(shots) + 1,
                start=start,
                end=end,
                section=_section_at(sections, (start + end) / 2.0),
            )
        )
    if not shots:
        shots = [ShotSlot(index=1, start=0.0, end=round(duration, 3))]
    return shots


def _enforce_monotonic(
    boundaries: list[float], duration: float, min_seconds: float, max_seconds: float
) -> list[float]:
    out = list(boundaries)
    for position in range(1, len(out)):
        lowest = out[position - 1] + min(min_seconds, duration / max(1, len(out) - 1))
        out[position] = max(out[position], lowest)
        out[position] = min(out[position], duration)
    out[-1] = duration
    # split anything that grew past the model's hard maximum
    result: list[float] = [out[0]]
    for position in range(1, len(out)):
        start = result[-1]
        end = out[position]
        span = end - start
        if span > max_seconds:
            pieces = int(span // max_seconds) + 1
            for step in range(1, pieces):
                result.append(round(start + span * step / pieces, 3))
        result.append(end)
    return result


def _section_at(sections: list[dict] | None, time: float) -> str:
    for section in sections or []:
        try:
            if float(section.get("start", 0.0)) <= time < float(section.get("end", 0.0)):
                return str(section.get("name") or "")
        except (TypeError, ValueError):
            continue
    return ""


def attach_lyrics(shots: list[ShotSlot], words: list[dict]) -> list[ShotSlot]:
    """Assign transcribed words to the shot whose range contains their midpoint."""
    for shot in shots:
        shot.words = []
    if not words:
        return shots
    for word in words:
        try:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
        except (TypeError, ValueError):
            continue
        middle = (start + end) / 2.0
        for shot in shots:
            if shot.start <= middle < shot.end:
                shot.words.append(word)
                break
    for shot in shots:
        shot.lyrics = " ".join(str(word.get("text", "")).strip() for word in shot.words).strip()
        shot.lyrics = " ".join(shot.lyrics.split())
    return shots
