"""Tests for the Whisper wrapper's pure helpers (no torch, no model download)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.asr import (  # noqa: E402
    _attempt_ladder,
    _guess_language,
    _is_out_of_memory,
    _language_of,
    _normalize,
    _shift,
    model_directory,
    resolve_device,
)


def test_ladder_steps_down_batch_then_timestamps_then_cpu():
    ladder = _attempt_ladder(batch_size=4, device="cuda:0", dtype="float16", word_timestamps=True)
    assert ladder[0] == ("cuda:0", "float16", 4, "word")
    assert ladder[1] == ("cuda:0", "float16", 1, "word")
    assert ladder[2] == ("cuda:0", "float16", 1, True)
    assert ladder[-1] == ("cpu", "float32", 1, "word")


def test_ladder_without_word_timestamps_skips_that_rung():
    ladder = _attempt_ladder(batch_size=1, device="cuda:0", dtype="float16", word_timestamps=False)
    assert all(step[3] != "word" for step in ladder)
    assert ladder[0] == ("cuda:0", "float16", 1, True)


def test_ladder_on_cpu_has_no_cpu_duplicate():
    ladder = _attempt_ladder(batch_size=1, device="cpu", dtype="float32", word_timestamps=True)
    assert [step[0] for step in ladder] == ["cpu"] * len(ladder)


def test_out_of_memory_detection():
    assert _is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 38.00 MiB"))
    assert not _is_out_of_memory(ValueError("bad dict"))


def test_shift_moves_window_timestamps_into_track_time():
    result = {"text": "hi", "chunks": [{"timestamp": (0.5, 1.5), "text": "hi"}]}
    shifted = _shift(result, 30.0)
    assert shifted["chunks"][0]["timestamp"] == (30.5, 31.5)


def test_shift_is_a_noop_for_the_first_window():
    result = {"chunks": [{"timestamp": (0.5, 1.5), "text": "hi"}]}
    assert _shift(result, 0.0)["chunks"][0]["timestamp"] == (0.5, 1.5)


def test_normalize_extracts_words():
    raw = {
        "text": " Na chłodnej ziemi ",
        "chunks": [
            {"timestamp": (0.0, 1.14), "text": " Na"},
            {"timestamp": (1.14, 1.56), "text": " chłodnej"},
            {"timestamp": (1.56, None), "text": " ziemi"},
        ],
    }
    out = _normalize(raw, "auto")
    assert out["text"] == "Na chłodnej ziemi"
    assert out["language"] == "Polish"
    assert [word["text"] for word in out["words"]] == ["Na", "chłodnej", "ziemi"]
    assert out["words"][0] == {"start": 0.0, "end": 1.14, "text": "Na"}


def test_explicit_language_wins_over_the_heuristic():
    assert _language_of("Na chłodnej ziemi", "en") == "en"
    assert _language_of("Na chłodnej ziemi", "auto") == "Polish"
    assert _guess_language("just a plain english line") == "English"


def test_resolve_device_forces_float32_on_cpu():
    assert resolve_device("cpu") == ("cpu", "float32")


def test_model_directory_is_sanitised():
    assert model_directory("openai/whisper-large-v3").endswith("openai--whisper-large-v3")


# --------------------------------------------------------------------------- what reaches the pipeline


class FakePipe:
    """Records the call. transformers' own pipeline is a model download away."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

        class Model:
            @staticmethod
            def generate(input_features=None, stopping_criteria=None, **kwargs):
                return None

        self.model = Model()

    def __call__(self, audio, **kwargs):
        self.calls.append(kwargs)
        return {"text": "hello", "chunks": []}


def slice_with(monkeypatch, seconds: float, chunk_length_s: int = 30, batch_size: int = 4) -> dict:
    from music2prompts import asr as asr_module

    pipe = FakePipe()
    monkeypatch.setattr(asr_module, "_build_pipeline", lambda *a, **k: pipe)
    context = {
        "model_path": "m",
        "device": "cpu",
        "dtype": "float32",
        "chunk_length_s": chunk_length_s,
        "batch_size": batch_size,
        "word_timestamps": True,
        "keep_loaded": False,
        "generate_kwargs": {"task": "transcribe"},
        "sample_rate": 16000,
    }
    asr_module._transcribe_slice([0.0] * int(seconds * 16000), 0.0, context)
    return pipe.calls[0]


def test_a_window_that_fits_goes_to_whisper_whole(monkeypatch):
    """transformers: chunking a seq2seq model is experimental and approximate at the seams.

    Below one chunk it also does nothing at all - one chunk in, one chunk out - so passing
    it only cost us Whisper's own long-form path and printed a warning about it.
    """
    call = slice_with(monkeypatch, seconds=30)
    assert "chunk_length_s" not in call
    assert "batch_size" not in call
    assert call["return_timestamps"] == "word"


def test_a_few_seconds_over_a_chunk_is_still_handed_over_whole(monkeypatch):
    """The last window of a track runs past the mark; splitting it there reads worse."""
    assert "chunk_length_s" not in slice_with(monkeypatch, seconds=34)


def test_a_slice_far_longer_than_a_chunk_is_chunked_after_all(monkeypatch):
    """whisper_window_seconds = 0 hands the whole track over; chunking is the memory guard."""
    call = slice_with(monkeypatch, seconds=180)
    assert call["chunk_length_s"] == 30
    assert call["batch_size"] == 4
    assert call["ignore_warning"] is True, "the caveat is reported once, in our own words"


def test_batching_is_dropped_where_there_is_nothing_to_batch(monkeypatch):
    """Batch size only exists across chunks, so the ladder must not retry it as a rung."""
    from music2prompts import asr as asr_module

    ladder = _attempt_ladder(batch_size=1, device="cpu", dtype="float32", word_timestamps=True)
    assert all(step[2] == 1 for step in ladder)
    assert "batch_size" not in slice_with(monkeypatch, seconds=30, batch_size=4)
    assert asr_module.WINDOW_SLACK == 5.0
