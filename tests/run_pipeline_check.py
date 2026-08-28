"""Runs the whole node with a stubbed LM Studio and asserts the plumbing.

Usable without pytest (ComfyUI's python has torch but usually no pytest)::

    set PYTHONPATH=<ComfyUI dir>
    <ComfyUI>/.venv/Scripts/python.exe tests/run_pipeline_check.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_lmstudio import SUBJECTS, FakeClient  # noqa: E402

TRACK_SECONDS = 42.0


def make_audio(seconds: float = TRACK_SECONDS, sample_rate: int = 22050) -> dict:
    import torch

    samples = torch.rand(1, 2, int(seconds * sample_rate)) * 0.2 - 0.1
    return {"waveform": samples, "sample_rate": sample_rate}


def run_pipeline(**overrides):
    from music2prompts import node as node_module
    from music2prompts.node import Music2PromptsLM

    node_module.LMStudioClient = FakeClient  # type: ignore[assignment]

    kwargs = dict(
        audio=make_audio(),
        instruction="A defiant night escape through a flooded garage.",
        lm_model="google/gemma-4-e4b",
        visual_style="",
        aspect_ratio="16:9",
        clip_seconds=6.0,
        num_shots=0,
        creativity=0.7,
        dynamicity=0.6,
        word_influence=0.6,
        whisper_device="cpu",
        seed=1234,
        whisper_skip=True,
        analyze_music=True,
        free_comfy_vram=False,
    )
    kwargs.update(overrides)
    return Music2PromptsLM.execute(**kwargs)


# --------------------------------------------------------------------------- checks


def check_lists_are_aligned(result) -> None:
    (
        start_frames,
        reference_prompts,
        subject_names,
        i2va,
        ref2va,
        negatives,
        indices,
        starts,
        ends,
        durations,
        transcript,
        analysis_json,
    ) = result.args
    count = len(indices)
    assert count >= 3, f"expected several shots, got {count}"
    for series in (start_frames, i2va, ref2va, negatives, starts, ends, durations):
        assert len(series) == count, "shot-aligned outputs must have equal length"
    assert indices == list(range(1, count + 1))
    assert len(reference_prompts) == len(subject_names) == len(SUBJECTS)
    assert transcript == ""
    json.loads(analysis_json)


def check_timing(result) -> None:
    starts, ends, durations = result.args[7], result.args[8], result.args[9]
    assert starts[0] == 0.0
    assert abs(ends[-1] - TRACK_SECONDS) < 0.05
    for index in range(1, len(starts)):
        assert abs(starts[index] - ends[index - 1]) < 0.01, "no gaps between shots"
    assert all(5.0 - 0.01 <= value <= 15.0 + 0.01 for value in durations)


def check_i2va_format(result) -> None:
    text = result.args[3][0]
    assert text.startswith(
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )
    assert "integrated_multimodal_description: [Shot 1] Live-action, grainy 16mm cinematic," in text
    assert "\n\noverall_soundscape: " in text
    assert "\n\nnon_diegetic_music: N/A" in text


def check_ref2va_format(result) -> None:
    text = result.args[4][0]
    for section in (
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ):
        assert section in text, f"missing section {section}"
    assert "<Subject 1> is a young singer" in text
    assert "<Subject 2> is a flooded underground parking garage" in text
    assert "<Subject 3>" not in text


def check_image_prompts(result) -> None:
    assert all(prompt.strip() for prompt in result.args[0])
    assert all(prompt.strip() for prompt in result.args[1])


def check_negatives(result) -> None:
    negatives = result.args[5]
    assert "watermark" in negatives[0]
    assert "daylight" in negatives[0]


CHECKS = {
    "lists_are_aligned": check_lists_are_aligned,
    "timing": check_timing,
    "i2va_format": check_i2va_format,
    "ref2va_format": check_ref2va_format,
    "image_prompts": check_image_prompts,
    "negatives": check_negatives,
}


def main() -> int:
    result = run_pipeline()
    failures = 0
    for name, check in CHECKS.items():
        try:
            check(result)
            print(f"  ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")

    forced = run_pipeline(num_shots=4)
    try:
        assert len(forced.args[6]) == 4
        print("  ok   num_shots_override")
    except AssertionError:
        failures += 1
        print(f"  FAIL num_shots_override: got {len(forced.args[6])} shots")

    print("\n--- sample I2VA prompt ---")
    print(result.args[3][0])
    print("\n--- sample Ref2VA prompt ---")
    print(result.args[4][0][:900])
    print(f"\n{'ALL CHECKS PASSED' if not failures else str(failures) + ' CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
