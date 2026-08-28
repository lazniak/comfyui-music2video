"""Format tests for the MiniMax H3 renderers (no torch, no network)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.h3_format import (  # noqa: E402
    H3FormatError,
    H3Shot,
    Speaker,
    Subject,
    format_timecode,
    normalize_camera,
    render_i2va,
    render_ref2va,
    undefined_labels,
)


def make_shot(**overrides) -> H3Shot:
    shot = H3Shot(
        index=1,
        duration=7.0,
        style="Live-action, cinematic",
        opening="a medium-wide shot frames a woman beside a rain-covered train window",
        action="she lifts her gaze from a folded letter toward the passing city lights",
        camera="The camera trucks right with small amplitude at slow speed",
        diegetic_sound="Paper rustles between her fingers",
        soundscape="Train wheels keep a steady metallic rhythm under a low ventilation hum",
        music="Sustained cello notes at a slow tempo",
        speakers=[Speaker(description="a quiet, breathy young woman", line="I get off at the next station", language="English", mode="spoken")],
        subjects=[
            Subject(name="the woman", description="the young woman in the carriage", identity_lock="long dark hair and a navy coat"),
            Subject(name="the carriage", kind="location", description="the night train carriage", identity_lock="warm amber ceiling lights"),
        ],
    )
    for key, value in overrides.items():
        setattr(shot, key, value)
    return shot


# --------------------------------------------------------------------------- I2VA


def test_i2va_skeleton_is_exact():
    text = render_i2va(make_shot())
    lines = text.split("\n")
    assert lines[0] == (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )
    assert lines[1] == ""
    assert lines[2].startswith("integrated_multimodal_description: [Shot 1] Live-action, cinematic,")
    assert text.index("integrated_multimodal_description:") < text.index("overall_soundscape:")
    assert text.index("overall_soundscape:") < text.index("non_diegetic_music:")
    assert "\n\noverall_soundscape: " in text
    assert "\n\nnon_diegetic_music: " in text


def test_i2va_keeps_dialogue_verbatim_and_tagged():
    shot = make_shot(
        speakers=[Speaker(description="a woman", line="Wysiadam na następnej stacji", language="Polish", mode="sung")]
    )
    text = render_i2va(shot)
    assert "<d>[Polish] Wysiadam na następnej stacji</d>" in text
    assert "(S1) sings:" in text


def test_i2va_voiceover_adds_closed_lips_clause():
    shot = make_shot(
        speakers=[Speaker(description="a man", line="I still remember that road.", mode="voiceover")]
    )
    text = render_i2va(shot)
    assert "says in an off-screen voiceover:" in text
    assert "while their lips remain completely closed." in text


def test_i2va_music_na_passthrough():
    assert render_i2va(make_shot(music="")).rstrip().endswith("non_diegetic_music: N/A")
    assert render_i2va(make_shot(music="n/a")).rstrip().endswith("non_diegetic_music: N/A")


def test_i2va_extra_cut_uses_timecode_notation():
    from music2prompts.h3_format import Cut

    text = render_i2va(make_shot(cuts=[Cut(time=3.5, description="a close-up of her hands")]))
    assert "[Shot 2] At 00:03.500, the camera cuts to a close-up of her hands." in text


def test_i2va_on_screen_text_is_quoted():
    text = render_i2va(make_shot(on_screen_text=["NIGHT LINE 22:40"]))
    assert 'The text "NIGHT LINE 22:40" is visible in the frame.' in text


# --------------------------------------------------------------------------- Ref2VA


def test_ref2va_section_order():
    text = render_ref2va(make_shot())
    positions = [
        text.index("subject_definitions:"),
        text.index("summary:"),
        text.index("retention_analysis:"),
        text.index("detailed_description:"),
        text.index("overall_soundscape:"),
        text.index("non_diegetic_music:"),
    ]
    assert positions == sorted(positions)


def test_ref2va_renumbers_subjects_per_shot():
    shot = make_shot(
        subjects=[
            Subject(name="the drummer", description="the drummer", identity_lock="a red bandana"),
            Subject(name="the alley", kind="location", description="the neon alley", identity_lock="wet asphalt"),
        ]
    )
    text = render_ref2va(shot)
    assert "<Subject 1> is the drummer" in text
    assert "<Subject 2> is the neon alley" in text
    assert "<Subject 3>" not in text


def test_ref2va_every_subject_has_retention_line():
    text = render_ref2va(make_shot())
    retention = text.split("retention_analysis:")[1].split("detailed_description:")[0]
    assert "<Subject 1> (appears in [Shot 1]): fully_preserved" in retention
    assert "<Subject 2> (appears in [Shot 1]): fully_preserved" in retention


def test_ref2va_without_subjects_still_defines_one():
    text = render_ref2va(make_shot(subjects=[]))
    assert "<Subject 1> is the main on-screen subject" in text


def test_ref2va_has_no_undefined_labels():
    shot = make_shot(action="she looks toward <Picture 4> in the distance")
    text = render_ref2va(shot)
    assert undefined_labels(text, {"<Subject 1>", "<Subject 2>"}) == set()


def test_ref2va_strict_mode_rejects_undefined_labels():
    shot = make_shot(action="she looks toward <Video 2> in the distance")
    with pytest.raises(H3FormatError):
        render_ref2va(shot, strict=True)


# --------------------------------------------------------------------------- helpers


def test_field_labels_echoed_by_the_model_are_stripped():
    shot = make_shot(
        soundscape="overall_soundscape: Wind rushes over dry grass; non_diegetic_music: N/A",
        music="non_diegetic_music: N/A",
    )
    text = render_i2va(shot)
    assert "overall_soundscape: Wind rushes over dry grass." in text
    assert text.count("overall_soundscape:") == 1
    assert text.count("non_diegetic_music:") == 1
    assert text.rstrip().endswith("non_diegetic_music: N/A")


def test_schema_noise_is_stripped_from_the_speaker():
    shot = make_shot(
        speakers=[
            Speaker(
                description="Dialogue_mode: sung a hoarse young singer",
                line="Na chłodnej ziemi",
                language="Polish",
                mode="sung",
            )
        ]
    )
    text = render_i2va(shot)
    assert "Dialogue_mode" not in text
    assert "A hoarse young singer (S1) sings:" in text
    assert "<d>[Polish] Na chłodnej ziemi</d>" in text


def test_label_style_camera_becomes_a_sentence():
    shot = make_shot(camera="Tracking Shot with large amplitude at slow speed")
    assert "The camera tracks the subject with large amplitude at slow speed." in render_i2va(shot)


def test_subject_definition_reads_as_prose():
    shot = make_shot(
        subjects=[
            Subject(
                name="the wanderer",
                description="A figure of indeterminate age",
                identity_lock="heavy indigo wool cloak, dust-worn boots",
            )
        ]
    )
    text = render_ref2va(shot)
    assert "<Subject 1> is a figure of indeterminate age." in text
    assert "Key features: heavy indigo wool cloak, dust-worn boots." in text


def test_normalize_camera_keeps_official_vocabulary():
    assert normalize_camera("The camera pushes in with small amplitude at slow speed") == (
        "The camera pushes in with small amplitude at slow speed."
    )


def test_normalize_camera_repairs_unknown_motion():
    assert "static shot" in normalize_camera("she runs past the window").lower()
    assert normalize_camera("") == "The camera holds a static shot on the subject."


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.0, "00:00.000"), (3.5, "00:03.500"), (65.25, "01:05.250"), (-2.0, "00:00.000")],
)
def test_format_timecode(seconds, expected):
    assert format_timecode(seconds) == expected
