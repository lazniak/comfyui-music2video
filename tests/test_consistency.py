"""Keeping one face, one look and one voice across a set of independently rendered shots."""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import render as render_module  # noqa: E402
from music2prompts.h3_format import H3Shot, Speaker, Subject, render_ref2va  # noqa: E402
from music2prompts.render import (  # noqa: E402
    VideoRequest,
    build_fal_video_payload,
    fal_audio_field,
    fal_image_needs_image,
    fal_image_reference_field,
)

AUDIO = "data:audio/mpeg;base64,QUJD"
FRAME = "data:image/png;base64,QUJD"

TEXT_TO_IMAGE = {"required": ["prompt"], "properties": {"prompt": {"type": "string"}}}
EDIT = {
    "required": ["prompt", "image_urls"],
    "properties": {"prompt": {"type": "string"}, "image_urls": {"type": "array"}},
}
DRIVING_AUDIO = {
    "required": [],
    "properties": {
        "prompt": {"type": "string"},
        "image_url": {"type": "string"},
        "audio_url": {"type": "string"},
        "duration": {"type": "integer", "minimum": 2, "maximum": 15},
    },
}
H3_REFERENCE = {
    "required": ["prompt"],
    "properties": {
        "prompt": {"type": "string"},
        "reference_image_urls": {"type": "array", "maxItems": 9},
        "reference_audio_urls": {"type": "array", "maxItems": 3},
        "duration": {"type": "integer"},
    },
}
# alibaba/wan-3.0-prime: `audio` here means "generate a soundtrack", not "follow this one"
GENERATES_AUDIO = {
    "required": ["start_image_url"],
    "properties": {"start_image_url": {"type": "string"}, "audio": {"type": "boolean"}},
}


@pytest.fixture(autouse=True)
def schemas(monkeypatch):
    known: dict[str, dict] = {}
    monkeypatch.setattr(render_module, "fal_schema", lambda model, timeout=6.0: known.get(model, {}))
    return known


# --------------------------------------------------------------------------- capabilities


def test_a_text_to_image_model_admits_it_cannot_take_a_reference(schemas):
    schemas["t2i"] = TEXT_TO_IMAGE
    assert fal_image_reference_field("t2i") == ""
    assert fal_image_needs_image("t2i") == ""


def test_an_edit_model_names_the_field_and_says_it_is_required(schemas):
    schemas["edit"] = EDIT
    assert fal_image_reference_field("edit") == "image_urls"
    assert fal_image_needs_image("edit") == "image_urls"


def test_a_boolean_named_audio_is_not_an_audio_input(schemas):
    """wan-3.0-prime's `audio` means 'generate a soundtrack' - sending a clip to it is nonsense."""
    schemas["wan3"] = GENERATES_AUDIO
    assert fal_audio_field("wan3") == ""


def test_a_driving_audio_field_is_recognised(schemas):
    schemas["wan27"] = DRIVING_AUDIO
    schemas["h3ref"] = H3_REFERENCE
    assert fal_audio_field("wan27") == "audio_url"
    assert fal_audio_field("h3ref") == "reference_audio_urls"


# --------------------------------------------------------------------------- payloads


def test_the_shot_audio_goes_into_whichever_field_the_endpoint_declares():
    request = VideoRequest(prompt="p", seconds=6, first_frame=FRAME, references=[FRAME], audio=AUDIO)
    driving, _ = build_fal_video_payload("wan27", request, DRIVING_AUDIO)
    assert driving["audio_url"] == AUDIO, "a string field takes the URI itself"
    listed, _ = build_fal_video_payload("h3ref", request, H3_REFERENCE)
    assert listed["reference_audio_urls"] == [AUDIO], "a list field takes a list"


def test_audio_is_never_sent_to_a_model_that_has_no_field_for_it():
    request = VideoRequest(prompt="p", seconds=6, first_frame=FRAME, audio=AUDIO)
    payload, _ = build_fal_video_payload("wan3", request, GENERATES_AUDIO)
    assert "audio" not in payload and "audio_url" not in payload


def test_the_audio_is_the_last_thing_to_be_dropped_on_a_refusal():
    request = VideoRequest(prompt="p", seconds=6, first_frame=FRAME, audio=AUDIO, seed=3)
    _, optional = build_fal_video_payload("wan27", request, DRIVING_AUDIO)
    assert "audio_url" not in optional, "dropping the vocal to satisfy the endpoint defeats the point"


def test_the_endpoint_is_told_not_to_rewrite_our_prompt():
    """H3 rewrites the prompt per request by default, re-inventing each shot's look."""
    h3 = {"required": ["prompt"], "properties": {"prompt": {}, "prompt_expansion_mode": {"default": "balanced"}}}
    wan = {"required": [], "properties": {"prompt": {}, "enable_prompt_expansion": {"type": "boolean"}}}
    quiet = VideoRequest(prompt="p", seconds=6, expansion="minimal")
    assert build_fal_video_payload("h3", quiet, h3)[0]["prompt_expansion_mode"] == "fast"
    assert build_fal_video_payload("wan", quiet, wan)[0]["enable_prompt_expansion"] is False
    loud = VideoRequest(prompt="p", seconds=6, expansion="rich")
    assert build_fal_video_payload("h3", loud, h3)[0]["prompt_expansion_mode"] == "quality"
    assert build_fal_video_payload("wan", loud, wan)[0]["enable_prompt_expansion"] is True


def test_the_endpoints_own_default_is_left_alone_when_asked():
    h3 = {"required": ["prompt"], "properties": {"prompt": {}, "prompt_expansion_mode": {}}}
    payload, _ = build_fal_video_payload("h3", VideoRequest(prompt="p", expansion="model default"), h3)
    assert "prompt_expansion_mode" not in payload


# --------------------------------------------------------------------------- model index


def test_the_model_index_is_paged_so_edit_endpoints_are_not_cut_off(monkeypatch):
    """fal ignores page_size and answers with 40 items; one call would hide most of them."""
    seen: list[dict] = []

    def get(url, params=None, timeout=None):
        seen.append(params or {})
        page = int((params or {}).get("page", 1))
        items = [{"id": f"{params['categories']}-{page}-{index}"} for index in range(40)]
        return types.SimpleNamespace(
            status_code=200, json=lambda: {"items": items, "page": page, "pages": 3}
        )

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))
    ids = render_module._fal_index(("image-to-image",), pages=2)
    assert [call["page"] for call in seen] == [1, 2]
    assert len(ids) == 80


def test_a_short_index_stops_early(monkeypatch):
    def get(url, params=None, timeout=None):
        return types.SimpleNamespace(
            status_code=200, json=lambda: {"items": [{"id": "only"}], "page": 1, "pages": 1}
        )

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))
    assert render_module._fal_index(("text-to-image",), pages=5) == ["only"]


def test_the_image_list_includes_the_category_the_edit_endpoints_live_in(monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(
        render_module, "_fal_index", lambda categories, pages=2: asked.extend(categories) or []
    )
    render_module.probe_fal_images_raw()
    assert "image-to-image" in asked, "without it no model can be told which face to keep"


# --------------------------------------------------------------------------- H3 prompt


def shot(**kwargs) -> H3Shot:
    base = dict(
        index=1,
        duration=6.0,
        style="grainy 16mm",
        opening="she stands under a flickering light",
        action="she steps forward",
        subjects=[
            Subject(name="Mara", description="a young woman with cropped hair", identity_lock="scarred brow"),
            Subject(name="the garage", kind="location", description="a flooded parking level"),
        ],
    )
    base.update(kwargs)
    return H3Shot(**base)


def test_a_reference_image_is_cited_inside_the_subject_it_defines():
    """The images were always sent; nothing in the prompt said which subject each one was."""
    subjects = shot().subjects
    subjects[0].picture = 1
    subjects[1].picture = 2
    text = render_ref2va(shot(subjects=subjects))
    assert "<Subject 1> is a young woman with cropped hair, whose appearance comes from <Picture 1>" in text
    assert "<Picture 2>" in text


def test_a_subject_without_a_picture_is_still_defined_in_words():
    text = render_ref2va(shot())
    assert "<Subject 1> is a young woman with cropped hair." in text
    assert "<Picture" not in text, "an undefined label would be stripped anyway - do not write one"


def test_the_audio_is_named_and_given_a_retention_line():
    spoken = shot(
        speakers=[Speaker(description="Mara", line="I am not going back", language="Polish", mode="sung")],
        audio_reference="this shot's own slice of the original track, carrying the vocal",
    )
    text = render_ref2va(spoken)
    assert "<Audio 1> is this shot's own slice of the original track" in text
    assert "voice-timbre reference for <Subject 1> (S1)" in text
    assert "<Audio 1>: fully_copy" in text
    assert "mouths it in exact sync" in text


def test_no_audio_label_appears_when_no_clip_is_sent():
    assert "<Audio" not in render_ref2va(shot())


def test_an_instrumental_shot_still_gets_the_audio_retention_line():
    text = render_ref2va(shot(audio_reference="the instrumental bar under this shot"))
    assert "<Audio 1>: fully_copy" in text
    assert "voice-timbre" not in text, "nobody is speaking, so there is no voice to reference"


# --------------------------------------------------------------------------- node wiring


def node():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.node import Music2PromptsLM

    return Music2PromptsLM


def clip(seconds: float) -> dict:
    torch = pytest.importorskip("torch")
    return {"waveform": torch.zeros(1, 2, int(44100 * seconds)), "sample_rate": 44100}


def test_no_audio_is_encoded_for_a_model_that_cannot_take_it(schemas, caplog):
    schemas["h3i2v"] = {"required": ["prompt"], "properties": {"prompt": {"type": "string"}}}
    assert node()._shot_audio([clip(6)], "fal", "h3i2v", True) == [""]
    assert "declares no input for a driving audio track" in caplog.text


def test_openrouter_is_told_plainly_that_it_has_no_audio_input(caplog):
    assert node()._shot_audio([clip(6)], "openrouter", "any", True) == [""]
    assert "no input for an audio track" in caplog.text


def test_a_shot_outside_the_endpoints_audio_window_goes_without_it(schemas, caplog):
    schemas["h3ref"] = H3_REFERENCE
    uris = node()._shot_audio([clip(1.0), clip(6.0), clip(40.0)], "fal", "h3ref", True)
    assert uris[0] == "" and uris[2] == "", "H3 takes 2-15 s per reference clip"
    assert uris[1].startswith("data:audio/mpeg;base64,")


def test_switching_off_lipsync_encodes_nothing(schemas):
    schemas["h3ref"] = H3_REFERENCE
    assert node()._shot_audio([clip(6)], "fal", "h3ref", False) == [""]


def test_the_edit_model_takes_over_once_there_is_an_identity_to_keep(schemas):
    schemas["t2i"] = TEXT_TO_IMAGE
    schemas["edit"] = EDIT
    client = object()
    assert node()._frame_model("fal", "t2i", "edit", True, client) == ("edit", client)
    assert node()._frame_model("fal", "t2i", "edit", False, client) == ("t2i", client)


def test_a_text_to_image_model_with_references_warns_instead_of_failing_quietly(schemas, caplog):
    schemas["t2i"] = TEXT_TO_IMAGE
    assert node()._frame_model("fal", "t2i", "", True, None)[0] == "t2i"
    assert "no field for a reference image" in caplog.text


def test_openrouter_needs_no_second_model(schemas):
    client = object()
    assert node()._frame_model("openrouter", "gemini", "", True, client) == ("gemini", client)
