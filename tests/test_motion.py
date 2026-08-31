"""The Motion Enhancer: the prompt is corrected against the frame it will start from.

The failure it exists for is quiet. The prompts are written before the images exist, the
image model draws its own reading of them, and nothing compares the two - so H3 is handed a
frame and a description of a different scene, and dissolves between them.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import pipe as pipe_module  # noqa: E402
from music2prompts.h3_format import assemble_i2va, split_i2va  # noqa: E402


def motion():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.motion import Music2VideoMotion

    return Music2VideoMotion


def helpers():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts import motion

    return motion


# --------------------------------------------------------------------------- the skeleton


def test_the_prompt_is_rebuilt_not_patched():
    """H3 parses the three labels; a rewritten field goes back through the assembler."""
    original = assemble_i2va("a wide shot", "rain on metal", "a low synth pulse")
    sections = split_i2va(original)
    rebuilt = assemble_i2va("a close shot", sections["overall_soundscape"], sections["non_diegetic_music"])
    assert split_i2va(rebuilt) == {
        "integrated_multimodal_description": "a close shot",
        "overall_soundscape": "rain on metal",
        "non_diegetic_music": "a low synth pulse",
    }
    assert rebuilt.startswith("For the target video, at 0.00 seconds")
    assert "<Picture 1>" in rebuilt, "the line that says the image is the first frame"


def test_a_prompt_with_no_labels_is_read_as_all_description():
    """Then rewriting it still produces a well-formed prompt instead of nothing."""
    assert split_i2va("just some text")["integrated_multimodal_description"] == "just some text"


# --------------------------------------------------------------------------- the length rule


def test_the_word_budget_follows_the_shot_length():
    module = helpers()
    assert module.target_words(6.0, 8.0) == 48
    assert module.target_words(12.0, 8.0) == 96


def test_a_very_short_or_very_long_shot_is_still_describable():
    module = helpers()
    assert module.target_words(0.5, 8.0) == module.MIN_WORDS
    assert module.target_words(600.0, 8.0) == module.MAX_WORDS


def test_the_density_is_what_the_widget_changes():
    module = helpers()
    assert module.target_words(10.0, 4.0) < module.target_words(10.0, 12.0)


# --------------------------------------------------------------------------- the frames


class FakeTensor:
    def __init__(self, batch: int) -> None:
        self.shape = (batch, 8, 8, 3)


def test_a_list_of_single_frames_is_one_frame_per_shot():
    """What this pack's own 'images' socket carries."""
    frames = helpers().frames_of([FakeTensor(1), FakeTensor(1), FakeTensor(1)])
    assert len(frames) == 3
    assert [index for _, index in frames] == [0, 0, 0]


def test_one_batch_of_frames_is_read_the_same_way():
    """What a sampler hands back."""
    frames = helpers().frames_of([FakeTensor(4)])
    assert len(frames) == 4
    assert [index for _, index in frames] == [0, 1, 2, 3]


def test_batches_of_batches_still_come_out_in_shot_order():
    frames = helpers().frames_of([FakeTensor(2), FakeTensor(3)])
    assert [index for _, index in frames] == [0, 1, 0, 1, 2]


def test_a_missing_frame_is_skipped_rather_than_shifting_the_rest():
    assert len(helpers().frames_of([FakeTensor(1), None, FakeTensor(1)])) == 2


# --------------------------------------------------------------------------- the run


class FakeClient:
    """Stands in for the LLM: records what it was shown, answers with a fixed rewrite."""

    def __init__(self, reply=None, fail_on=()):
        self.calls: list[dict] = []
        self.reply = reply or (lambda index: {"integrated_multimodal_description": f"corrected {index}", "changed": "the coat"})
        self.fail_on = set(fail_on)

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls)
        if index in self.fail_on:
            raise RuntimeError("model said no")
        return self.reply(index)


@pytest.fixture
def wired(monkeypatch):
    """A pipe of three shots, three frames, and a fake client behind make_llm_client."""
    module = helpers()
    client = FakeClient()
    monkeypatch.setattr(module, "make_llm_client", lambda *a, **k: client)
    monkeypatch.setattr(module, "image_tensor_to_data_uri", lambda *a, **k: "data:image/png;base64,AA")
    packed = pipe_module.pack(
        video_prompts_i2va=[assemble_i2va(f"shot {n} as written", "rain", "a pulse") for n in (1, 2, 3)],
        image_prompts_start=[f"intended {n}" for n in (1, 2, 3)],
        durations=[6.0, 12.0, 6.0],
    )
    return client, packed, [FakeTensor(1), FakeTensor(1), FakeTensor(1)]


def run(pipe, images, **overrides):
    kwargs = dict(pipe=[pipe], images=images, llm_provider=["lmstudio"], lm_model=["a-model"])
    kwargs.update({key: [value] for key, value in overrides.items()})
    return motion().execute(**kwargs)


def test_every_shot_comes_back_corrected_and_still_well_formed(wired):
    client, packed, images = wired
    result = run(packed, images)

    prompts = result.args[1]
    assert len(prompts) == 3
    for index, prompt in enumerate(prompts, start=1):
        assert split_i2va(prompt)["integrated_multimodal_description"] == f"corrected {index}"
        assert prompt.startswith("For the target video")
    assert len(client.calls) == 3


def test_the_frame_is_what_the_model_is_shown(wired):
    client, packed, images = wired
    run(packed, images)
    assert all(call["images"] == ["data:image/png;base64,AA"] for call in client.calls)


def test_each_shot_is_told_how_long_it_is(wired):
    """The whole point of the length half: a 12 s shot may hold twice the action of a 6 s one."""
    client, packed, images = wired
    run(packed, images)
    assert "6.0 seconds" in client.calls[0]["user"]
    assert "12.0 seconds" in client.calls[1]["user"]
    assert "48 words" in client.calls[0]["user"] and "96 words" in client.calls[1]["user"]


def test_the_model_is_shown_what_the_frame_was_meant_to_be(wired):
    """The contrast between the image prompt and the frame is the correction it has to make."""
    client, packed, images = wired
    run(packed, images)
    assert "intended 1" in client.calls[0]["user"]


def test_one_shot_at_a_time_so_two_frames_cannot_be_confused(wired):
    client, packed, images = wired
    run(packed, images)
    assert all(len(call["images"]) == 1 for call in client.calls)


def test_the_sound_is_carried_over_untouched_by_default(wired):
    client, packed, images = wired
    prompts = run(packed, images).args[1]
    assert split_i2va(prompts[0])["overall_soundscape"] == "rain"
    assert split_i2va(prompts[0])["non_diegetic_music"] == "a pulse"


def test_the_sound_is_rewritten_when_asked_for(monkeypatch, wired):
    client, packed, images = wired
    client.reply = lambda index: {
        "integrated_multimodal_description": "corrected",
        "overall_soundscape": "wind",
        "non_diegetic_music": "silence",
        "changed": "the place",
    }
    prompts = run(packed, images, rewrite_sound=True).args[1]
    assert split_i2va(prompts[0])["overall_soundscape"] == "wind"


def test_a_shot_that_fails_keeps_the_prompt_it_had(monkeypatch, wired):
    """A correction is an improvement, never a precondition: nothing is lost when it fails."""
    client, packed, images = wired
    client.fail_on = {2}
    result = run(packed, images)

    prompts = result.args[1]
    assert split_i2va(prompts[1])["integrated_multimodal_description"] == "shot 2 as written"
    assert "shot 2: failed" in result.args[2]


def test_an_empty_answer_keeps_the_prompt_too(wired):
    client, packed, images = wired
    client.reply = lambda index: {"integrated_multimodal_description": "  ", "changed": ""}
    result = run(packed, images)
    assert split_i2va(result.args[1][0])["integrated_multimodal_description"] == "shot 1 as written"
    assert "returned nothing" in result.args[2]


def test_fewer_frames_than_shots_leaves_the_rest_alone(wired):
    """A failed render leaves a shot without a frame; it must not shift the others."""
    client, packed, _ = wired
    result = run(packed, [FakeTensor(1), FakeTensor(1)])

    prompts = result.args[1]
    assert split_i2va(prompts[0])["integrated_multimodal_description"] == "corrected 1"
    assert split_i2va(prompts[2])["integrated_multimodal_description"] == "shot 3 as written"
    assert "shot 3: no frame" in result.args[2]


def test_the_pipe_comes_back_with_only_that_one_field_changed(wired):
    client, packed, images = wired
    updated = run(packed, images).args[0]

    assert updated["video_prompts_i2va"] != packed["video_prompts_i2va"]
    for name in pipe_module.NAMES:
        if name != "video_prompts_i2va":
            assert updated[name] == packed[name], f"{name} must not have been touched"


def test_the_report_names_what_disagreed(wired):
    client, packed, images = wired
    assert "shot 1: the coat" in run(packed, images).args[2]


def test_a_pipe_with_no_prompts_says_so_instead_of_calling_anything(wired):
    client, _, images = wired
    with pytest.raises(ValueError, match="no video prompts"):
        run(pipe_module.pack(durations=[6.0]), images)
    assert not client.calls


def test_no_frames_at_all_says_so(wired):
    client, packed, _ = wired
    with pytest.raises(ValueError, match="no start frames"):
        run(packed, [])
    assert not client.calls


# --------------------------------------------------------------------------- the schema


def test_it_offers_the_same_providers_as_the_main_node():
    from music2prompts.providers import LLM_PROVIDERS

    ids = {item.id: item for item in motion().define_schema().inputs}
    assert ids["llm_provider"].options == list(LLM_PROVIDERS)
    for name in ("lm_model", "openrouter_model", "openai_model", "anthropic_model"):
        assert name in ids, "the same widget names, so the same JS hides the same ones"


def test_it_hands_back_a_pipe_and_the_prompts():
    outputs = motion().define_schema().outputs
    assert [item.display_name for item in outputs] == ["pipe", "video_prompts_i2va", "report"]
    assert outputs[1].is_output_list is True
