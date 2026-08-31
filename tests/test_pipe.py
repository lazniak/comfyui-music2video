"""One wire instead of twelve: what the pipe carries and what the expander gives back."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import pipe as pipe_module  # noqa: E402


def expander():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.expand import Music2PromptsPipeExpand

    return Music2PromptsPipeExpand


def node():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.node import Music2PromptsLM

    return Music2PromptsLM


# --------------------------------------------------------------------------- the pipe


def test_the_pipe_always_holds_every_field_so_unpacking_never_has_to_guess():
    packed = pipe_module.pack(shot_index=[1, 2])
    assert list(packed) == list(pipe_module.NAMES)
    assert packed["transcript"] == "", "a single value defaults to an empty string"
    assert packed["start_times"] == [], "a per-shot value defaults to an empty list"


def test_packing_something_the_pipe_does_not_carry_is_an_error_not_a_silent_drop():
    with pytest.raises(KeyError, match="images"):
        pipe_module.pack(images=[1])


def test_the_values_come_back_in_the_order_the_fields_declare():
    values = {name: [index] for index, name in enumerate(pipe_module.NAMES)}
    values["transcript"] = "t"
    values["analysis_json"] = "{}"
    unpacked = pipe_module.unpack(pipe_module.pack(**values))
    assert list(unpacked) == [values[name] for name in pipe_module.NAMES]


def test_a_pipe_from_an_older_version_expands_minus_what_it_never_held():
    partial = {"shot_index": [1, 2, 3]}
    unpacked = pipe_module.unpack(partial)
    assert unpacked[pipe_module.NAMES.index("shot_index")] == [1, 2, 3]
    assert unpacked[pipe_module.NAMES.index("transcript")] == ""
    assert unpacked[pipe_module.NAMES.index("durations")] == []


def test_something_that_is_not_a_pipe_says_so_rather_than_raising_a_key_error():
    with pytest.raises(TypeError, match="Music2Video node"):
        pipe_module.unpack(["a", "b"])


def test_the_summary_says_what_the_pipe_is_carrying():
    packed = pipe_module.pack(shot_index=[1, 2, 3], reference_subjects=["Mara", "the garage"])
    assert pipe_module.summary(packed) == "3 shot(s), 2 subject(s)"


# --------------------------------------------------------------------------- the sockets


def test_the_node_emits_one_pipe_and_keeps_the_media_on_their_own_sockets():
    schema = node().define_schema()
    names = [output.display_name for output in schema.outputs]
    assert names == ["pipe", "audio_clips", "images", "subject_images", "videos", "final_video"]
    assert schema.outputs[0].get_io_type() == pipe_module.PIPE_TYPE


def test_the_expander_declares_exactly_what_the_pipe_carries_and_in_that_order():
    """Built from the same tuple, so an output can never carry its neighbour's value."""
    schema = expander().define_schema()
    names = [output.display_name for output in schema.outputs]
    assert names == ["pipe", *pipe_module.NAMES]
    assert schema.inputs[0].get_io_type() == pipe_module.PIPE_TYPE


def test_the_list_shape_of_every_output_survives_the_round_trip():
    produced = {o.display_name: o for o in node().define_schema().outputs}
    expanded = {o.display_name: o for o in expander().define_schema().outputs}
    for field in pipe_module.FIELDS:
        assert expanded[field.name].is_output_list == field.is_list
    for name in ("audio_clips", "images", "videos"):
        assert produced[name].is_output_list is True


def test_every_expander_output_carries_the_tooltip_the_socket_used_to_have():
    assert all(output.tooltip for output in expander().define_schema().outputs)


def test_the_pipe_is_handed_through_so_a_chain_can_tap_it_more_than_once():
    packed = pipe_module.pack(shot_index=[1])
    result = expander().execute(packed)
    assert result.args[0] is packed
    assert len(result.args) == 1 + len(pipe_module.FIELDS)


def test_wiring_the_wrong_thing_into_the_expander_names_the_mistake():
    with pytest.raises(ValueError, match="pipe output of the Music2Video node"):
        expander().execute("not a pipe")


def test_the_pipe_type_is_its_own_so_a_pipe_from_another_pack_cannot_connect():
    assert pipe_module.PIPE_TYPE == "M2P_PIPE"
    assert pipe_module.PIPE_TYPE not in {"STRING", "INT", "FLOAT", "IMAGE", "VIDEO", "AUDIO", "PIPE"}


def test_the_per_shot_audio_rides_the_pipe_as_a_list():
    """Shot data, like the timings - a lipsync graph wants it off the same wire."""
    field = {item.name: item for item in pipe_module.FIELDS}["audio_clips"]
    assert (field.kind, field.is_list) == ("Audio", True)
    packed = pipe_module.pack(audio_clips=[{"waveform": 1}, {"waveform": 2}])
    assert packed["audio_clips"] == [{"waveform": 1}, {"waveform": 2}]
    assert pipe_module.unpack(packed)[pipe_module.NAMES.index("audio_clips")] == packed["audio_clips"]


def test_new_fields_are_appended_never_inserted():
    """Field order is the expander's socket order, and a saved wire refers to it by index."""
    assert pipe_module.NAMES[:13] == (
        "image_prompts_start", "image_prompts_reference", "reference_subjects",
        "video_prompts_i2va", "video_prompts_ref2va", "negative_prompts",
        "shot_index", "start_times", "end_times", "durations",
        "transcript", "analysis_json", "audio_clips",
    ), "inserting a field above these would silently re-point every wire below it"


def test_the_clips_are_reachable_both_ways():
    """Kept on its own socket as well: a lipsync node is often nowhere near an expander."""
    produced = [output.display_name for output in node().define_schema().outputs]
    expanded = [output.display_name for output in expander().define_schema().outputs]
    assert "audio_clips" in produced and "audio_clips" in expanded


def test_an_empty_field_blocks_its_own_socket_and_leaves_the_rest_alone():
    """One empty list must not take the whole graph down with an IndexError."""
    pytest.importorskip("comfy_execution.graph_utils", reason="needs a ComfyUI installation")
    from comfy_execution.graph_utils import ExecutionBlocker

    packed = pipe_module.pack(shot_index=[1, 2], transcript="")
    result = expander().execute(packed)
    got = dict(zip(pipe_module.NAMES, result.args[1:]))

    assert result.args[0] is packed, "the pipe itself is never blocked"
    assert got["shot_index"] == [1, 2], "a field with content is handed straight back"
    assert isinstance(got["durations"][0], ExecutionBlocker), "an empty list is blocked"
    assert got["transcript"] == "", "an empty string breaks nothing downstream - leave it"
    assert packed["durations"] == [], "the pipe keeps the real value; only the socket is blocked"



# --------------------------------------------------------------------------- the names


def test_the_pipe_carries_the_names_this_run_writes_under():
    """So a clip rendered by another subgraph can be saved into this run's own folder."""
    fields = {item.name: item for item in pipe_module.FIELDS}
    assert (fields["clip_prefixes"].kind, fields["clip_prefixes"].is_list) == ("String", True)
    assert (fields["final_video_name"].kind, fields["final_video_name"].is_list) == ("String", False)


def test_a_missing_name_is_an_empty_string_not_a_missing_key():
    packed = pipe_module.pack(shot_index=[1])
    assert packed["final_video_name"] == ""
    assert packed["clip_prefixes"] == []
