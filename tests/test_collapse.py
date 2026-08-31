"""Pipe Collapse: values go back into a pipe, so a pipe can be edited mid-graph.

Expand takes the pipe apart to *use* a value. Collapse takes values back in so one can be
*changed* - hand-edited prompts, a translation, another node's output - without losing the
run's timings, transcript and per-shot audio, which are on the same wire.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import pipe as pipe_module  # noqa: E402


def collapse():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.collapse import Music2PromptsPipeCollapse

    return Music2PromptsPipeCollapse


def full_pipe() -> dict:
    """A pipe with something recognisable in every field."""
    return pipe_module.pack(
        **{
            field.name: ([f"{field.name} 1", f"{field.name} 2"] if field.is_list else f"{field.name} value")
            for field in pipe_module.FIELDS
            if field.kind == "String"
        }
    )


def run(**inputs):
    """Every input of an is_input_list node arrives wrapped in a list."""
    return collapse().execute(**{name: (value if isinstance(value, list) else [value]) for name, value in inputs.items()})


# --------------------------------------------------------------------------- the round trip


def test_a_pipe_that_goes_straight_through_comes_out_the_same():
    original = full_pipe()
    assert run(pipe=[original]).args[0] == original


def test_expand_then_collapse_is_the_pipe_it_started_as():
    """The two nodes are each other's inverse; that is the whole contract."""
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.expand import Music2PromptsPipeExpand

    original = full_pipe()
    values = Music2PromptsPipeExpand.execute(original).args[1:]  # [0] is the pipe passed through
    supplied = {
        field.name: (list(value) if field.is_list else [value])
        for field, value in zip(pipe_module.FIELDS, values)
        if field.kind == "String"
    }
    assert run(pipe=[original], **supplied).args[0] == original


# --------------------------------------------------------------------------- replacing


def test_only_what_is_wired_is_replaced():
    original = full_pipe()
    result = run(pipe=[original], video_prompts_i2va=["edited 1", "edited 2"]).args[0]

    assert result["video_prompts_i2va"] == ["edited 1", "edited 2"]
    for name in pipe_module.NAMES:
        if name != "video_prompts_i2va":
            assert result[name] == original[name], f"{name} came from the pipe, untouched"


def test_a_replacement_of_a_different_length_is_taken_as_it_is():
    """Dropping a shot is a legitimate edit; nothing here should pad it back."""
    result = run(pipe=[full_pipe()], video_prompts_i2va=["only one"]).args[0]
    assert result["video_prompts_i2va"] == ["only one"]


def test_a_scalar_field_takes_one_value_not_a_list():
    scalars = [field for field in pipe_module.FIELDS if not field.is_list and field.kind == "String"]
    assert scalars, "the pipe has scalar fields; this test is about them"
    result = run(pipe=[full_pipe()], **{scalars[0].name: ["new"]}).args[0]
    assert result[scalars[0].name] == "new"


def test_the_original_pipe_is_not_modified():
    """It may still be wired somewhere else in the graph."""
    original = full_pipe()
    before = original["video_prompts_i2va"]
    run(pipe=[original], video_prompts_i2va=["edited"])
    assert original["video_prompts_i2va"] is before


# --------------------------------------------------------------------------- nothing wired


def test_an_input_nothing_is_wired_to_does_not_count_as_a_replacement():
    """ComfyUI hands an unconnected optional input through as None, or as [None]."""
    original = full_pipe()
    assert run(pipe=[original], video_prompts_i2va=[None]).args[0] == original
    assert collapse().execute(pipe=[original], video_prompts_i2va=None).args[0] == original


def test_an_empty_list_is_not_a_replacement_either():
    original = full_pipe()
    assert collapse().execute(pipe=[original], video_prompts_i2va=[]).args[0] == original


def test_with_no_pipe_at_all_it_builds_one_from_what_is_wired():
    """How a pipe is made for a graph that never ran the Music2Video node."""
    result = collapse().execute(video_prompts_i2va=["a", "b"]).args[0]
    assert result["video_prompts_i2va"] == ["a", "b"]
    assert set(result) == set(pipe_module.NAMES), "still a complete pipe"
    assert result["durations"] == []


def test_with_nothing_wired_at_all_it_is_an_empty_pipe():
    result = collapse().execute().args[0]
    assert result == pipe_module.pack()


def test_a_pipe_from_an_older_version_gains_the_fields_it_never_had():
    result = collapse().execute(pipe=[{"video_prompts_i2va": ["kept"]}]).args[0]
    assert result["video_prompts_i2va"] == ["kept"]
    assert set(result) == set(pipe_module.NAMES)


def test_something_that_is_not_a_pipe_is_ignored_rather_than_crashing():
    assert collapse().execute(pipe=["not a pipe"]).args[0] == pipe_module.pack()


# --------------------------------------------------------------------------- the schema


def test_there_is_one_socket_per_pipe_field_plus_the_pipe_itself():
    inputs = collapse().define_schema().inputs
    assert [item.id for item in inputs] == ["pipe", *pipe_module.NAMES]


def test_every_socket_is_optional_so_you_wire_only_what_you_changed():
    assert all(item.optional for item in collapse().define_schema().inputs)


def test_the_text_and_number_fields_are_sockets_not_widgets():
    """Without force_input a String input draws a text box, which cannot be wired into."""
    inputs = {item.id: item for item in collapse().define_schema().inputs}
    for field in pipe_module.FIELDS:
        if field.kind in {"String", "Int", "Float"}:
            assert getattr(inputs[field.name], "force_input", False) is True, field.name


def test_the_sockets_are_in_the_same_order_as_the_expander_s_outputs():
    """So the two nodes can be wired straight across, socket for socket."""
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.expand import Music2PromptsPipeExpand

    outputs = [item.display_name for item in Music2PromptsPipeExpand.define_schema().outputs]
    assert [item.id for item in collapse().define_schema().inputs][1:] == outputs[1:]


def test_it_hands_back_one_pipe():
    outputs = collapse().define_schema().outputs
    assert [item.display_name for item in outputs] == ["pipe"]
