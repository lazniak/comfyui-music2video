"""Width and height from a ratio - ComfyUI's own arithmetic, plus a ratio you can type.

The parity tests matter more than they look: the point of this node is that a workflow can
swap it in for the core Resolution Selector and get identical numbers. If the two ever
disagree, the one that changed is this one.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.render import ASPECT_RATIO_OPTIONS, parse_ratio, ratio_label, resolution  # noqa: E402


def node():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.resolution import Music2VideoResolution

    return Music2VideoResolution


# --------------------------------------------------------------------------- the arithmetic


def test_a_square_megapixel_is_the_number_everyone_knows():
    assert resolution(1, 1, 1.0, 8) == (1024, 1024)


def test_the_ratio_decides_the_shape_and_the_megapixels_the_size():
    assert resolution(16, 9, 1.0, 8) == (1368, 768)
    assert resolution(9, 16, 1.0, 8) == (768, 1368)


def test_a_pixel_size_is_read_as_the_ratio_it_reduces_to():
    """1920x1080 is not a size here - it is 16:9 written the long way."""
    assert resolution(1920, 1080, 1.0, 8) == resolution(16, 9, 1.0, 8)


def test_both_sides_land_on_the_multiple():
    for multiple in (8, 16, 32, 64):
        width, height = resolution(21, 9, 1.7, multiple)
        assert width % multiple == 0 and height % multiple == 0


def test_more_megapixels_is_more_pixels():
    small = resolution(16, 9, 0.5, 8)
    large = resolution(16, 9, 2.0, 8)
    assert large[0] > small[0] and large[1] > small[1]


# --------------------------------------------------------------------------- typing one


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("21:10", (21.0, 10.0)),
        ("21x10", (21.0, 10.0)),
        ("21/10", (21.0, 10.0)),
        ("1920x1080", (1920.0, 1080.0)),
        ("2.39", (2.39, 1.0)),
        ("2.39:1", (2.39, 1.0)),
        (" 16 : 9 ", (16.0, 9.0)),
        ("4X3", (4.0, 3.0)),
    ],
)
def test_the_forms_people_actually_write(text, expected):
    assert parse_ratio(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "wide", "16:", "16:0", "-16:9", "1:2:3"])
def test_nonsense_is_said_out_loud_rather_than_guessed_at(text):
    """This is a text box. Silently rendering the wrong shape would be worse."""
    with pytest.raises(ValueError):
        parse_ratio(text)


def test_the_label_is_the_shortest_honest_form():
    assert ratio_label(1920, 1080) == "16:9"
    assert ratio_label(21, 10) == "21:10"
    assert ratio_label(2.39, 1) == "2.39:1"
    assert ratio_label(1, 1) == "1:1"


# --------------------------------------------------------------------------- the node


def test_the_presets_are_the_core_node_s_own():
    node()  # the presets live beside the schema, which needs ComfyUI to import
    from music2prompts.resolution import ASPECT_RATIOS

    assert list(ASPECT_RATIOS) == [
        "1:1 (Square)",
        "2:3 (Portrait Photo)",
        "3:2 (Photo)",
        "3:4 (Portrait Standard)",
        "4:3 (Standard)",
        "9:16 (Portrait Widescreen)",
        "16:9 (Widescreen)",
        "21:9 (Ultrawide)",
    ], "a workflow that swaps one node for the other must read the same"


def test_a_preset_comes_out_as_width_height_and_a_plain_ratio():
    assert node().execute("16:9 (Widescreen)", "", 1.0, 8).args == (1368, 768, "16:9")


def test_a_typed_ratio_reaches_the_output_as_typed():
    result = node().execute("custom", "21:10", 1.0, 8)
    assert result.args[2] == "21:10"
    assert result.args[:2] == resolution(21, 10, 1.0, 8)


def test_a_typed_size_comes_out_reduced():
    assert node().execute("custom", "1920x1080", 1.0, 8).args[2] == "16:9"


def test_custom_with_nothing_typed_says_so():
    with pytest.raises(ValueError, match="custom ratio"):
        node().execute("custom", "", 1.0, 8)


def test_the_ratio_leaves_on_a_socket_the_main_node_accepts():
    """The main node's aspect_ratio is a COMBO widget; only a COMBO output reaches it."""
    outputs = node().define_schema().outputs
    assert [item.id for item in outputs] == ["width", "height", "aspect_ratio"]
    assert outputs[2].io_type == "COMBO"
    assert outputs[2].options == ASPECT_RATIO_OPTIONS


def test_the_options_are_the_ones_the_main_node_offers():
    """So the socket's dropdown reads like the widget it is meant to feed."""
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.node import ASPECT_RATIOS as main_node_ratios

    assert ASPECT_RATIO_OPTIONS == main_node_ratios


def test_the_typed_ratio_is_offered_but_never_required():
    schema = node().define_schema()
    ids = [item.id for item in schema.inputs]
    assert ids == ["aspect_ratio", "custom_ratio", "megapixels", "multiple"]
    assert schema.inputs[0].default == "16:9 (Widescreen)"
