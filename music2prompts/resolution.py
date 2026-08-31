"""Width and height from an aspect ratio and a megapixel target.

This is ComfyUI's own **Resolution Selector**, with the arithmetic kept identical - a
workflow that swaps one for the other gets the same numbers back - and two things added
that the core node has no reason to carry:

* a **custom** ratio, typed rather than picked, for the shapes no preset covers
  (``21:10``, ``2.39``, ``1920x1080``);
* an **aspect_ratio** output, so the same ratio that sized the latent also reaches the
  Music2Video node's own ``aspect_ratio`` widget instead of being set twice by hand and
  drifting apart. It is a COMBO socket, which is what that widget is.
"""

from __future__ import annotations

from comfy_api.latest import io

from . import render as render_module
from .util import log

CUSTOM = "custom"

#: The core node's list, verbatim, so the dropdown reads the same in both.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}

OPTIONS = [*ASPECT_RATIOS, CUSTOM]


class Music2VideoResolution(io.ComfyNode):
    """Calculate width and height from an aspect ratio and a megapixel target."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Music2VideoResolution",
            display_name="🎵 Music2Video Resolution",
            category="Music2Video",
            description=(
                "Width and height from an aspect ratio and a megapixel target - ComfyUI's "
                "Resolution Selector arithmetic exactly, plus a ratio you can type yourself "
                "and an aspect_ratio output that plugs straight into the Music2Video node, so "
                "the shape of the latent and the shape the prompts are written for cannot "
                "drift apart."
            ),
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=OPTIONS,
                    default="16:9 (Widescreen)",
                    tooltip=(
                        "The aspect ratio for the output dimensions. 'custom' reads "
                        "'custom_ratio' instead, for the shapes no preset covers."
                    ),
                ),
                io.String.Input(
                    "custom_ratio",
                    default="",
                    tooltip=(
                        "Used only while 'aspect_ratio' is 'custom'. Write it as width:height "
                        "- '21:10', '5:4', '2.39' (cinemascope, same as '2.39:1'), or a size "
                        "you already have in mind, '1920x1080', which is read as the ratio it "
                        "reduces to (16:9). Only the shape is taken from it, never the size: "
                        "'megapixels' decides that."
                    ),
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Target total megapixels. 1.0 MP is 1024x1024 for a square.",
                ),
                io.Int.Input(
                    "multiple",
                    default=8,
                    min=8,
                    max=128,
                    step=4,
                    tooltip=(
                        "Both sides are rounded to a multiple of this. 8 suits latents; "
                        "raise it for a model that insists on 16 or 32."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Calculated width in pixels."),
                io.Int.Output("height", tooltip="Calculated height in pixels."),
                io.Combo.Output(
                    "aspect_ratio",
                    options=render_module.ASPECT_RATIO_OPTIONS,
                    tooltip=(
                        "The same ratio in the plain 'W:H' form, reduced ('1920x1080' comes "
                        "out as '16:9'). Wire it into the Music2Video node's 'aspect_ratio' "
                        "input: a ratio arriving down a wire is not held to that widget's six "
                        "presets, so a custom one reaches the prompts and the render payloads "
                        "as typed."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        aspect_ratio: str,
        custom_ratio: str = "",
        megapixels: float = 1.0,
        multiple: int = 8,
    ) -> io.NodeOutput:
        if aspect_ratio == CUSTOM or aspect_ratio not in ASPECT_RATIOS:
            ratio = render_module.parse_ratio(custom_ratio)
        else:
            ratio = ASPECT_RATIOS[aspect_ratio]
        width, height = render_module.resolution(ratio[0], ratio[1], megapixels, multiple)
        label = render_module.ratio_label(ratio[0], ratio[1])
        log(f"{label} at {megapixels:g} MP -> {width}x{height}")
        return io.NodeOutput(width, height, label)
