"""Take the pipe apart again, wherever one of its values is actually needed.

The producing node emits one ``M2P_PIPE`` instead of twelve string and number sockets.
This node hands them back. It also passes the pipe straight through, so it can be tapped
several times along a chain without a reroute: expand once next to the sampler for the
prompts, again next to a text preview for the transcript.

The schema is built from :data:`pipe.FIELDS`, the same tuple the producing node packs
from, so an output here can never end up carrying its neighbour's value.
"""

from __future__ import annotations

from comfy_api.latest import io

from . import pipe as pipe_module
from .util import PREFIX, blocked_when_empty

#: display_name -> the io class that declares it
KINDS = {"String": io.String, "Int": io.Int, "Float": io.Float, "Audio": io.Audio,
         "Image": io.Image, "Video": io.Video}


class Music2PromptsPipeExpand(io.ComfyNode):
    """Unpack a Music2Video pipe into its individual outputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Music2PromptsPipeExpand",
            display_name="🎵 Music2Video Pipe Expand",
            category="Music2Video",
            description=(
                "Splits the pipe from the Music2Video node back into the prompts, the "
                "subject names, the shot timings, the transcript and the analysis JSON. "
                "The pipe is passed through as well, so several of these can sit along one "
                "wire. Costs nothing and calls nothing - it only unpacks what the run "
                "already produced."
            ),
            inputs=[
                io.Custom(pipe_module.PIPE_TYPE).Input(
                    "pipe",
                    tooltip=(
                        "The 'pipe' output of the Music2Video node. Every field is always "
                        "present in it, so an output here is empty only when that part of "
                        "the run produced nothing."
                    ),
                ),
            ],
            outputs=[
                io.Custom(pipe_module.PIPE_TYPE).Output(
                    display_name="pipe",
                    tooltip=(
                        "The same pipe, untouched, so the next expander down the chain can "
                        "take its own values out of it."
                    ),
                ),
                *[
                    KINDS[field.kind].Output(
                        display_name=field.name,
                        is_output_list=field.is_list,
                        tooltip=field.tooltip,
                    )
                    for field in pipe_module.FIELDS
                ],
            ],
        )

    @classmethod
    def execute(cls, pipe) -> io.NodeOutput:
        # a plain TypeError here would surface as a bare traceback; name the likely mistake
        if not isinstance(pipe, dict):
            raise ValueError(
                f"{PREFIX} the 'pipe' input takes the pipe output of the Music2Video node, "
                f"not a {type(pipe).__name__}."
            )
        # A list output that came back empty would crash ComfyUI's list expansion in
        # whatever is wired to it; block that socket instead and let the rest run. The
        # single-value fields are left alone - an empty transcript is a normal answer
        # for an instrumental, and an empty string breaks nothing downstream.
        values = [
            blocked_when_empty(value, field.name, "this run produced none")
            if field.is_list
            else value
            for field, value in zip(pipe_module.FIELDS, pipe_module.unpack(pipe))
        ]
        return io.NodeOutput(pipe, *values)
