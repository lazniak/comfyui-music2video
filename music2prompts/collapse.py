"""The pipe, put back together - Pipe Expand read backwards.

Expand takes the pipe apart so a value can be used. This takes the values back in, so a
value can be *changed*: edit the prompts by hand, run them through another node, translate
them, filter the shots - then put them back on the wire and carry on with the rest of the
run's timings, transcript and audio untouched.

Wire the original pipe into ``pipe`` and only what you actually changed into the sockets
below it: everything you leave unwired comes through from that pipe unaltered. With no
pipe wired it builds one from nothing, which is how a pipe is made for a graph that never
ran the Music2Video node at all.
"""

from __future__ import annotations

from comfy_api.latest import io

from . import pipe as pipe_module
from .expand import KINDS
from .util import log

#: Kinds ComfyUI would otherwise draw as a text box or a number spinner. These are meant
#: to be wired, so they are declared as sockets.
WIDGET_KINDS = {"String", "Int", "Float"}


def _first(value, default=None):
    """One value out of what an is_input_list node receives (every input is a list)."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


class Music2PromptsPipeCollapse(io.ComfyNode):
    """Build a Music2Video pipe, or replace parts of one."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Music2PromptsPipeCollapse",
            display_name="🎵 Music2Video Pipe Collapse",
            category="Music2Video",
            description=(
                "Puts values back into a pipe - the other direction from Pipe Expand. Wire "
                "the pipe you started from into 'pipe' and only the sockets you actually "
                "changed; everything left unwired comes through unaltered. Costs nothing and "
                "calls nothing. Use it to edit prompts by hand, translate them, or swap in "
                "what another node produced, and still keep the run's timings, transcript "
                "and per-shot audio on the same wire."
            ),
            is_input_list=True,
            inputs=[
                io.Custom(pipe_module.PIPE_TYPE).Input(
                    "pipe",
                    optional=True,
                    tooltip=(
                        "The pipe to start from. Every field not wired below is taken from "
                        "it. Leave it empty to build a pipe from nothing, in which case the "
                        "fields you do not supply come out empty."
                    ),
                ),
                *[
                    KINDS[field.kind].Input(
                        field.name,
                        optional=True,
                        tooltip=f"Replaces this field on the pipe. {field.tooltip}",
                        **({"force_input": True} if field.kind in WIDGET_KINDS else {}),
                    )
                    for field in pipe_module.FIELDS
                ],
            ],
            outputs=[
                io.Custom(pipe_module.PIPE_TYPE).Output(
                    display_name="pipe",
                    tooltip="The pipe with whatever you wired in, and the rest as it was.",
                ),
            ],
        )

    @classmethod
    def execute(cls, pipe=None, **fields) -> io.NodeOutput:
        base = _first(pipe)
        values = {
            name: base[name] for name in pipe_module.NAMES if isinstance(base, dict) and name in base
        }

        replaced: list[str] = []
        for field in pipe_module.FIELDS:
            supplied = fields.get(field.name)
            if supplied is None:
                continue
            # is_input_list wraps everything; an input nothing is wired to arrives empty
            items = [item for item in supplied if item is not None] if isinstance(supplied, list) else [supplied]
            if not items:
                continue
            values[field.name] = items if field.is_list else items[0]
            replaced.append(field.name)

        if replaced:
            log(f"pipe rebuilt with {len(replaced)} field(s) replaced: {', '.join(replaced)}")
        elif not isinstance(base, dict):
            log("pipe built from nothing - no pipe wired in and no field supplied")
        return io.NodeOutput(pipe_module.pack(**values))
