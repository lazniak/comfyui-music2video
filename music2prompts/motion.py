"""Motion Enhancer: rewrite each shot's video prompt against the frame it will start from.

The shot prompts are written before a single pixel exists. The LLM describes what it
*intends* the frame to contain, the image model then draws its own reading of that
description, and the two are never compared. Whatever the image actually shows - a
different framing, one subject instead of two, a coat that came out red - the video prompt
still describes the intention, and MiniMax H3 resolves the disagreement the only way it
can: it starts on the frame it was handed and dissolves into the scene it was told about.

This node closes that loop. It takes the pipe and the rendered start frames, shows each
frame to a vision-capable LLM together with that shot's own i2va prompt, and asks for the
description to be rewritten so it describes *this* frame - and so the motion in it fits the
shot's real duration rather than an arbitrary amount of action.

Only ``integrated_multimodal_description`` is rewritten. The H3 skeleton, the
``<Picture 1>`` reference and (unless you ask otherwise) the two sound fields are kept as
they were, so what comes back is the same prompt with its picture of the world corrected.

This node calls an LLM, once per shot. On ``lmstudio`` that is local and free; on the
cloud providers it is billed per token, and the panel under the node reports what it cost.
"""

from __future__ import annotations

import json

from comfy_api.latest import io

from . import cost as cost_module
from . import model_cache
from . import pipe as pipe_module
from .h3_format import assemble_i2va, split_i2va
from .lmstudio import DEFAULT_URL
from .providers import LLM_PROVIDERS, make_llm_client, pick_model
from .util import PREFIX, image_tensor_to_data_uri, log, raise_if_interrupted, warn

#: Words per second of clip. Eight is roughly what a shot description needs to fill the
#: time without running past it; the widget exists because models differ.
DEFAULT_DENSITY = 8.0

MIN_WORDS, MAX_WORDS = 25, 220

SYSTEM = (
    "You correct video prompts for MiniMax H3 (Hailuo 3) image-to-video generation.\n\n"
    "You are shown ONE image and the prompt that was written for it. The image is the "
    "first frame of the clip: at 0.00 seconds the video IS this image. Your job is to "
    "rewrite the prompt's 'integrated_multimodal_description' so that it describes this "
    "exact frame and what happens next, starting from it.\n\n"
    "Rules:\n"
    "1. The opening of the description must match the image: the same subjects, in the "
    "same number, wearing what they are wearing, in this location, framed by this camera "
    "position. If the image shows one person, do not write two. If the coat is red, it is "
    "red. Nothing may be described at 0.00 s that is not visible in the frame.\n"
    "2. Never introduce a subject, prop or location the image does not contain. Something "
    "may enter the frame during the shot, but it has to arrive - it cannot already be "
    "there.\n"
    "3. The clip is a fixed length. Describe only as much motion as fits in it: one clear "
    "action and one camera move for a short shot, not a sequence of events. Do not "
    "compress a scene; drop what does not fit.\n"
    "4. Keep the camera language natural ('the camera pushes in slowly'), keep any "
    "dialogue and on-screen text exactly as they are, and keep the shot's style and mood.\n"
    "5. Write one flowing paragraph in English, present tense, no headings, no bullet "
    "points, no field names, and never mention the image, the prompt or yourself.\n\n"
    "Answer strictly as JSON."
)


def _schema(rewrite_sound: bool) -> dict:
    properties = {
        "integrated_multimodal_description": {
            "type": "string",
            "description": "The rewritten description: this frame, and the motion that fits the clip.",
        },
        "changed": {
            "type": "string",
            "description": "One short sentence naming what disagreed with the image, or 'nothing'.",
        },
    }
    if rewrite_sound:
        properties["overall_soundscape"] = {"type": "string"}
        properties["non_diegetic_music"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _first(value, default=None):
    """One value out of what an is_input_list node receives (every input is a list)."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def frames_of(images) -> list[tuple]:
    """One (tensor, index) per frame, however the images arrived.

    A list of single-frame IMAGEs (what this pack's own 'images' socket carries), one
    batched IMAGE of N frames (what a sampler hands back), or a list of batches - all
    three mean the same thing here: the start frames, in shot order.
    """
    frames: list[tuple] = []
    for item in images or []:
        if item is None:
            continue
        shape = getattr(item, "shape", None)
        count = int(shape[0]) if shape is not None and len(shape) == 4 else 1
        frames.extend((item, index) for index in range(count))
    return frames


def target_words(seconds: float, density: float = DEFAULT_DENSITY) -> int:
    """How long the description should be for a clip of this length."""
    words = float(seconds or 0.0) * max(0.5, float(density))
    return int(max(MIN_WORDS, min(MAX_WORDS, round(words))))


class Music2VideoMotion(io.ComfyNode):
    """Rewrite each shot's video prompt against the frame it will start from."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        options = {kind: model_cache.snapshot(kind) or ["(none found)"] for kind in model_cache.KINDS}
        return io.Schema(
            node_id="Music2VideoMotion",
            display_name="🎵 Music2Video Motion Enhancer",
            category="Music2Video",
            description=(
                "Rewrites each shot's MiniMax H3 image-to-video prompt against the frame it "
                "will actually start from. The prompts are written before the images exist, "
                "so whatever the image model drew instead - a different framing, one subject "
                "instead of two, another colour - the prompt still describes the intention, "
                "and H3 resolves that by dissolving out of the frame it was given. This shows "
                "each frame to a vision-capable LLM together with its own prompt and asks for "
                "the description to be rewritten to match the frame, with only as much motion "
                "as the shot's real duration can hold. The H3 skeleton, the <Picture 1> "
                "reference and the sound fields are kept. One LLM call per shot: free on "
                "lmstudio, billed per token on the cloud providers."
            ),
            is_input_list=True,
            inputs=[
                io.Custom(pipe_module.PIPE_TYPE).Input(
                    "pipe",
                    tooltip=(
                        "The pipe from the Music2Video node. 'video_prompts_i2va' is what "
                        "gets rewritten; 'durations' decides how much motion each shot can "
                        "hold, and 'image_prompts_start' is shown to the model as what the "
                        "frame was meant to be."
                    ),
                ),
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The start frames, one per shot, in shot order - the node's own "
                        "'images' socket, or whatever your sampler produced from "
                        "'image_prompts_start'. A batch of N frames and a list of N frames "
                        "are read the same way. Fewer frames than shots is fine: the shots "
                        "without one keep their prompt untouched."
                    ),
                ),
                io.Combo.Input(
                    "llm_provider",
                    options=list(LLM_PROVIDERS),
                    default="lmstudio",
                    tooltip=(
                        "Which LLM rewrites the prompts. It must be able to see images: a "
                        "text-only model returns a description of nothing and the node keeps "
                        "the original prompt instead. 'lmstudio' is local and free - load a "
                        "vision model there (gemma, qwen-vl, llava class). The cloud "
                        "providers bill per token, once per shot."
                    ),
                ),
                io.Combo.Input(
                    "lm_model",
                    options=options["lmstudio"],
                    tooltip="Model served by LM Studio. It has to be a vision model.",
                ),
                io.Combo.Input(
                    "openrouter_model",
                    options=options["openrouter_llm"],
                    tooltip="Model used when 'llm_provider' is 'openrouter'. Pick one that accepts images.",
                ),
                io.Combo.Input(
                    "openai_model",
                    options=options["openai_llm"],
                    tooltip="Model used when 'llm_provider' is 'openai'. Pick one that accepts images.",
                ),
                io.Combo.Input(
                    "anthropic_model",
                    options=options["anthropic_llm"],
                    tooltip="Model used when 'llm_provider' is 'anthropic'. Pick one that accepts images.",
                ),
                io.String.Input(
                    "lm_url",
                    default=DEFAULT_URL,
                    tooltip="Address of the LM Studio server. Used only for 'lmstudio'.",
                ),
                io.String.Input(
                    "lm_api_key", default="", tooltip="Only if your LM Studio server asks for one."
                ),
                io.String.Input(
                    "openrouter_api_key",
                    default="",
                    tooltip="Empty reads OPENROUTER_API_KEY from the environment.",
                ),
                io.String.Input(
                    "openai_api_key", default="", tooltip="Empty reads OPENAI_API_KEY from the environment."
                ),
                io.String.Input(
                    "anthropic_api_key",
                    default="",
                    tooltip="Empty reads ANTHROPIC_API_KEY from the environment.",
                ),
                io.Float.Input(
                    "words_per_second",
                    default=DEFAULT_DENSITY,
                    min=2.0,
                    max=25.0,
                    step=0.5,
                    advanced=True,
                    tooltip=(
                        "How many words of description one second of clip is worth. The model "
                        f"is given a word budget of duration x this, clamped to {MIN_WORDS}-"
                        f"{MAX_WORDS}, which is what keeps a six-second shot from being "
                        "described as a minute of events. Raise it for a model that writes "
                        "thin, lower it for one that rambles."
                    ),
                ),
                io.Boolean.Input(
                    "rewrite_sound",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Off: 'overall_soundscape' and 'non_diegetic_music' are carried over "
                        "untouched, because a still frame says nothing about sound and the "
                        "originals were written against the track. On: the model rewrites "
                        "them too, which is worth it when the image turned out to be a "
                        "different place than the prompt assumed."
                    ),
                ),
                io.Int.Input(
                    "image_detail",
                    default=768,
                    min=256,
                    max=1536,
                    step=64,
                    advanced=True,
                    tooltip=(
                        "Longest side of the frame as it is sent, in pixels. Bigger sees more "
                        "and costs more; on a cloud provider this is most of the bill."
                    ),
                ),
                io.Float.Input(
                    "lm_temperature", default=0.4, min=0.0, max=2.0, step=0.05, advanced=True,
                    tooltip="Lower than the writing stages on purpose: this is a correction, not an invention.",
                ),
                io.Int.Input(
                    "lm_max_tokens", default=2048, min=256, max=32768, step=256, advanced=True,
                    tooltip="Cap on each reply. One shot's description, not the whole film.",
                ),
                io.Int.Input(
                    "lm_timeout", default=300, min=10, max=3600, step=10, advanced=True,
                    tooltip="Seconds to wait for one reply.",
                ),
                io.Int.Input(
                    "lm_retries", default=2, min=0, max=5, advanced=True,
                    tooltip="Retries per shot. A shot that still fails keeps its original prompt.",
                ),
                io.Boolean.Input(
                    "verbose", default=False, advanced=True,
                    tooltip="Log each shot's before and after lengths.",
                ),
            ],
            outputs=[
                io.Custom(pipe_module.PIPE_TYPE).Output(
                    display_name="pipe",
                    tooltip="The same pipe with the rewritten 'video_prompts_i2va' in it. Everything else is untouched.",
                ),
                io.String.Output(
                    display_name="video_prompts_i2va",
                    is_output_list=True,
                    tooltip="The rewritten prompts, in shot order, so they can be used without an expander.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="One line per shot: what disagreed with the frame, or what went wrong.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, **inputs) -> io.NodeOutput:
        pipe = _first(inputs.get("pipe"))
        if not isinstance(pipe, dict):
            raise ValueError(f"{PREFIX} 'pipe' must be the pipe from the Music2Video node.")

        prompts = [str(item) for item in (pipe.get("video_prompts_i2va") or [])]
        if not prompts:
            raise ValueError(
                f"{PREFIX} this pipe carries no video prompts. Wire the pipe from a run that "
                "produced shots."
            )
        durations = [float(value) for value in (pipe.get("durations") or [])]
        wanted = [str(item) for item in (pipe.get("image_prompts_start") or [])]

        frames = frames_of(inputs.get("images"))
        if not frames:
            raise ValueError(f"{PREFIX} no start frames came in on 'images'.")
        if len(frames) != len(prompts):
            warn(
                f"{len(frames)} frame(s) for {len(prompts)} shot(s) - the extra shots keep "
                "their prompt as it is. This happens when a render failed."
            )

        provider = str(_first(inputs.get("llm_provider"), "lmstudio")).strip().lower()
        model_key, api_key = pick_model(
            provider,
            str(_first(inputs.get("lm_model"), "")),
            {
                "openrouter": str(_first(inputs.get("openrouter_model"), "")),
                "openai": str(_first(inputs.get("openai_model"), "")),
                "anthropic": str(_first(inputs.get("anthropic_model"), "")),
            },
            {
                "lmstudio": str(_first(inputs.get("lm_api_key"), "")),
                "openrouter": str(_first(inputs.get("openrouter_api_key"), "")),
                "openai": str(_first(inputs.get("openai_api_key"), "")),
                "anthropic": str(_first(inputs.get("anthropic_api_key"), "")),
            },
        )

        density = float(_first(inputs.get("words_per_second"), DEFAULT_DENSITY))
        rewrite_sound = bool(_first(inputs.get("rewrite_sound"), False))
        detail = int(_first(inputs.get("image_detail"), 768))
        verbose = bool(_first(inputs.get("verbose"), False))

        ledger = cost_module.CostLedger(getattr(cls.hidden, "unique_id", None))
        client = make_llm_client(
            provider,
            lm_url=str(_first(inputs.get("lm_url"), DEFAULT_URL)),
            api_key=api_key,
            timeout=int(_first(inputs.get("lm_timeout"), 300)),
            retries=int(_first(inputs.get("lm_retries"), 2)),
            verbose=verbose,
            ledger=ledger,
        )
        schema = _schema(rewrite_sound)

        log(f"correcting {min(len(frames), len(prompts))} shot prompt(s) against their frames ({provider}/{model_key})")
        out: list[str] = list(prompts)
        notes: list[str] = []
        fixed = 0

        for index, prompt in enumerate(prompts):
            raise_if_interrupted()
            if index >= len(frames):
                notes.append(f"shot {index + 1}: no frame, left as it was")
                continue
            tensor, offset = frames[index]
            uri = image_tensor_to_data_uri(tensor, offset, max_side=detail)
            if not uri:
                notes.append(f"shot {index + 1}: the frame could not be read, left as it was")
                continue

            seconds = durations[index] if index < len(durations) else 6.0
            sections = split_i2va(prompt)
            budget = target_words(seconds, density)
            user = (
                f"Shot {index + 1} of {len(prompts)}. The clip is {seconds:.1f} seconds long, "
                f"so write about {budget} words - fit the motion to that time.\n\n"
                "CURRENT DESCRIPTION (rewrite this):\n"
                f"{sections['integrated_multimodal_description']}\n\n"
                + (
                    f"WHAT THE FRAME WAS MEANT TO BE (the image prompt; the frame is what it "
                    f"actually became):\n{wanted[index]}\n\n"
                    if index < len(wanted) and wanted[index]
                    else ""
                )
                + (
                    f"CURRENT SOUND:\n{json.dumps({k: sections[k] for k in list(sections)[1:]}, ensure_ascii=False)}\n\n"
                    if rewrite_sound
                    else ""
                )
                + "The attached image is the first frame. Correct the description to it."
            )

            try:
                data = client.chat_json(
                    model=model_key,
                    system=SYSTEM,
                    user=user,
                    schema=schema,
                    images=[uri],
                    temperature=float(_first(inputs.get("lm_temperature"), 0.4)),
                    max_tokens=int(_first(inputs.get("lm_max_tokens"), 2048)),
                    stage=f"motion {index + 1}",
                )
            except Exception as exc:
                warn(f"shot {index + 1} could not be corrected ({exc}); keeping its prompt")
                notes.append(f"shot {index + 1}: failed ({str(exc)[:120]}), left as it was")
                continue

            description = str((data or {}).get("integrated_multimodal_description") or "").strip()
            if not description:
                notes.append(f"shot {index + 1}: the model returned nothing, left as it was")
                continue

            soundscape = sections["overall_soundscape"]
            music = sections["non_diegetic_music"]
            if rewrite_sound:
                soundscape = str((data or {}).get("overall_soundscape") or "").strip() or soundscape
                music = str((data or {}).get("non_diegetic_music") or "").strip() or music

            out[index] = assemble_i2va(description, soundscape, music)
            fixed += 1
            changed = str((data or {}).get("changed") or "").strip() or "corrected"
            notes.append(f"shot {index + 1}: {changed}")
            if verbose:
                before = len(sections["integrated_multimodal_description"].split())
                log(f"shot {index + 1}: {before} -> {len(description.split())} words (budget {budget})")
            ledger.publish()

        log(f"{fixed}/{len(prompts)} prompt(s) corrected against their frames")
        ledger.publish(final=True)

        updated = dict(pipe)
        updated["video_prompts_i2va"] = out
        report = "\n".join(notes) or "nothing to correct"
        return io.NodeOutput(updated, out, report, ui=ledger.ui() or None)
