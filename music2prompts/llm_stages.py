"""Prompt-engineering stages executed on the local LM Studio model.

Every stage asks for strict JSON. Nothing here formats a final MiniMax H3
prompt - that is done deterministically in :mod:`h3_format`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from .h3_format import CAMERA_MOTIONS
from .lmstudio import LMStudioClient
from .util import as_list, chunked, first_str, log, raise_if_interrupted, warn

# --------------------------------------------------------------------------- static knowledge

CAMERA_VOCAB = ", ".join(CAMERA_MOTIONS)

H3_RULES = f"""MiniMax H3 prompt rules you must respect when writing shot content:
- Camera motion must use this vocabulary: {CAMERA_VOCAB}; optionally followed by
  "with small amplitude" / "with large amplitude" and "at slow speed" / "at fast speed",
  written as a natural English sentence (e.g. "The camera pushes in with small amplitude at slow speed").
- Everything you write must be visible or audible. No plot summaries, no emotions-as-abstractions,
  no explanations of meaning.
- Dialogue or sung lyrics stay verbatim in their original language; never translate them.
- Text that is physically visible in the frame is quoted verbatim.
- overall_soundscape covers ambience, physical action sounds and non-verbal human sounds only -
  never dialogue, singing or music.
- non_diegetic_music is score the characters cannot hear: instrumentation, tempo, rhythm, dynamics.
  Use "N/A" when there should be none.
"""

IMAGE_PROMPT_RULES = """Image prompt rules:
- One flowing paragraph of natural English, 40-90 words. No tag lists, no "masterpiece, 8k, trending".
- Cover: main subject and what they are doing, environment, composition/shot size, lens and depth of
  field, lighting and time of day, colour palette, film stock or rendering style.
- Describe only what is inside the frame. No camera motion (it is a still), no sound, no story.
- Never mention the music, the song, the artist or the prompt itself.
"""


def load_h3_guide(max_chars: int = 0) -> str:
    """Optionally enrich the system prompt with the official H3 guides.

    Uses the guides shipped with ``ComfyUI-MiniMaxH3-Easy`` when that pack is
    installed next to this one. Absence is fine - the compact rules above are
    always included.
    """
    if max_chars <= 0:
        return ""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    custom_nodes = os.path.dirname(here)
    base = os.path.join(
        custom_nodes, "ComfyUI-MiniMaxH3-Easy", "prompt_guides", "h3_general", "references"
    )
    parts: list[str] = []
    budget = max_chars
    for name in ("base-en.txt", "ref-en.txt"):
        path = os.path.join(base, name)
        if not os.path.isfile(path) or budget <= 0:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        excerpt = text[: max(0, budget // 2)]
        budget -= len(excerpt)
        parts.append(f"--- {name} (excerpt) ---\n{excerpt}")
    if parts:
        log("using the official MiniMax H3 guides found in ComfyUI-MiniMaxH3-Easy")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- schemas


def _object(properties: dict[str, Any]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _string(description: str = "") -> dict:
    return {"type": "string", "description": description} if description else {"type": "string"}


def _string_array(description: str = "") -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


SCHEMA_INTERPRETATION = _object(
    {
        "genre": _string("music genre"),
        "mood": _string("overall emotional tone"),
        "themes": _string_array("3-6 short thematic keywords"),
        "narrative_arc": _string("2-4 sentences describing the story the video tells"),
        "lyrics_language": _string("language of the sung or spoken words, or 'instrumental'"),
        "summary": _string("one sentence summary of the piece"),
    }
)

SCHEMA_ART_DIRECTION = _object(
    {
        "visual_style": _string("e.g. 'grainy 16mm live-action', '3D CG', 'watercolor 2D animation'"),
        "color_palette": _string_array("3-6 concrete colours"),
        "lighting": _string("lighting design in one sentence"),
        "lens_and_texture": _string("lens choice, grain, halation, depth of field"),
        "camera_language": _string("how the camera behaves across the film"),
        "world": _string("2-3 sentences describing the world the shots live in"),
        "negative_extra": _string_array("things that must never appear"),
    }
)

SCHEMA_SUBJECTS = _object(
    {
        "subjects": {
            "type": "array",
            "items": _object(
                {
                    "name": _string("short unique name, e.g. 'the drummer'"),
                    "kind": {"type": "string", "enum": ["character", "location", "prop", "vehicle", "style"]},
                    "description": _string("what it is, as a noun phrase: 'a young drummer in a wet leather coat'"),
                    "identity_lock": _string(
                        "reusable visual details as a comma-separated noun phrase "
                        "('shaved head, silver raincoat, cracked microphone'). Never write instructions "
                        "like 'must maintain'."
                    ),
                    "reference_prompt_hint": _string("what a clean reference image of it should show"),
                }
            ),
        }
    }
)

SCHEMA_SHOTS = _object(
    {
        "shots": {
            "type": "array",
            "items": _object(
                {
                    "shot": {"type": "integer"},
                    "subjects": _string_array("names of the subjects visible in this shot"),
                    "opening": _string(
                        "the composition at the first frame as a noun phrase: framing, subject, environment"
                    ),
                    "action": _string("what develops during the shot, in visible terms, as full sentences"),
                    "camera": _string(
                        "one full sentence starting with 'The camera ...' and using the official motion "
                        "vocabulary, e.g. 'The camera pushes in with small amplitude at slow speed'"
                    ),
                    "diegetic_sound": _string("sound produced inside the scene, or empty"),
                    "soundscape": _string(
                        "1-3 sentences of ambience and physical sounds. Plain prose only - never repeat "
                        "field names such as 'overall_soundscape:' or 'non_diegetic_music:'"
                    ),
                    "music": _string("non-diegetic score description, or exactly 'N/A'"),
                    "speaker": _string("who performs the line: type, age, voice character, or empty"),
                    "dialogue": _string("the verbatim line or lyric to be heard, or empty"),
                    "dialogue_mode": {"type": "string", "enum": ["spoken", "sung", "voiceover", "none"]},
                    "on_screen_text": _string("text physically visible in frame, or empty"),
                    "negative_extra": _string("extra things to avoid in this shot, or empty"),
                }
            ),
        }
    }
)

SCHEMA_IMAGE_PROMPTS = _object(
    {
        "prompts": {
            "type": "array",
            "items": _object({"shot": {"type": "integer"}, "prompt": _string()}),
        }
    }
)

SCHEMA_REFERENCE_PROMPTS = _object(
    {
        "prompts": {
            "type": "array",
            "items": _object({"name": _string(), "prompt": _string()}),
        }
    }
)

SCHEMA_IMAGE_DESCRIPTIONS = _object(
    {"descriptions": _string_array("one description per supplied image")}
)


# --------------------------------------------------------------------------- runner


def pair_with_shots(batch: list[Any], items: list[dict], stage: str = "") -> list[tuple[Any, dict | None]]:
    """Match what the model wrote to the shots it was asked to write about.

    Every batched stage sends a list of shots, each carrying its own number, and asks for
    one object per shot echoing that number back. Whether the right number comes back is
    the question. A model that renumbers each batch from 1, or slips by one, hands every
    shot its neighbour's description - and because the shot text and the image prompt are
    two separate calls, they can slip differently. The result is a start frame showing one
    thing and a video prompt describing another, which is exactly what a video model then
    dissolves between.

    So the numbers are trusted only when they are precisely the ones that were sent, which
    also covers a model that answered out of order. Otherwise the order is used: a model
    that gets the numbering wrong still writes the shots in the order it was given them.
    """
    wanted = [slot.index for slot in batch]
    numbers = [int(item.get("shot", 0) or 0) for item in items]
    if len(items) == len(batch):
        if sorted(numbers) == sorted(wanted):
            by_number = dict(zip(numbers, items))
            return [(slot, by_number[slot.index]) for slot in batch]
        warn(
            f"{stage or 'a stage'} numbered its shots {numbers} when it was asked for {wanted}; "
            "going by the order they came back in instead"
        )
        return list(zip(batch, items))
    if items:
        warn(f"{stage or 'a stage'} answered for {len(items)} of the {len(batch)} shot(s) it was sent")
    by_number: dict[int, dict] = {}
    for number, item in zip(numbers, items):
        by_number.setdefault(number, item)
    return [(slot, by_number.get(slot.index)) for slot in batch]


class StageRunner:
    """Runs the LLM stages against one LM Studio model."""

    def __init__(
        self,
        client: LMStudioClient,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        seed: int | None = None,
        guide_excerpt: str = "",
        verbose: bool = False,
        progress: Callable[[str], None] | None = None,
        reasoning_effort: str = "none",
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.guide_excerpt = guide_excerpt
        self.verbose = verbose
        self.progress = progress or (lambda message: log(message))

    def _call(self, system: str, user: str, schema: dict, stage: str, images: list[str] | None = None) -> Any:
        # every stage and every batch within a stage comes through here, so this one
        # check is what stops a cancelled run before it pays for the next call
        raise_if_interrupted()
        return self.client.chat_json(
            model=self.model,
            system=system,
            user=user,
            schema=schema,
            images=images,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stage=stage,
            reasoning_effort=self.reasoning_effort,
        )

    # ---------------------------------------------------------------- stages

    def describe_reference_images(self, images: list[str]) -> list[str]:
        if not images:
            return []
        self.progress("describing reference images")
        try:
            data = self._call(
                system=(
                    "You are a film production designer. Describe each supplied reference image so it can be "
                    "recreated consistently: subject identity, wardrobe, materials, colours, environment, "
                    "lighting and style. One dense paragraph per image, no speculation about meaning."
                ),
                user=f"Describe the {len(images)} reference image(s), in order.",
                schema=SCHEMA_IMAGE_DESCRIPTIONS,
                stage="reference images",
                images=images,
            )
            return [str(item) for item in as_list(data.get("descriptions"))][: len(images)]
        except Exception as exc:
            warn(f"reference image description failed ({exc}); continuing without it")
            return []

    def interpret(self, facts: dict, instruction: str, transcript: str, word_influence: float) -> dict:
        self.progress("stage 1/5 - interpreting the track")
        lyric_hint = (
            "Follow the lyrics closely and visualise what they literally say."
            if word_influence > 0.33
            else "Ignore the literal words; build on atmosphere and rhythm."
            if word_influence < -0.33
            else "Balance literal lyric imagery with atmosphere."
        )
        user = (
            f"MEASURED FACTS (trust these numbers):\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            f"TRANSCRIPT (may be empty for instrumentals):\n{transcript[:4000]}\n\n"
            f"USER BRIEF:\n{instruction}\n\n{lyric_hint}"
        )
        return self._call(
            system=(
                "You are a music-video director analysing a track before shooting. "
                "You never invent numbers: BPM, duration and section times come from the measured facts. "
                "Answer strictly as JSON."
            ),
            user=user,
            schema=SCHEMA_INTERPRETATION,
            stage="interpretation",
        )

    def art_direction(
        self, interpretation: dict, instruction: str, visual_style: str, creativity: float, references: list[str]
    ) -> dict:
        self.progress("stage 2/5 - art direction")
        style_line = (
            f"The user requires this visual style: {visual_style}."
            if visual_style.strip()
            else "Choose a visual style that fits the music."
        )
        reference_block = (
            "\n\nREFERENCE IMAGES:\n" + "\n".join(f"- {text}" for text in references) if references else ""
        )
        user = (
            f"INTERPRETATION:\n{json.dumps(interpretation, ensure_ascii=False)}\n\n"
            f"USER BRIEF:\n{instruction}\n\n{style_line}\n"
            f"Creativity level {creativity:.2f} (0 = stay literal and grounded, 1 = bold and surreal)."
            f"{reference_block}"
        )
        return self._call(
            system=(
                "You are a cinematographer and colourist defining the look of a short film. "
                "Be concrete and physical: lenses, stocks, fixtures, colours, textures. Answer strictly as JSON."
            ),
            user=user,
            schema=SCHEMA_ART_DIRECTION,
            stage="art direction",
        )

    def subjects(
        self, interpretation: dict, art: dict, instruction: str, max_subjects: int, references: list[str]
    ) -> list[dict]:
        self.progress("stage 3/5 - recurring subjects")
        reference_block = (
            "\n\nREFERENCE IMAGES (these MUST become subjects, described verbatim):\n"
            + "\n".join(f"- {text}" for text in references)
            if references
            else ""
        )
        user = (
            f"INTERPRETATION:\n{json.dumps(interpretation, ensure_ascii=False)}\n\n"
            f"ART DIRECTION:\n{json.dumps(art, ensure_ascii=False)}\n\n"
            f"USER BRIEF:\n{instruction}\n\n"
            f"Define at most {max_subjects} recurring subjects that must stay identical across every shot."
            f"{reference_block}"
        )
        data = self._call(
            system=(
                "You are a production designer building a consistency bible. Each subject must be reusable "
                "verbatim across shots: exact wardrobe, hair, materials, colours, distinguishing marks. "
                "Answer strictly as JSON."
            ),
            user=user,
            schema=SCHEMA_SUBJECTS,
            stage="subjects",
        )
        subjects = [item for item in as_list(data.get("subjects")) if isinstance(item, dict)]
        return subjects[:max_subjects]

    def shot_content(
        self,
        slots: list[Any],
        interpretation: dict,
        art: dict,
        subjects: list[dict],
        instruction: str,
        dynamicity: float,
        word_influence: float,
        include_dialogue: bool,
        batch_size: int = 4,
    ) -> dict[int, dict]:
        """Fill in the creative content of every shot. Timing is fixed, not negotiable."""
        system = (
            "You are directing individual shots of a music video. The timing is already locked - you only "
            "describe what happens inside each shot.\n\n"
            + H3_RULES
            + ("\n\n" + self.guide_excerpt if self.guide_excerpt else "")
            + "\nAnswer strictly as JSON."
        )
        subject_block = json.dumps(
            [
                {
                    "name": first_str(subject, "name"),
                    "identity_lock": first_str(subject, "identity_lock", "description"),
                }
                for subject in subjects
            ],
            ensure_ascii=False,
        )

        results: dict[int, dict] = {}
        batches = list(chunked(slots, batch_size))
        for position, batch in enumerate(batches, start=1):
            self.progress(f"stage 4/5 - shot content {position}/{len(batches)}")
            shot_block = json.dumps(
                [
                    {
                        "shot": slot.index,
                        "start": slot.start,
                        "end": slot.end,
                        "duration": slot.duration,
                        "section": slot.section,
                        "lyrics_in_shot": slot.lyrics if include_dialogue else "",
                    }
                    for slot in batch
                ],
                ensure_ascii=False,
            )
            previous = ""
            if results:
                last_index = max(results)
                previous = (
                    "\n\nPREVIOUS SHOT (continue from it):\n"
                    + json.dumps(results[last_index], ensure_ascii=False)[:1200]
                )
            user = (
                f"USER BRIEF:\n{instruction}\n\n"
                f"INTERPRETATION:\n{json.dumps(interpretation, ensure_ascii=False)}\n\n"
                f"ART DIRECTION:\n{json.dumps(art, ensure_ascii=False)}\n\n"
                f"SUBJECT BIBLE:\n{subject_block}\n\n"
                f"SHOTS TO WRITE (timing is fixed):\n{shot_block}"
                f"{previous}\n\n"
                f"Pacing level {dynamicity:.2f} (0 = calm and static, 1 = restless and kinetic). "
                f"Lyric literalness {word_influence:+.2f}. "
                + (
                    "When a shot has lyrics, put them verbatim in 'dialogue' with dialogue_mode 'sung'."
                    if include_dialogue
                    else "Leave 'dialogue' empty and dialogue_mode 'none'."
                )
                + f" Return exactly {len(batch)} shots, in the order they are listed above, each "
                "echoing its own 'shot' number unchanged."
            )
            try:
                data = self._call(system, user, SCHEMA_SHOTS, stage=f"shot content {position}")
            except Exception as exc:
                warn(f"shot batch {position} failed ({exc}); using a neutral fallback for it")
                data = {"shots": []}

            items = [item for item in as_list(data.get("shots")) if isinstance(item, dict)]
            for slot, item in pair_with_shots(batch, items, f"shot content {position}"):
                results[slot.index] = item or {}
        return results

    def image_prompts(
        self,
        slots: list[Any],
        content: dict[int, dict],
        art: dict,
        subjects: list[dict],
        aspect_ratio: str,
        batch_size: int = 4,
    ) -> dict[int, str]:
        system = (
            "You write prompts for a modern text-to-image model (Flux / Qwen-Image / Z-Image class).\n\n"
            + IMAGE_PROMPT_RULES
            + "\nAnswer strictly as JSON."
        )
        identity = json.dumps(
            {first_str(s, "name"): first_str(s, "identity_lock", "description") for s in subjects},
            ensure_ascii=False,
        )
        results: dict[int, str] = {}
        batches = list(chunked(slots, batch_size))
        for position, batch in enumerate(batches, start=1):
            self.progress(f"stage 5/5 - image prompts {position}/{len(batches)}")
            payload = json.dumps(
                [
                    {
                        "shot": slot.index,
                        "opening": first_str(content.get(slot.index, {}), "opening"),
                        "action": first_str(content.get(slot.index, {}), "action"),
                        "subjects": as_list(content.get(slot.index, {}).get("subjects")),
                    }
                    for slot in batch
                ],
                ensure_ascii=False,
            )
            user = (
                f"LOOK:\n{json.dumps(art, ensure_ascii=False)}\n\n"
                f"IDENTITY LOCK:\n{identity}\n\n"
                f"SHOTS:\n{payload}\n\n"
                f"Frame everything for a {aspect_ratio} image. Write the first frame of each shot as a still. "
                f"Return exactly {len(batch)} prompts, in the same order as SHOTS, each echoing its "
                "own 'shot' number unchanged."
            )
            try:
                data = self._call(system, user, SCHEMA_IMAGE_PROMPTS, stage=f"image prompts {position}")
            except Exception as exc:
                warn(f"image prompt batch {position} failed ({exc}); falling back to shot text")
                data = {"prompts": []}
            items = [item for item in as_list(data.get("prompts")) if isinstance(item, dict)]
            for slot, item in pair_with_shots(batch, items, f"image prompts {position}"):
                results[slot.index] = str((item or {}).get("prompt", "") or "")
        return results

    def reference_prompts(self, subjects: list[dict], art: dict, aspect_ratio: str) -> dict[str, str]:
        if not subjects:
            return {}
        self.progress("reference / character-sheet prompts")
        user = (
            f"LOOK:\n{json.dumps(art, ensure_ascii=False)}\n\n"
            f"SUBJECTS:\n{json.dumps(subjects, ensure_ascii=False)}\n\n"
            f"Write one clean reference image prompt per subject for a {aspect_ratio} frame. "
            "Neutral studio-like presentation, even lighting, full identity visible, no narrative action, "
            "no text overlays - these images are used as identity references for a video model."
        )
        try:
            data = self._call(
                system=(
                    "You write reference-sheet image prompts that lock a character, location or prop so a video "
                    "model can reuse it.\n\n" + IMAGE_PROMPT_RULES + "\nAnswer strictly as JSON."
                ),
                user=user,
                schema=SCHEMA_REFERENCE_PROMPTS,
                stage="reference prompts",
            )
        except Exception as exc:
            warn(f"reference prompts failed ({exc})")
            return {}
        return {
            str(item.get("name", "")).strip(): str(item.get("prompt", "")).strip()
            for item in as_list(data.get("prompts"))
            if isinstance(item, dict) and item.get("prompt")
        }
