"""The bundle of text and numbers one run produces, carried down a single wire.

Seventeen sockets on one node is a wall of noodles, and most of them were per-shot values
that almost always travel together. They now leave as one ``M2P_PIPE`` value - a plain
dict - and the *Music2Video Pipe Expand* node hands them back individually wherever they
are actually needed. The per-shot audio rides along with them, because it is shot data
like the timings are, and a lipsync graph wants it next to the prompts.

IMAGE and VIDEO stay as sockets: those are normally wired straight into a preview or a
save node, so hiding them behind an expander would cost a node and buy nothing.

This module is the single source of truth for what the pipe holds. The producing node and
the expander both build their schema from :data:`FIELDS`, so the two cannot drift apart -
the failure that would otherwise show up as an output silently carrying its neighbour's
value.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The socket type. Distinctive on purpose: a pipe from another pack must not connect here.
PIPE_TYPE = "M2P_PIPE"


@dataclass(frozen=True)
class Field:
    """One value inside the pipe, and how the expander should declare it."""

    name: str
    kind: str  # the io.<kind>.Output to build for it
    is_list: bool
    tooltip: str


#: In the order the node used to emit them, so a rewired workflow reads the same top to bottom.
FIELDS: tuple[Field, ...] = (
    Field(
        "image_prompts_start",
        "String",
        True,
        (
        "One start-frame image prompt per shot, in shot order, written by the LLM's "
        "image-prompt stage; a shot the model skipped falls back to a string assembled "
        "locally from that shot's opening, action and the art direction, so this list is "
        "never short. Always produced, whatever 'image_provider' is set to. Wire it into a "
        "CLIPTextEncode node (whose CONDITIONING feeds a sampler) or into a text preview node "
        "- it is the exact text the built-in renderer sends when 'image_provider' is not "
        "'pipe-steps'."
        ),
    ),
    Field(
        "image_prompts_reference",
        "String",
        True,
        (
        "One reference-sheet image prompt per subject (not per shot), index-aligned with "
        "'reference_subjects': neutral background, even lighting, whole subject visible, no "
        "narrative action. Empty only when no subjects were defined (the subjects stage "
        "returned none, or 'max_subjects' is 0). If the reference-prompt stage itself fails, "
        "every entry is filled from a locally assembled fallback instead, so the list still "
        "matches 'reference_subjects' in length. These are exactly the prompts "
        "'render_subject_sheets' renders into 'subject_images'."
        ),
    ),
    Field(
        "reference_subjects",
        "String",
        True,
        (
        "The subject names only - not prompts - one per subject, in the same order and length "
        "as 'image_prompts_reference'. A subject can be a character, location, prop, vehicle "
        "or style; the name is what binds a subject to its <Picture N> reference in the "
        "ref2va prompts. Empty when no subjects were defined (the subjects stage returned "
        "none, or 'max_subjects' is 0)."
        ),
    ),
    Field(
        "video_prompts_i2va",
        "String",
        True,
        (
        "One MiniMax H3 image-to-video (I2VA) prompt per shot: a header declaring <Picture 1> "
        "fully referenced at 0.00 s, then integrated_multimodal_description, "
        "overall_soundscape and non_diegetic_music. It assumes the matching entry of 'images' "
        "is supplied as the clip's first frame - the <Picture 1> reference is written even on "
        "a prompts-only run where no image exists. Always produced; feed it to an "
        "image-to-video endpoint together with that start frame."
        ),
    ),
    Field(
        "video_prompts_ref2va",
        "String",
        True,
        (
        "One MiniMax H3 reference-to-video (Ref2VA) prompt per shot, in the full block form: "
        "subject_definitions, summary, retention_analysis, detailed_description, "
        "overall_soundscape, non_diegetic_music. Unlike 'video_prompts_i2va' it carries no "
        "first frame. Identity comes from reference images cited as <Picture N>, and those "
        "citations appear only once the run has rendered subject sheets: with "
        "'video_provider' set and 'video_prompt_source' = ref2va, the strings are re-rendered "
        "with the sheet numbers bound in. The same re-render also adds an <Audio 1> "
        "definition and retention line when the shot's audio is being sent. Without either, "
        "undefined labels are stripped and the subjects are described in words only. Suits "
        "minimax/h3/reference-to-video."
        ),
    ),
    Field(
        "negative_prompts",
        "String",
        True,
        (
        "One negative prompt per shot: 'negative_prompt_base', the run-wide negative the "
        "art-direction stage added, and the shot's own, merged with duplicates dropped. It "
        "reaches only the built-in start-frame renders on fal: the fal payload builder adds "
        "negative_prompt when the endpoint's schema declares it (or when that schema could "
        "not be read at all), and it is the first field dropped if the endpoint then refuses "
        "the request. OpenRouter's image API is never sent a negative, no video request "
        "carries one, and the subject sheets are rendered with the raw 'negative_prompt_base' "
        "text only, not with these merged strings. Encode it with CLIPTextEncode and wire the "
        "result into a sampler's negative conditioning when you render images yourself."
        ),
    ),
    Field(
        "shot_index",
        "Int",
        True,
        (
        "The shot numbers, 1-based and consecutive (1..N), one per shot. The prompts, times, "
        "audio clips and images are all in this order and the same length, so use it as the "
        "index or label when you fan the lists out into batch nodes. 'videos' is the "
        "exception - failed clips are skipped, so it can be shorter."
        ),
    ),
    Field(
        "start_times",
        "Float",
        True,
        (
        "Each shot's start in seconds from the beginning of the input track, rounded to 3 "
        "decimals, one per shot. The shots are consecutive with no gaps or overlaps, so the "
        "first value is 0.0 and each value equals the previous shot's entry in 'end_times'."
        ),
    ),
    Field(
        "end_times",
        "Float",
        True,
        (
        "Each shot's end in seconds from the beginning of the input track, rounded to 3 "
        "decimals, one per shot. The last value is the full track duration; every other value "
        "is the next shot's start."
        ),
    ),
    Field(
        "durations",
        "Float",
        True,
        (
        "Shot length in seconds (end minus start, rounded to 3 decimals), one per shot; the "
        "planner keeps these inside the 'min_shot_seconds'-'max_shot_seconds' window wherever "
        "the track length allows. The video endpoint is asked for this length rounded to "
        "whole seconds - on fal then clamped into whatever range that endpoint declares and "
        "snapped to the nearest value of its duration list - and the returned clip is "
        "trimmed, or its last frame held, back to the exact figure here during concatenation."
        ),
    ),
    Field(
        "transcript",
        "String",
        False,
        (
        "A single string: the raw Whisper text of the whole track, with no timings. Empty "
        "when 'whisper_skip' is on, when transcription failed, or when the track has no "
        "vocal. A .txt is written to ComfyUI/output/music2prompts when 'save_transcript' is "
        "on: the whole transcript, then one block per shot with that shot's timings, section, "
        "start-frame image prompt and the first line of its i2va video prompt truncated to "
        "200 characters (the i2va prompt is logged even when the clips were rendered from "
        "ref2va), and the words sung inside that shot."
        ),
    ),
    Field(
        "analysis_json",
        "String",
        False,
        (
        "A single JSON string holding the whole run: track duration, the librosa analysis, "
        "the full transcript with its language and word count, the LLM's interpretation, art "
        "direction and subject bible, the descriptions of any wired-in reference images, "
        "per-shot timings, section, lyrics and raw LLM fields, the rendering report "
        "(providers, models, clip paths) and the main settings. Always produced. The same "
        "text is written to ..._analysis.json in ComfyUI/output/music2prompts when "
        "'save_json' is on."
        ),
    ),
    # Appended rather than filed next to 'durations' on purpose: the order of these fields
    # is the order of the expander's output sockets, and a saved workflow refers to those
    # by index. Inserting one in the middle would silently re-point every wire below it.
    Field(
        "audio_clips",
        "Audio",
        True,
        (
        "One AUDIO per shot, cut from the input track at that shot's boundaries and widened "
        "on both sides by 'audio_clip_padding', with the original sample rate and channel "
        "layout untouched. Always produced, even on a prompts-only run. Send them to "
        "PreviewAudio / SaveAudio or a lipsync node. These are the same clips the node also "
        "offers on its own 'audio_clips' socket - here so a lipsync graph can take them off "
        "the pipe next to the prompts instead of running a second wire across the canvas."
        ),
    ),
)

#: Fast lookup, and the check the expander uses to reject a foreign dict.
NAMES: tuple[str, ...] = tuple(field.name for field in FIELDS)


def pack(**values) -> dict:
    """Build a pipe. Every field is always present, so unpacking never has to guess."""
    unknown = set(values) - set(NAMES)
    if unknown:
        raise KeyError(f"not part of the pipe: {', '.join(sorted(unknown))}")
    return {field.name: values.get(field.name, [] if field.is_list else "") for field in FIELDS}


def unpack(pipe) -> tuple:
    """The values in FIELDS order.

    A missing key yields an empty list or an empty string rather than raising: a pipe
    built by an older version of this pack should still expand, minus what it never held.
    """
    if not isinstance(pipe, dict):
        raise TypeError(
            "this input takes the pipe from the Music2Video node, not a "
            f"{type(pipe).__name__}"
        )
    return tuple(pipe.get(field.name, [] if field.is_list else "") for field in FIELDS)


def summary(pipe) -> str:
    """One line for a log or a tooltip: what this pipe actually carries."""
    if not isinstance(pipe, dict):
        return "not a pipe"
    shots = len(pipe.get("shot_index") or [])
    subjects = len(pipe.get("reference_subjects") or [])
    return f"{shots} shot(s), {subjects} subject(s)"
