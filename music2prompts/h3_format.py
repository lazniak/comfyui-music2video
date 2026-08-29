"""Deterministic renderers for MiniMax H3 (Hailuo 3) prompt formats.

The language model only supplies *fields*; this module assembles the final
strings so the required skeleton is exact regardless of model quality.

Reference: MiniMax H3 official prompt-writing guides
(``base-en.txt`` for T2VA/I2VA/FL2VA/L2VA and ``ref-en.txt`` for Ref2VA),
shipped with the ``ComfyUI-MiniMaxH3-Easy`` node pack.

This module is intentionally dependency-free (stdlib only) so it can be unit
tested without torch, numpy or a running ComfyUI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- vocabulary

CAMERA_MOTIONS: tuple[str, ...] = (
    "Zoom In",
    "Zoom Out",
    "Push In",
    "Pull Out",
    "Pan Left",
    "Pan Right",
    "Truck Left",
    "Truck Right",
    "Tilt Up",
    "Tilt Down",
    "Pedestal Up",
    "Pedestal Down",
    "Arc Shot",
    "Tracking Shot",
    "Static Shot",
    "Shake Slightly",
    "Shake Strongly",
    "POV",
    "Roll Clockwise",
    "Roll Counterclockwise",
)

CAMERA_AMPLITUDES: tuple[str, ...] = ("with small amplitude", "with large amplitude")
CAMERA_SPEEDS: tuple[str, ...] = ("at slow speed", "at fast speed")

CUT_PHRASES: tuple[str, ...] = (
    "the camera cuts to",
    "the shot cuts to",
    "the shot transitions to",
    "the shot changes to",
    "the shot switches to",
)

RETENTION_STATES: tuple[str, ...] = (
    "fully_preserved",
    "partially_preserved",
    "transferred",
    "reused",
    "reference",
)

#: The markers the guide allows for ``<Audio N>``; visible content uses RETENTION_STATES.
AUDIO_STATES: tuple[str, ...] = ("fully_copy", "partially_copy", "reference", "weak_reference")

DEFAULT_CAMERA = "The camera holds a static shot on the subject."

#: Label-style motions rewritten as natural verb phrases, as the guide requires.
MOTION_PHRASES: dict[str, str] = {
    "zoom in": "zooms in",
    "zoom out": "zooms out",
    "push in": "pushes in",
    "pull out": "pulls out",
    "pan left": "pans left",
    "pan right": "pans right",
    "truck left": "trucks left",
    "truck right": "trucks right",
    "tilt up": "tilts up",
    "tilt down": "tilts down",
    "pedestal up": "rises on the pedestal",
    "pedestal down": "lowers on the pedestal",
    "arc shot": "moves in an arc around the subject",
    "tracking shot": "tracks the subject",
    "static shot": "holds a static shot",
    "shake slightly": "shakes slightly",
    "shake strongly": "shakes strongly",
    "pov": "takes the subject's point of view",
    "roll clockwise": "rolls clockwise",
    "roll counterclockwise": "rolls counterclockwise",
}

#: Field names models like to echo back into their own answers.
FIELD_LABELS: tuple[str, ...] = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
    "subject_definitions",
    "retention_analysis",
    "detailed_description",
    "summary",
)

#: Schema field names models sometimes prepend to their own values
#: ("Dialogue_mode: sung" ending up inside the speaker description).
SHOT_FIELD_LABELS: tuple[str, ...] = FIELD_LABELS + (
    "shot",
    "subjects",
    "opening",
    "action",
    "camera",
    "diegetic_sound",
    "soundscape",
    "music",
    "speaker",
    "dialogue",
    "dialogue_mode",
    "on_screen_text",
    "negative_extra",
    "style",
)

def _label_alternation(labels: tuple[str, ...]) -> str:
    """Match a field name however the model spelled it: underscore, space or hyphen."""
    return "|".join(r"[_\s-]*".join(re.escape(part) for part in label.split("_")) for label in labels)


_LEADING_LABEL_RE = re.compile(r"^\s*(?:" + _label_alternation(SHOT_FIELD_LABELS) + r")\s*:\s*", re.IGNORECASE)
_EMBEDDED_LABEL_RE = re.compile(
    r"(?:^|[\s;.,])(?:" + _label_alternation(FIELD_LABELS) + r")\s*:", re.IGNORECASE
)
#: enum-like fields whose value is noise inside prose ("Dialogue_mode: sung")
_ENUM_PAIR_RE = re.compile(
    r"\b(?:" + _label_alternation(("dialogue_mode", "dialogue", "shot", "subjects")) + r")\s*:\s*\S+\s*",
    re.IGNORECASE,
)
_LEADING_MODE_RE = re.compile(r"^(?:sung|spoken|voiceover|none)\b[\s,:;-]*", re.IGNORECASE)

_LABEL_RE = re.compile(r"<(Subject|Picture|Video|Audio)\s+(\d+)>")


def _motion_pattern(motion: str) -> re.Pattern[str]:
    """Match a camera motion in any conjugation ("Push In" -> "pushes in")."""
    words = motion.lower().split()
    stem = words[0][:-1] if words[0].endswith("e") else words[0]
    parts = [stem + "[a-z]*"] + [re.escape(word) for word in words[1:]]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b")


_CAMERA_PATTERNS = tuple(_motion_pattern(motion) for motion in CAMERA_MOTIONS)


class H3FormatError(ValueError):
    """Raised when a rendered prompt would violate the H3 format rules."""


# --------------------------------------------------------------------------- data


@dataclass
class Speaker:
    """One vocal performance inside a shot."""

    description: str = "a voice"
    line: str = ""
    language: str = "English"
    mode: str = "spoken"  # spoken | sung | voiceover
    on_screen: bool = True

    def is_valid(self) -> bool:
        return bool(self.line and self.line.strip())


@dataclass
class Subject:
    """A recurring element that must stay consistent across shots."""

    name: str = "subject"
    kind: str = "character"  # character | location | prop | vehicle | style
    description: str = ""
    identity_lock: str = ""
    retention: str = "fully_preserved"
    #: 1-based position of the reference image that defines this subject, in the same
    #: order the images are sent to the model. 0 means no image was supplied.
    picture: int = 0

    def definition(self, label: str, picture_label: str = "") -> str:
        """One ``subject_definitions`` line.

        When a reference image defines this subject, the guide requires the image to be
        cited *inside* the subject's definition rather than declared on its own - that
        citation is the only thing that tells the model which picture is which subject.
        Without it the images are supplied but never bound to anything.
        """
        body = _lower_first(strip_field_labels(self.description) or self.name)
        lock = strip_field_labels(self.identity_lock)
        head = f"{label} is {body}"
        if picture_label:
            head = f"{head}, whose appearance comes from {picture_label}"
        text = _end_sentence(head, capitalize=False)
        if lock:
            text = f"{text} Key features: {_end_sentence(lock, capitalize=False)}"
        return text


@dataclass
class Cut:
    """An additional cut inside a single generated clip."""

    time: float = 0.0
    description: str = ""
    phrase: str = "the camera cuts to"


@dataclass
class H3Shot:
    """Everything needed to render one 5-15 s MiniMax H3 clip."""

    index: int = 1
    duration: float = 6.0
    style: str = "Live-action, cinematic"
    opening: str = ""
    action: str = ""
    camera: str = ""
    diegetic_sound: str = ""
    on_screen_text: list[str] = field(default_factory=list)
    soundscape: str = ""
    music: str = "N/A"
    speakers: list[Speaker] = field(default_factory=list)
    subjects: list[Subject] = field(default_factory=list)
    cuts: list[Cut] = field(default_factory=list)
    extra_style_directive: str = ""
    #: What the supplied audio clip is, when one is sent with the shot. Empty means no
    #: audio reference, and no ``<Audio 1>`` label is written.
    audio_reference: str = ""
    audio_state: str = "fully_copy"


# --------------------------------------------------------------------------- text helpers


def _collapse(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()


def strip_field_labels(text: str) -> str:
    """Remove field names a model echoed into its own answer."""
    cleaned = _collapse(text).replace("\n", " ")
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _LEADING_LABEL_RE.sub("", cleaned)
    match = _EMBEDDED_LABEL_RE.search(cleaned)
    if match:
        cleaned = cleaned[: match.start()]
    return cleaned.strip(" ;,")


def _end_sentence(text: str, capitalize: bool = True) -> str:
    text = _collapse(text)
    if not text:
        return ""
    if capitalize and text[0].isalpha() and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1] in ".!?":
        return text
    return text + "."


def _lower_first(text: str) -> str:
    """Lowercase the first letter, leaving acronyms such as "POV" alone."""
    text = _collapse(text)
    if not text or not text[0].isalpha():
        return text
    is_acronym = len(text) > 1 and text[0].isupper() and text[1].isalpha() and text[1].isupper()
    return text if is_acronym else text[0].lower() + text[1:]


def _sentences(*parts: str) -> str:
    """Join fragments into one paragraph, each closed with a full stop."""
    out = [_end_sentence(strip_field_labels(part)) for part in parts]
    return " ".join(part for part in out if part)


def format_timecode(seconds: float) -> str:
    """Format seconds as the H3 cut notation ``MM:SS.mmm``."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:06.3f}"


def normalize_camera(text: str) -> str:
    """Ensure the camera line is a natural sentence using the H3 motion vocabulary."""
    cleaned = strip_field_labels(text)
    if not cleaned:
        return DEFAULT_CAMERA
    lowered = cleaned.lower()
    has_motion = any(pattern.search(lowered) for pattern in _CAMERA_PATTERNS)

    if has_motion and "camera" in lowered:
        return _end_sentence(cleaned)
    if has_motion:
        # label style ("Tracking Shot with large amplitude at slow speed") -> a sentence
        for motion, phrase in MOTION_PHRASES.items():
            if lowered.startswith(motion):
                remainder = cleaned[len(motion) :].strip(" ,")
                return _end_sentence(f"The camera {phrase} {remainder}".strip())
        return _end_sentence(f"The camera moves: {cleaned}")
    if "camera" in lowered:
        return _end_sentence(cleaned + ", holding a static shot")
    return _end_sentence("The camera holds a static shot while " + _lower_first(cleaned))


def render_dialogue(speakers: list[Speaker]) -> str:
    """Render speaker lines with stable ``(S1)`` ids and ``<d>`` blocks."""
    fragments: list[str] = []
    for position, speaker in enumerate([s for s in speakers if s.is_valid()], start=1):
        sid = f"(S{position})"
        who = strip_field_labels(_ENUM_PAIR_RE.sub("", _collapse(speaker.description)))
        who = _LEADING_MODE_RE.sub("", who).strip(" ,;") or "a voice"
        language = _collapse(speaker.language) or "English"
        line = _collapse(speaker.line)
        block = f"<d>[{language}] {line}</d>"
        mode = (speaker.mode or "spoken").lower()
        if mode == "voiceover":
            fragments.append(
                f"{who.capitalize()} {sid} says in an off-screen voiceover: {block} "
                "while their lips remain completely closed."
            )
        elif mode in {"sung", "sings", "singing"}:
            fragments.append(f"{who.capitalize()} {sid} sings: {block}")
        else:
            fragments.append(f"{who.capitalize()} {sid} says: {block}")
    return " ".join(fragments)


def render_on_screen_text(items: list[str]) -> str:
    quoted = [f'"{_collapse(item)}"' for item in items if _collapse(item)]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return f"The text {quoted[0]} is visible in the frame."
    return "The texts " + ", ".join(quoted) + " are visible in the frame."


def _render_cuts(cuts: list[Cut], duration: float, start_index: int) -> str:
    fragments: list[str] = []
    last_time = 0.0
    shot_number = start_index
    for cut in sorted(cuts, key=lambda c: float(c.time or 0.0)):
        time = float(cut.time or 0.0)
        if time <= last_time or time >= float(duration):
            continue
        description = _collapse(cut.description)
        if not description:
            continue
        phrase = cut.phrase if cut.phrase in CUT_PHRASES else CUT_PHRASES[0]
        shot_number += 1
        last_time = time
        fragments.append(
            f"[Shot {shot_number}] At {format_timecode(time)}, {phrase} "
            f"{_end_sentence(_lower_first(description), capitalize=False)}"
        )
    return " ".join(fragments)


def _soundscape(shot: H3Shot) -> str:
    """H3 allows N/A here only for deliberate silence, so always fall back to prose."""
    text = strip_field_labels(shot.soundscape)
    if not text or text.upper().strip(".") == "N/A":
        text = strip_field_labels(shot.diegetic_sound)
    if not text or text.upper().strip(".") == "N/A":
        text = "Room tone continues quietly under the action with the natural sounds of the scene."
    return _end_sentence(text)


def _music(shot: H3Shot) -> str:
    text = strip_field_labels(shot.music)
    if not text or text.upper().strip(".") == "N/A":
        return "N/A"
    return _end_sentence(text)


def _style_line(shot: H3Shot) -> str:
    style = strip_field_labels(shot.style).rstrip(".") or "Live-action, cinematic"
    extra = strip_field_labels(shot.extra_style_directive).rstrip(".")
    return f"{style}, {extra}" if extra else style


# --------------------------------------------------------------------------- I2VA


def render_i2va(shot: H3Shot) -> str:
    """Render an image-to-video prompt whose first frame is the generated start frame."""
    opening = _lower_first(
        strip_field_labels(shot.opening) or "the framing established by <Picture 1> is preserved"
    )
    body = _end_sentence(f"[Shot 1] {_style_line(shot)}, {opening}")
    body = _sentences(
        body,
        "Character identity, clothing, colors, key objects and spatial relationships from <Picture 1> remain consistent",
        normalize_camera(shot.camera),
        shot.action,
        render_dialogue(shot.speakers),
        shot.diegetic_sound,
        render_on_screen_text(shot.on_screen_text),
    )
    cuts = _render_cuts(shot.cuts, shot.duration, start_index=1)
    if cuts:
        body = f"{body} {cuts}"

    return (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.\n"
        "\n"
        f"integrated_multimodal_description: {body}\n"
        "\n"
        f"overall_soundscape: {_soundscape(shot)}\n"
        "\n"
        f"non_diegetic_music: {_music(shot)}"
    )


# --------------------------------------------------------------------------- Ref2VA


def render_ref2va(shot: H3Shot, strict: bool = False) -> str:
    """Render a full-reference prompt; subjects are renumbered locally per shot."""
    subjects = [s for s in shot.subjects if (s.name or s.description)]
    labels = {id(subject): f"<Subject {position}>" for position, subject in enumerate(subjects, start=1)}
    pictures = {
        id(subject): f"<Picture {int(subject.picture)}>"
        for subject in subjects
        if int(subject.picture or 0) > 0
    }

    definitions = [
        subject.definition(labels[id(subject)], pictures.get(id(subject), "")) for subject in subjects
    ]
    if not definitions:
        definitions = [
            "<Subject 1> is the main on-screen subject described in the detailed description below."
        ]
        subjects = [Subject(name="main subject", description="the main on-screen subject")]
        labels = {id(subjects[0]): "<Subject 1>"}
        pictures = {}

    speaking = render_dialogue(shot.speakers)
    audio_reference = _collapse(shot.audio_reference)
    if audio_reference:
        # The audio is what the performer must mouth, so it is bound to the first subject
        # and to speaker S1 - the guide's way of saying "these lips follow this signal".
        voice = f", and the voice-timbre reference for {labels[id(subjects[0])]} (S1)" if speaking else ""
        definitions.append(
            _end_sentence(f"<Audio 1> is {_lower_first(audio_reference)}{voice}", capitalize=False)
        )

    label_list = ", ".join(labels[id(subject)] for subject in subjects)
    action_line = _collapse(shot.action) or _collapse(shot.opening) or "the referenced subjects stay in motion"
    summary = (
        f"[reference generation] The target video reuses {label_list} from the supplied references. "
        + _end_sentence(action_line)
    )

    retention_lines = []
    for subject in subjects:
        state = subject.retention if subject.retention in RETENTION_STATES else "fully_preserved"
        detail = _collapse(subject.identity_lock or subject.description or subject.name)
        retention_lines.append(
            f"{labels[id(subject)]} (appears in [Shot 1]): {state} - {detail} is retained."
        )

    if audio_reference:
        state = shot.audio_state if shot.audio_state in AUDIO_STATES else "fully_copy"
        synced = (
            f" {labels[id(subjects[0])]} mouths it in exact sync, frame by frame."
            if speaking
            else ""
        )
        retention_lines.append(
            f"<Audio 1>: {state} - <Audio 1> is reused 1:1 as the target video's "
            f"complete final audio track.{synced}"
        )

    described = _sentences(
        _collapse(shot.opening) or f"{label_list} establish the opening composition",
        normalize_camera(shot.camera),
        shot.action,
        speaking,
        shot.diegetic_sound,
        render_on_screen_text(shot.on_screen_text),
    )
    detailed = f"The target video uses a {_style_line(shot).lower()} look.\n[Shot 1] {described}"
    cuts = _render_cuts(shot.cuts, shot.duration, start_index=1)
    if cuts:
        detailed = f"{detailed} {cuts}"

    text = (
        "subject_definitions:\n"
        + "\n".join(definitions)
        + "\n\nsummary:\n"
        + summary
        + "\n\nretention_analysis:\n"
        + "\n".join(retention_lines)
        + "\n\ndetailed_description:\n"
        + detailed
        + "\n\noverall_soundscape:\n"
        + _soundscape(shot)
        + "\n\nnon_diegetic_music:\n"
        + _music(shot)
    )
    defined = set(labels.values()) | set(pictures.values())
    if audio_reference:
        defined.add("<Audio 1>")
    return _resolve_labels(text, defined=defined, strict=strict)


def _resolve_labels(text: str, defined: set[str], strict: bool) -> str:
    """Drop or reject reference labels that were never defined."""
    undefined = {match.group(0) for match in _LABEL_RE.finditer(text)} - set(defined)
    if not undefined:
        return text
    if strict:
        raise H3FormatError(f"undefined reference labels: {sorted(undefined)}")
    for label in undefined:
        text = text.replace(label + " ", "").replace(label, "")
    return re.sub(r"[ \t]{2,}", " ", text)


def undefined_labels(text: str, defined: set[str]) -> set[str]:
    """Public helper used by tests and runtime validation."""
    return {match.group(0) for match in _LABEL_RE.finditer(text)} - set(defined)
