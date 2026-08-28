"""A stand-in for LMStudioClient that answers every stage with plausible JSON.

Kept free of pytest imports so it can also be used from plain scripts run with
ComfyUI's own python (which has torch but usually no pytest).
"""

from __future__ import annotations

import json

SUBJECTS = [
    {
        "name": "the singer",
        "kind": "character",
        "description": "a young singer in a flooded parking garage",
        "identity_lock": "shaved head, oversized silver raincoat, cracked LED microphone",
        "reference_prompt_hint": "full body, neutral backdrop",
    },
    {
        "name": "the garage",
        "kind": "location",
        "description": "a flooded underground parking garage",
        "identity_lock": "ankle-deep water, sodium vapour lamps, peeling yellow pillars",
        "reference_prompt_hint": "wide empty establishing view",
    },
]


class FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[str] = []

    def ensure_model(self, *args, **kwargs):
        return None

    def unload(self, *args, **kwargs):
        return None

    def chat_json(self, model, system, user, schema=None, images=None, stage="stage", **kwargs):
        self.calls.append(stage)
        properties = (schema or {}).get("properties", {})
        if "genre" in properties:
            return {
                "genre": "industrial pop",
                "mood": "cold and defiant",
                "themes": ["water", "neon", "escape"],
                "narrative_arc": "A singer wades through a flooded garage and finds the exit ramp.",
                "lyrics_language": "English",
                "summary": "A defiant night-time escape.",
            }
        if "visual_style" in properties:
            return {
                "visual_style": "Live-action, grainy 16mm cinematic",
                "color_palette": ["sodium orange", "deep teal", "wet concrete grey"],
                "lighting": "Hard sodium vapour lamps with long reflections on standing water.",
                "lens_and_texture": "35mm anamorphic, heavy grain, soft halation.",
                "camera_language": "Slow deliberate moves broken by handheld bursts.",
                "world": "An abandoned flooded parking structure at 3 a.m.",
                "negative_extra": ["daylight", "clean dry floor"],
            }
        if "subjects" in properties:
            return {"subjects": SUBJECTS}
        if "shots" in properties:
            payload = json.loads(user.split("SHOTS TO WRITE (timing is fixed):\n")[1].split("\n\n")[0])
            return {
                "shots": [
                    {
                        "shot": entry["shot"],
                        "subjects": ["the singer", "the garage"],
                        "opening": "a low wide shot frames the singer ankle-deep in black water",
                        "action": "she steps forward and the reflection breaks apart around her boots",
                        "camera": "The camera pushes in with small amplitude at slow speed",
                        "diegetic_sound": "Water sloshes around her boots",
                        "soundscape": "Dripping water echoes across the concrete deck under a low electrical hum.",
                        "music": "N/A",
                        "speaker": "a young singer with a hoarse low voice",
                        "dialogue": entry.get("lyrics_in_shot") or "",
                        "dialogue_mode": "sung" if entry.get("lyrics_in_shot") else "none",
                        "on_screen_text": "",
                        "negative_extra": "no daylight",
                    }
                    for entry in payload
                ]
            }
        if "prompts" in properties and "SHOTS:\n" in user:
            payload = json.loads(user.split("SHOTS:\n")[1].split("\n\n")[0])
            return {
                "prompts": [
                    {"shot": entry["shot"], "prompt": f"cinematic still of shot {entry['shot']}, flooded garage"}
                    for entry in payload
                ]
            }
        if "prompts" in properties:
            return {"prompts": [{"name": s["name"], "prompt": f"reference sheet of {s['name']}"} for s in SUBJECTS]}
        raise AssertionError(f"unexpected schema for stage {stage}")
