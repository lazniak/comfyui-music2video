"""The Music2Video node: audio in, prompts out - and, optionally, rendered media."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from comfy_api.latest import io

from . import asr as asr_module
from . import audio_io
from . import cost as cost_module
from . import model_cache
from . import music_dsp
from . import pipe as pipe_module
from . import preview as preview_module
from . import render as render_module
from . import video as video_module
from .h3_format import H3Shot, Speaker, Subject, render_i2va, render_ref2va
from .llm_stages import StageRunner, load_h3_guide
from .lmstudio import DEFAULT_URL, FALLBACK_MODELS, LMStudioClient
from .providers import LLM_PROVIDERS, make_llm_client
from .render import MEDIA_PROVIDERS, ImageRequest, VideoRequest
from .video import AUDIO_MODES, FIT_MODES
from .shots import ShotSlot, attach_lyrics, plan_shots
from .util import (
    PREFIX,
    as_list,
    audio_to_mono,
    first_str,
    image_tensor_to_data_uri,
    log,
    slice_audio,
    warn,
)

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
DEVICES = ["auto", "cuda:0", "cuda:1", "cpu"]
VIDEO_PROMPT_SOURCES = ["i2va", "ref2va"]
DEFAULT_NEGATIVE = (
    "blurry, low quality, watermark, signature, text artifacts, deformed hands, extra limbs, "
    "oversaturated colors, jpeg artifacts, plastic skin"
)
BASE_STAGES = 7


def _schema_options() -> dict[str, list[str]]:
    """Dropdown contents, read from the shared cache.

    ComfyUI re-runs ``define_schema`` on every /object_info request and twice per
    queued prompt, so this must not touch the network. The cache is filled by the
    background warm-up and by the pack's HTTP route; the lists never shrink, so a
    model picked in a saved workflow stays selectable.
    """
    return {kind: model_cache.snapshot(kind) or ["(none found)"] for kind in model_cache.KINDS}


def _interrupt_check() -> None:
    try:
        import comfy.model_management as mm  # type: ignore

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


class Music2PromptsLM(io.ComfyNode):
    """Turn a track into image prompts and MiniMax H3 video prompts, fully locally."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        options = _schema_options()
        return io.Schema(
            node_id="Music2PromptsLM",
            display_name="🎵 Music2Video",
            category="Music2Video",
            description=(
                "One node from a track to a finished film. It transcribes the lyrics with word-level "
                "timing (Whisper large-v3) and reads the BPM, the beat grid and the sections "
                "(librosa, or a built-in numpy/scipy fallback), then has an LLM - LM Studio locally by "
                "default, or OpenRouter / OpenAI / Anthropic - write the treatment, the art direction, "
                "a bible of the recurring subjects and every shot, with the cuts snapped to the beat.\n\n"
                "Out come start-frame image prompts, subject reference-sheet prompts, MiniMax H3 video "
                "prompts (I2VA and Ref2VA, in their exact six-section format), negatives, per-shot "
                "timings and sample-accurate audio slices for lipsync - all on one 'pipe', which "
                "'Music2Video Pipe Expand' takes apart wherever a value is needed.\n\n"
                "It can also render: start frames and clips through fal.ai or OpenRouter, then every "
                "clip trimmed to its shot and cut together under the original music with PyAV. Each "
                "frame and clip appears in the node as it lands, and the cost meter below the gallery "
                "reports in USD what every model actually billed.\n\n"
                "The analysis is always local and free. Rendering is off until you pick a provider, "
                "because those providers bill per call."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip=(
                        "The track. Its length is what the shot plan is divided out of; "
                        "librosa analyses a mono copy at the original sample rate, Whisper "
                        "gets a mono copy resampled to 16 kHz, and the per-shot slices on "
                        "'audio_clips' keep the original sample rate and channels. The same "
                        "track is muxed under the finished film when 'final_audio' is 'music'."
                    ),
                ),
                io.String.Input(
                    "instruction",
                    multiline=True,
                    default="Cinematic music video. Describe the mood, story or visual world you want.",
                    tooltip=(
                        "Your brief: story, mood, world, constraints. It is pasted verbatim "
                        "into four stages - the interpretation, the art direction, the subject "
                        "bible and every shot-content batch - so it steers the whole film. The "
                        "image-prompt stage never sees it; what reaches the frames is whatever "
                        "those stages made of it."
                    ),
                ),
                io.Combo.Input(
                    "llm_provider",
                    options=list(LLM_PROVIDERS),
                    default="lmstudio",
                    tooltip=(
                        "Who writes the prompts. 'lmstudio' runs locally and is free; the "
                        "other three bill per token, and one run is at least four requests "
                        "plus two per batch of 'shots_per_request' shots. Switching away from "
                        "'lmstudio' hides that provider's server widgets ('lm_url', "
                        "'lm_api_key', 'lm_context_length', the auto-load switches) along with "
                        "the other providers' model and key widgets, but 'lm_max_tokens', "
                        "'lm_timeout' and 'lm_retries' still apply to the cloud providers, and "
                        "'lm_temperature' and 'lm_reasoning_effort' apply to all of them "
                        "except anthropic, whose request carries neither."
                    ),
                ),
                io.Combo.Input(
                    "lm_model",
                    options=options["lmstudio"],
                    tooltip=(
                        "Model served by LM Studio, used only when 'llm_provider' is "
                        "'lmstudio'. The list is read from the running server; when the server "
                        "was unreachable the dropdown falls back to a single built-in id "
                        "('google/gemma-4-e4b') that you may not have installed - start LM "
                        "Studio, then right-click the node and choose 'Refresh model lists', "
                        "or type the key into 'lm_model_override', which wins over this "
                        "dropdown. With 'lm_auto_download' and 'lm_auto_load' on, the node "
                        "installs and loads it itself."
                    ),
                ),
                io.Combo.Input(
                    "openrouter_model", options=options["openrouter_llm"],
                    tooltip=(
                        "Model used when 'llm_provider' is 'openrouter'; billed per token. The "
                        "list is OpenRouter's public catalogue filtered to text-output models, "
                        "so it fills in without a key. If you wire 'reference_images', pick "
                        "one that accepts image input - the description stage sends them as "
                        "images, and a model that refuses only produces a warning and no "
                        "description."
                    ),
                ),
                io.Combo.Input(
                    "openai_model", options=options["openai_llm"],
                    tooltip=(
                        "Model used when 'llm_provider' is 'openai'; billed per token. Until a "
                        "key is found in the environment the dropdown shows only a built-in "
                        "fallback list ('gpt-5.2', 'gpt-5.1', 'gpt-5-mini', 'gpt-4.1'); the "
                        "live catalogue - filtered to gpt / o1 / o3 / o4 / chatgpt chat ids - "
                        "appears once OPENAI_API_KEY is set in the environment and you "
                        "right-click the node and choose 'Refresh model lists'. The "
                        "'openai_api_key' widget is used for the run itself, not for this "
                        "probe. Structured stages are sent as strict json_schema responses; "
                        "the final retry drops the schema and parses the reply loosely."
                    ),
                ),
                io.Combo.Input(
                    "anthropic_model", options=options["anthropic_llm"],
                    tooltip=(
                        "Model used when 'llm_provider' is 'anthropic'. Structured stages go "
                        "through forced tool use; temperature is not sent because current "
                        "Claude models reject it, so 'lm_temperature' does nothing here. Until "
                        "ANTHROPIC_API_KEY is set in the environment the dropdown shows only a "
                        "built-in fallback list; the live catalogue appears after you set that "
                        "variable and choose 'Refresh model lists' from the node's right-click "
                        "menu. The 'anthropic_api_key' widget is used for the run, not for "
                        "this probe."
                    ),
                ),
                io.String.Input(
                    "visual_style",
                    default="",
                    tooltip=(
                        "Forces a look, e.g. 'grainy 16mm night photography'. It reaches the "
                        "art-direction stage only, as a hard requirement; empty lets that "
                        "stage choose. Every later stage sees the art direction it produced, "
                        "not this text, so a short entry gets elaborated rather than pasted "
                        "into the prompts."
                    ),
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=ASPECT_RATIOS,
                    default="16:9",
                    tooltip=(
                        "Framing for the image prompts and, when rendering, for the payload: "
                        "fal endpoints get a pixel size derived from it (1024-based, snapped "
                        "to multiples of 32) or the nearest named preset, OpenRouter gets the "
                        "string as it stands. A value the endpoint does not declare is dropped "
                        "rather than substituted, and on preset-only endpoints 21:9 collapses "
                        "to landscape_16_9. On a clip that goes out with a start frame, the "
                        "aspect is not sent at all - the frame decides it."
                    ),
                ),
                io.Float.Input(
                    "clip_seconds",
                    default=6.0,
                    min=1.0,
                    max=60.0,
                    step=0.5,
                    tooltip=(
                        "Target length of one shot before pacing: 'dynamicity' scales it from "
                        "1.3x (at 0) to 0.7x (at 1), the result is clamped into "
                        "'min_shot_seconds'..'max_shot_seconds', and the shot count is the "
                        "track length divided by that. It is ignored completely when "
                        "'num_shots' is anything other than 0."
                    ),
                ),
                io.Float.Input(
                    "min_shot_seconds", default=5.0, min=0.5, max=60.0, step=0.5,
                    tooltip=(
                        "Shortest shot the planner will produce. It also caps the shot count "
                        "at track length / this value, so it silently overrules a larger "
                        "'num_shots', and a track shorter than it comes back as one single "
                        "shot. Shots under 2 s are refused by every audio field this node can "
                        "fill, so their slice goes out without the vocal."
                    ),
                ),
                io.Float.Input(
                    "max_shot_seconds", default=15.0, min=1.0, max=60.0, step=0.5,
                    tooltip=(
                        "Longest shot allowed: the planner adds shots until none is longer, "
                        "then splits any span that still exceeds it. Each shot's duration is "
                        "what the video endpoint is asked for, clamped to the range or enum "
                        "that endpoint declares - so a shot longer than the model allows comes "
                        "back short, and 'concat_video' holds its last frame to fill the gap. "
                        "Past 15 s a shot also loses its audio slice on endpoints that take "
                        "reference audio as a list field, so lipsync stops for it."
                    ),
                ),
                io.Int.Input(
                    "num_shots",
                    default=0,
                    min=0,
                    max=400,
                    tooltip=(
                        "0 derives the count from the track length, 'clip_seconds' and "
                        "'dynamicity'. Any other value asks for exactly that many and makes "
                        "'clip_seconds' irrelevant, but it is still clamped: never more than "
                        "track length / 'min_shot_seconds', and shots are added back if that "
                        "would push any of them past 'max_shot_seconds'. Every shot is one LLM "
                        "shot-content entry and, when rendering, one paid image and one paid "
                        "clip."
                    ),
                ),
                io.Float.Input(
                    "creativity", default=0.7, min=0.0, max=1.0, step=0.05,
                    tooltip=(
                        "0 = grounded and literal, 1 = bold and surreal. It goes to the "
                        "art-direction stage alone, printed into its brief as a number; it is "
                        "not a sampling parameter - 'lm_temperature' is the one that changes "
                        "decoding - and it reaches the shots only through the art direction it "
                        "produced."
                    ),
                ),
                io.Float.Input(
                    "dynamicity", default=0.6, min=0.0, max=1.0, step=0.05,
                    tooltip=(
                        "Two effects: it scales 'clip_seconds' by 1.3 at 0 down to 0.7 at 1 "
                        "while the shot count is being derived, and it is printed into every "
                        "shot-content request as the pacing level (0 = calm and static, 1 = "
                        "restless and kinetic). The scaled value is clamped into "
                        "'min_shot_seconds'..'max_shot_seconds' first, so at the defaults (6 s "
                        "target, 5 s floor) anything above about 0.78 no longer shortens the "
                        "shots - only the pacing wording changes. With 'num_shots' set, only "
                        "the wording effect remains."
                    ),
                ),
                io.Float.Input(
                    "word_influence", default=0.6, min=-1.0, max=1.0, step=0.1,
                    tooltip=(
                        "+1 = visualise the lyrics literally, -1 = ignore the words and use "
                        "the vibe. The interpretation stage only sees three bands - above "
                        "+0.33, below -0.33, and everything between reads identically - while "
                        "the exact number is printed into every shot-content request. With "
                        "'whisper_skip' on or an instrumental track there are no words to "
                        "weight."
                    ),
                ),
                io.Combo.Input(
                    "whisper_device", options=DEVICES, default="auto",
                    tooltip=(
                        "Where Whisper runs. 'auto' takes cuda:0 when CUDA is present and CPU "
                        "otherwise; a cuda index that does not exist falls back to cuda:0 with "
                        "a warning, and CPU forces float32 whatever 'whisper_dtype' says. "
                        "Pointing it at a second card is the alternative to the unloads run "
                        "before transcription - 'free_comfy_vram', and 'free_lmstudio_vram' "
                        "when the LLM is LM Studio."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip=(
                        "0 means no seed is sent at all - not seed zero - so the LLM and every "
                        "render run unseeded. Any other value is applied per item, not per "
                        "run: shot 1 gets the seed itself, shot 2 seed+1 and so on; the first "
                        "subject sheet gets seed+1000; each clip uses the same offset as its "
                        "shot. Everything is folded into the 32-bit range providers accept "
                        "(abs(seed) modulo 2^31), and fal folds it again into whatever range "
                        "the endpoint declares, so distant seeds can land on the same value. "
                        "The LLM seed reaches LM Studio, OpenAI and OpenRouter; the Anthropic "
                        "request carries no seed field, so on that provider the prompt writing "
                        "is unseeded whatever you set here."
                    ),
                ),
                io.Combo.Input(
                    "image_provider",
                    options=list(MEDIA_PROVIDERS),
                    default="none",
                    tooltip=(
                        "Where the start frames are rendered. 'none' skips the images: nothing "
                        "is billed for them and the shots go out as prompts only. 'fal' and "
                        "'openrouter' are paid per-call APIs - one image per shot, plus one "
                        "per subject when 'render_subject_sheets' is on. Setting it also "
                        "unhides the render widgets, and a fal video model whose schema "
                        "requires an image aborts the run at once while this is 'none', before "
                        "anything is billed. Neither this switch nor 'video_provider' affects "
                        "the LLM: a cloud 'llm_provider' still bills per token."
                    ),
                ),
                io.Combo.Input(
                    "fal_image_model",
                    options=options["fal_image"],
                    tooltip=(
                        "The plain text-to-image fal endpoint. It always draws the subject "
                        "sheets, and it draws the shots as well whenever there is nothing to "
                        "reference (no subject sheets, no 'reference_images'); once references "
                        "exist the shots move to 'fal_image_edit_model'. The list is fal's "
                        "text-to-image and image-to-image index combined, so edit endpoints "
                        "appear in it too - one picked here has nothing to edit unless you "
                        "wired 'reference_images'."
                    ),
                ),
                io.Combo.Input(
                    "fal_image_edit_model",
                    options=options["fal_image"],
                    tooltip=(
                        "fal image model used for the shots once there is an identity to hold "
                        "on to - the subject sheets, or your 'reference_images'; with neither "
                        "present it is never consulted and 'fal_image_model' draws everything. "
                        "The style anchor is added to what this model receives, but it never "
                        "selects it on its own. This has to be an endpoint that accepts a "
                        "reference image: the node reads the endpoint's schema and warns when "
                        "it declares no field for one, and a plain text-to-image model "
                        "silently throws the identity away so every shot comes back a "
                        "different person. Edit endpoints usually carry 'edit', 'kontext' or "
                        "'image-to-image' in the id, but the schema, not the name, is what "
                        "decides."
                    ),
                ),
                io.Combo.Input(
                    "openrouter_image_model",
                    options=options["openrouter_image"],
                    tooltip=(
                        "Image model used when 'image_provider' is 'openrouter'. The node "
                        "always sends the subject sheets and the style anchor as "
                        "'input_references' on this provider - there is no OpenRouter "
                        "counterpart to 'fal_image_edit_model' and no capability check, so a "
                        "model that ignores references simply loses the identity. The request "
                        "carries prompt, aspect ratio, seed and references: negatives "
                        "('negative_prompt_base' and the per-shot extras) are not part of that "
                        "API and are dropped."
                    ),
                ),
                io.Combo.Input(
                    "video_provider",
                    options=list(MEDIA_PROVIDERS),
                    default="none",
                    tooltip=(
                        "Where the clips are rendered. 'none' skips the clips: nothing is "
                        "billed for them. 'fal' and 'openrouter' are paid per-call APIs. Only "
                        "'fal' is checked against the endpoint's published schema before "
                        "submitting - the OpenRouter payload is fixed (prompt, duration, "
                        "aspect ratio, seed, and either a first frame or references), and it "
                        "has no input for a driving audio track at all, so 'lipsync_audio' "
                        "cannot work there. Neither this switch nor 'image_provider' affects "
                        "the LLM: a cloud 'llm_provider' still bills per token."
                    ),
                ),
                io.Combo.Input(
                    "fal_video_model",
                    options=options["fal_video"],
                    tooltip=(
                        "Video model used when 'video_provider' is 'fal'. Its published schema "
                        "decides most of the payload: the shot duration is clamped to the "
                        "range or enum it declares, and 'lipsync_audio' only does something if "
                        "it declares a real audio input. The first-frame/reference choice "
                        "comes from 'video_prompt_source' plus the id - with 'ref2va' "
                        "selected, an id containing 'reference-to-video' is sent the subject "
                        "references instead of a first frame. If the schema marks an image "
                        "field required while no images are being rendered, the run stops "
                        "before anything is billed."
                    ),
                ),
                io.Combo.Input(
                    "openrouter_video_model",
                    options=options["openrouter_video"],
                    tooltip=(
                        "Video model used when 'video_provider' is 'openrouter'. The payload "
                        "is fixed and unverified - prompt, duration rounded to whole seconds, "
                        "aspect ratio, seed, and either a first frame or references - so a "
                        "model that expects anything else simply fails. 'lipsync_audio' and "
                        "'prompt_expansion' have no field to go into on this API and are "
                        "ignored."
                    ),
                ),
                io.Int.Input(
                    "render_concurrency", default=2, min=1, max=16,
                    tooltip=(
                        "How many renders are in flight at once within one pass - the subject "
                        "sheets, then the start frames, then the clips - never across passes. "
                        "1 runs them in a plain loop; when 'style_anchor' is on and there is "
                        "something to reference, shot 1 is rendered alone first whatever this "
                        "is set to. The limit that bites is your provider's own concurrency "
                        "allowance rather than the 16 here. Failures are per item: a failed "
                        "image keeps its slot as a black frame, a failed clip is dropped from "
                        "'videos' and from the final cut."
                    ),
                ),
                # ------------------------------------------------------------------ advanced
                io.String.Input(
                    "lm_url", default=DEFAULT_URL, advanced=True,
                    tooltip=(
                        "Address of the LM Studio server: the model list, the load/unload "
                        "calls and the completions all go here. Only used when 'llm_provider' "
                        "is lmstudio - the three cloud providers have fixed endpoints and this "
                        "widget is hidden for them. Default http://127.0.0.1:1234. Nothing "
                        "here is contacted until the model-preparation step, which runs after "
                        "transcription, so an unreachable server fails the run minutes in with "
                        "'LM Studio unreachable' rather than immediately (start it under "
                        "Developer -> Start Server)."
                    ),
                ),
                io.String.Input(
                    "lm_api_key", default="", advanced=True,
                    tooltip=(
                        "Bearer token for LM Studio, sent only when this field is non-empty. "
                        "LM Studio needs none by default - fill it in only if you put the "
                        "server behind a proxy that demands one. Unlike the four cloud key "
                        "fields, this one has no environment-variable fallback: empty means no "
                        "Authorization header at all."
                    ),
                ),
                io.String.Input(
                    "lm_model_override", default="", advanced=True,
                    tooltip=(
                        "Model key sent instead of the 'lm_model' dropdown, e.g. "
                        "google/gemma-4-e4b. Use it when LM Studio was offline while ComfyUI "
                        "built the dropdown, so the model you want was never in the list. "
                        "Ignored for every provider other than lmstudio - "
                        "openrouter/openai/anthropic always use their own dropdown."
                    ),
                ),
                io.Boolean.Input(
                    "lm_auto_download", default=True, advanced=True,
                    tooltip=(
                        "When the chosen key is not installed in LM Studio yet, ask the server "
                        "to download it. If LM Studio returns a job id the node waits for it "
                        "and reports progress at most every 5 s, capped at max(600 s, 4x "
                        "'lm_timeout'), after which the run continues anyway; if it returns no "
                        "job id the node does not wait at all. Off = the node only warns that "
                        "the model is missing and skips the whole load/reload step. lmstudio "
                        "only."
                    ),
                ),
                io.Boolean.Input(
                    "lm_auto_load", default=True, advanced=True,
                    tooltip=(
                        "Load the model before the stages start, and reload it when the "
                        "instance LM Studio currently holds was loaded with a smaller context "
                        "than 'lm_context_length'. Off = whatever is loaded is used as it "
                        "stands, including a context too small for this node's prompts. If the "
                        "load call fails, the node retries without the context setting and "
                        "then falls back to LM Studio's just-in-time loading. lmstudio only."
                    ),
                ),
                io.Int.Input(
                    "lm_context_length", default=32768, min=4096, max=262144, step=4096, advanced=True,
                    tooltip=(
                        "Context the model is loaded with, in tokens. If it is already loaded "
                        "with less, the node unloads and reloads it at this size - but only "
                        "while 'lm_auto_load' is on. This is the input side: "
                        "'guide_excerpt_chars' adds its characters to every shot-content "
                        "request and 'shots_per_request' decides how many shots share one "
                        "request, so raise this when you raise either. 'lm_max_tokens' caps "
                        "the reply, not the prompt."
                    ),
                ),
                io.Boolean.Input(
                    "lm_unload_after", default=False, advanced=True,
                    tooltip=(
                        "Unload the model from LM Studio once the prompts are written, before "
                        "any image or video rendering starts - frees the VRAM for the rest of "
                        "the workflow at the cost of a load on the next run. lmstudio only: "
                        "the cloud clients' unload is a no-op."
                    ),
                ),
                io.Float.Input(
                    "lm_temperature", default=0.8, min=0.0, max=2.0, step=0.05, advanced=True,
                    tooltip=(
                        "Sampling temperature applied to every LLM stage, from the track "
                        "interpretation to the image prompts. Sent to LM Studio, OpenAI and "
                        "OpenRouter. Never sent to Anthropic - current Claude models reject "
                        "the field - so on 'llm_provider' = anthropic this widget does "
                        "nothing."
                    ),
                ),
                io.Int.Input(
                    "lm_max_tokens", default=4096, min=256, max=32768, step=256, advanced=True,
                    tooltip=(
                        "Ceiling on one stage's reply, per request - not per run, and one "
                        "request has to hold every shot in its batch. Run out and the request "
                        "fails: either with a JSON parse error on the truncated reply, or, "
                        "when the model wrote nothing at all, with 'the model hit the token "
                        "limit before answering'. A failed shot-content or image-prompt batch "
                        "does not stop the run - those shots come back with empty content, and "
                        "a missing image prompt is rebuilt from the shot text and the art "
                        "direction - so raise this or lower 'shots_per_request'. On the paid "
                        "providers it is also the cap on billed output tokens per request."
                    ),
                ),
                io.Int.Input(
                    "lm_timeout", default=300, min=30, max=3600, advanced=True,
                    tooltip=(
                        "Seconds one LLM request may take before it is abandoned; the widget's "
                        "own minimum is 30. It also sizes the lifecycle calls - a model load "
                        "gets max(120 s, this) and a download waits up to max(600 s, 4x this). "
                        "A timeout counts as a failed attempt and consumes one of "
                        "'lm_retries'."
                    ),
                ),
                io.Int.Input(
                    "lm_retries", default=2, min=0, max=5, advanced=True,
                    tooltip=(
                        "Extra attempts per stage after a failure, with a growing wait between "
                        "them (1.5 s times the attempt number, so 1.5 s then 3 s at the "
                        "default of 2). On the final attempt the JSON schema is dropped and "
                        "the model is simply told to reply with raw JSON - the fallback that "
                        "rescues models which cannot do structured output, and which you lose "
                        "entirely at 0. Every retry is a fresh billed request on the paid "
                        "providers."
                    ),
                ),
                io.Combo.Input(
                    "lm_reasoning_effort",
                    options=["none", "low", "medium", "high", "default"],
                    default="none",
                    advanced=True,
                    tooltip=(
                        "Thinking budget sent with each request. 'none' is the default because "
                        "a reasoning model (Gemma 4, Qwen3) otherwise spends the whole "
                        "'lm_max_tokens' budget inside reasoning_content and returns an empty "
                        "message, which fails the stage. LM Studio is sent every value except "
                        "'default' and retries without the field if the model rejects it; "
                        "OpenAI and OpenRouter only receive low/medium/high; Anthropic never "
                        "receives it."
                    ),
                ),
                io.Int.Input(
                    "shots_per_request", default=4, min=1, max=16, advanced=True,
                    tooltip=(
                        "How many shots go into one LLM request, in both the shot-content "
                        "stage and the image-prompt stage: 20 shots at 4 means 5 requests "
                        "each. Lower is safer for small models because less has to fit in "
                        "'lm_max_tokens', at the cost of more round trips. In the shot-content "
                        "stage only, each batch after the first is also shown the previous "
                        "batch's last shot (first 1200 characters) so the look carries across "
                        "the seams."
                    ),
                ),
                io.Int.Input(
                    "guide_excerpt_chars", default=0, min=0, max=40000, step=1000, advanced=True,
                    tooltip=(
                        "Characters of the official MiniMax H3 guides pasted into the "
                        "shot-content system prompt. The budget is spent in order: base-en.txt "
                        "takes up to half of it, then ref-en.txt takes up to half of what is "
                        "left, so base gets about twice as much as ref and roughly three "
                        "quarters of the number you type is actually injected. It only does "
                        "something when ComfyUI-MiniMaxH3-Easy sits beside this pack in "
                        "custom_nodes; if it does not, nothing is injected and nothing is "
                        "logged. The text is repeated in every shot-content request, so raise "
                        "'lm_context_length' with it. 0 (default) sends the compact built-in "
                        "rules only."
                    ),
                ),
                io.String.Input(
                    "openrouter_api_key", default="", advanced=True,
                    tooltip=(
                        "Key for OpenRouter - one field covers all three uses: the LLM stages "
                        "('llm_provider' = openrouter) and the image and video rendering "
                        "('image_provider' / 'video_provider' = openrouter). Empty falls back "
                        "to OPENROUTER_API_KEY, then OPEN_ROUTER_API_KEY, then OPENROUTER_KEY; "
                        "a value typed here wins over all of them. The 'openrouter_model' list "
                        "is public and fills in without any key."
                    ),
                ),
                io.String.Input(
                    "openai_api_key", default="", advanced=True,
                    tooltip=(
                        "Key for the OpenAI LLM stages; OpenAI is never used for image or "
                        "video rendering here. Empty falls back to OPENAI_API_KEY, then "
                        "OPEN_AI_API_KEY; a value typed here wins over both. The "
                        "'openai_model' dropdown is probed with the environment variables "
                        "only, so a key typed here authorises the run but leaves that list at "
                        "its built-in fallbacks."
                    ),
                ),
                io.String.Input(
                    "anthropic_api_key", default="", advanced=True,
                    tooltip=(
                        "Key for the Anthropic LLM stages; Anthropic is never used for image "
                        "or video rendering here. Empty falls back to ANTHROPIC_API_KEY, then "
                        "ANTHROPIC_AUTH_TOKEN; a value typed here wins over both. The "
                        "'anthropic_model' dropdown is probed with the environment variables "
                        "only, so a key typed here authorises the run but leaves that list at "
                        "its built-in fallbacks."
                    ),
                ),
                io.String.Input(
                    "fal_api_key", default="", advanced=True,
                    tooltip=(
                        "Key for fal.ai, used for rendering images and clips only - fal is not "
                        "one of the LLM providers. Empty falls back to FAL_KEY, then "
                        "FAL_API_KEY, then FAL_ADMIN_API_KEY; a value typed here wins over all "
                        "three. With no key anywhere the run stops the moment rendering starts "
                        "('no fal.ai key'); the fal model dropdowns come from fal's public "
                        "index and need none."
                    ),
                ),
                io.Combo.Input(
                    "video_prompt_source", options=list(VIDEO_PROMPT_SOURCES), default="i2va",
                    advanced=True,
                    tooltip=(
                        "Which prompt set is actually sent to the video model. Both are always "
                        "written to the 'video_prompts_i2va' and 'video_prompts_ref2va' "
                        "outputs - this only picks the one that gets rendered. 'i2va' sends "
                        "the rendered start frame as the first frame. 'ref2va' sends "
                        "references instead of a start frame - the wired reference_images and "
                        "the rendered subject sheets, in that order, capped at the first 9. It "
                        "needs two things: at least one reference (turn on "
                        "'render_subject_sheets' with an 'image_provider', or wire "
                        "reference_images), and a video endpoint that declares a reference "
                        "field, such as minimax/h3/reference-to-video. Without references the "
                        "node warns and the clip goes out as text only; on an endpoint whose "
                        "schema has only a first-frame field, just the first reference is sent "
                        "as that frame; and a fal endpoint that requires an image aborts the "
                        "run before anything is billed."
                    ),
                ),
                io.Boolean.Input(
                    "render_subject_sheets", default=False, advanced=True,
                    tooltip=(
                        "Render one reference image per subject (up to 'max_subjects', default "
                        "6) before the shots, billed per image like any other. On fal they are "
                        "drawn by the plain 'fal_image_model', because an edit model cannot "
                        "draw a subject that does not exist yet - and once they exist the shot "
                        "frames switch over to 'fal_image_edit_model'. On OpenRouter there is "
                        "no switch: one image model handles both. Not strictly required for "
                        "'video_prompt_source' = ref2va - wired reference_images are sent too "
                        "- but only rendered sheets get the <Picture N> number the prompt "
                        "cites, so without them the references go out unlabelled."
                    ),
                ),
                io.Boolean.Input(
                    "live_preview", default=True, advanced=True,
                    tooltip=(
                        "Push each finished image and clip into the node's gallery the moment "
                        "it lands, instead of after the whole batch - which is how you catch a "
                        "bad prompt at shot 1 rather than paying for twelve. It costs nothing: "
                        "files go to ComfyUI/temp/music2prompts, and when "
                        "'save_rendered_video' is off the clips already written there are "
                        "reused instead of written twice. The gallery does not survive a page "
                        "reload; the results themselves come back on the IMAGE and VIDEO "
                        "outputs."
                    ),
                ),
                io.Combo.Input(
                    "prompt_expansion",
                    options=render_module.EXPANSION_MODES,
                    default="minimal",
                    advanced=True,
                    tooltip=(
                        "How much the video endpoint may rewrite the prompt before generating. "
                        "'minimal' and 'rich' set whichever of the two fields the endpoint "
                        "declares: 'prompt_expansion_mode' to 'fast' or 'quality' (the MiniMax "
                        "H3 endpoints), 'enable_prompt_expansion' to false or true "
                        "(fal-ai/wan/v2.7/image-to-video). 'model default' sends neither. On "
                        "an endpoint that declares neither - and every OpenRouter video model, "
                        "since that payload has no such field at all - this setting does "
                        "nothing. It matters because every MiniMax H3 endpoint defaults to "
                        "prompt_expansion_mode 'balanced', which decides per request, so each "
                        "shot's look is re-invented independently of the art direction."
                    ),
                ),
                io.Boolean.Input(
                    "lipsync_audio", default=True, advanced=True,
                    tooltip=(
                        "Send each shot's own slice of the track (MP3, inline in the request) "
                        "so the performance follows the vocal. It only reaches endpoints that "
                        "declare an audio input: on fal that is audio_url "
                        "(fal-ai/wan/v2.7/image-to-video) or reference_audio_urls "
                        "(minimax/h3/reference-to-video); a boolean named 'audio' means "
                        "'generate a soundtrack' and is skipped, and OpenRouter's video API "
                        "has no audio input at all - the node warns instead of sending. The "
                        "accepted window is 2-15 s for the list-shaped reference field and "
                        "2-30 s for a single audio_url; a shot outside it is warned about and "
                        "rendered without audio, so watch 'max_shot_seconds' and "
                        "'audio_clip_padding', which widens every clip."
                    ),
                ),
                io.Boolean.Input(
                    "style_anchor", default=True, advanced=True,
                    tooltip=(
                        "Render shot 1 on its own first, then hand it to every later shot as "
                        "an extra reference - this is what holds the grade, the grain and the "
                        "wardrobe together. The cost is serialisation: that one image renders "
                        "alone while 'render_concurrency' is ignored, and only the remaining "
                        "shots run concurrently. It does nothing unless there are at least 2 "
                        "shots and at least one reference already in play (wired "
                        "reference_images or 'render_subject_sheets'), and on fal it needs a "
                        "frame model that declares a reference field, i.e. "
                        "'fal_image_edit_model'. If shot 1 fails, the rest go out without an "
                        "anchor."
                    ),
                ),
                io.Boolean.Input(
                    "save_rendered_images", default=True, advanced=True,
                    tooltip=(
                        "Keep the rendered start frames and subject sheets in "
                        "ComfyUI/output/music2prompts, named <prefix>_<stamp>_frame001 and "
                        "<prefix>_<stamp>_subject001, in whatever format the endpoint returned "
                        "(png/jpg/gif/webp is detected from the bytes; anything unrecognised "
                        "is written with a .png name). Off only skips that write - the renders "
                        "were paid for either way, still leave the node on the 'images' and "
                        "'subject_images' outputs, and with 'live_preview' on a copy of each "
                        "is still written to ComfyUI/temp/music2prompts for the gallery. "
                        "Hidden while both 'image_provider' and 'video_provider' are 'none'."
                    ),
                ),
                io.Boolean.Input(
                    "save_rendered_video", default=True, advanced=True,
                    tooltip=(
                        "The clips are always written to disk - the 'videos' output and the "
                        "final film both need real files - so this only picks the folder: on, "
                        "ComfyUI/output/music2prompts as <prefix>_<stamp>_shot001.mp4; off, "
                        "ComfyUI/temp/music2prompts, which ComfyUI clears out, and there the "
                        "file is normally the one 'live_preview' already wrote, named "
                        "<prefix>_<stamp>_video001.mp4. The concatenated film always lands in "
                        "the output folder regardless."
                    ),
                ),
                io.Boolean.Input(
                    "concat_video", default=True, advanced=True,
                    tooltip=(
                        "Re-encode the clips into one H.264/mp4 on the 'final_video' output, "
                        "written to ComfyUI/output/music2prompts as <prefix>_<time>_final.mp4 "
                        "(PyAV, no ffmpeg binary, and never a stream copy). Every clip is "
                        "re-timed onto one grid to the exact length of its shot: one that came "
                        "back long is cut, one that came back short holds its last frame so "
                        "the film stays in sync with the music. Shots whose render failed are "
                        "left out entirely, so the film ends up shorter than the track by "
                        "their length."
                    ),
                ),
                io.Combo.Input(
                    "final_audio", options=list(AUDIO_MODES), default="music", advanced=True,
                    tooltip=(
                        "'music' muxes the track you fed in as AAC, cut to the film's length "
                        "or padded with silence if the film outlasts it. 'clips' keeps the "
                        "audio the video model returned, padding any silent clip so the cuts "
                        "stay aligned, and drops to silence if no clip carries an audio track "
                        "at all. 'none' leaves the film silent."
                    ),
                ),
                io.Combo.Input(
                    "final_fit", options=list(FIT_MODES), default="pad", advanced=True,
                    tooltip=(
                        "What happens to a clip whose aspect differs from the film's: 'pad' "
                        "letterboxes it on black, 'crop' scales up and cuts the edges off "
                        "centre, 'stretch' distorts it to fill the frame. A clip whose aspect "
                        "ratio is within 0.005 of the film's is only scaled, so this setting "
                        "does nothing when every clip has the same shape - which is the normal "
                        "case, since all shots are rendered at one 'aspect_ratio'."
                    ),
                ),
                io.Float.Input(
                    "final_fps", default=0.0, min=0.0, max=120.0, step=1.0, advanced=True,
                    tooltip=(
                        "Frame rate of the finished film; every clip is resampled onto it by "
                        "duplicating or dropping frames. 0 takes the highest rate any clip "
                        "reports - so slower clips get frames duplicated rather than faster "
                        "ones losing them - ignoring rates above 120 fps as mis-reported and "
                        "falling back to 24 if no clip reports one. A value near 23.976 / "
                        "29.97 / 59.94 is snapped to the exact 1001-based rational."
                    ),
                ),
                io.Int.Input(
                    "final_crf", default=20, min=0, max=51, advanced=True,
                    tooltip=(
                        "libx264 -crf for the final film, over the widget's 0-51 range: lower "
                        "is better quality and a bigger file, 20 is the default, and the "
                        "encode runs at preset 'medium'. It applies to the concatenated film "
                        "only - the individual clips are stored exactly as the provider "
                        "returned them."
                    ),
                ),
                io.Int.Input(
                    "render_timeout", default=600, min=60, max=3600, advanced=True,
                    tooltip=(
                        "Seconds one image or clip may take before the node gives up on it; "
                        "that shot then comes back empty - a black placeholder frame keeps its "
                        "slot on 'images', a failed clip is simply missing - while the rest of "
                        "the batch continues. It bounds the fal queue polling (checked every 2 "
                        "s) and the OpenRouter image request and video polling (every 3 s); "
                        "the submit call and the download of the finished file have their own "
                        "fixed timeouts. It has nothing to do with 'lm_timeout' - the LLM "
                        "stages are timed separately."
                    ),
                ),
                io.Combo.Input(
                    "whisper_model", options=list(asr_module.SUPPORTED_MODELS),
                    default="openai/whisper-large-v3", advanced=True,
                    tooltip=(
                        "Which local Whisper does the transcription - it runs on this machine, "
                        "nothing is uploaded and nothing is billed. On first use the weights "
                        "are pulled from HuggingFace into ComfyUI/models/whisper/<repo--id>, "
                        "several GB, once per model. Measured for large-v3 on an 11 GB card: "
                        "about 2.9 GB for the weights, ~7 GB peak with word timestamps and ~4 "
                        "GB with segment timestamps."
                    ),
                ),
                io.Combo.Input(
                    "whisper_dtype", options=["float16", "float32"], default="float16", advanced=True,
                    tooltip=(
                        "Precision of the Whisper weights. float16 halves the VRAM of float32 "
                        "and is chosen deliberately over bfloat16 so Turing cards stay "
                        "supported. It applies on GPU only: if 'whisper_device' resolves to "
                        "cpu, or no CUDA is present, the run is forced to float32 and this "
                        "widget does nothing."
                    ),
                ),
                io.String.Input(
                    "whisper_language", default="auto", advanced=True,
                    tooltip=(
                        "'auto' lets Whisper detect the language; anything else is passed to "
                        "the decoder as the forced language. Whatever you type here is also "
                        "used verbatim as the [Language] tag inside the H3 <d>...</d> dialogue "
                        "blocks, so 'Polish' reads better in a prompt than 'pl'. On 'auto' "
                        "that tag comes from a character-set guess that can only tell apart "
                        "English, Polish, German, Spanish, Chinese and Japanese - anything "
                        "else is tagged English."
                    ),
                ),
                io.Int.Input(
                    "whisper_chunk_length_s", default=30, min=5, max=30, advanced=True,
                    tooltip=(
                        "Length of the audio chunks the transformers ASR pipeline decodes when "
                        "the input is longer than one chunk; 30 is both the default and the "
                        "widget's maximum. It does not bound the memory word timestamps need - "
                        "that grows with the whole input, which is what "
                        "'whisper_window_seconds' is for. Lowering it does not fix an "
                        "out-of-memory error."
                    ),
                ),
                io.Int.Input(
                    "whisper_batch_size", default=1, min=1, max=32, advanced=True,
                    tooltip=(
                        "How many chunks are decoded at once. Word-level timestamps peak "
                        "around 7 GB of VRAM on an 11 GB card even at 1, so raise this only on "
                        "a card with memory to spare. If an attempt fails - out of memory or "
                        "for any other reason - the run steps down a ladder by itself: batch "
                        "1, then segment timestamps instead of word ones, then no timestamps "
                        "at all (which leaves every shot without lyrics), then CPU at float32."
                    ),
                ),
                io.Boolean.Input(
                    "whisper_word_timestamps", default=True, advanced=True,
                    tooltip=(
                        "Ask for per-word timings - a DTW pass over the cross-attentions - so "
                        "each lyric is assigned to the shot its midpoint falls in. This is the "
                        "expensive part of transcription: ~7 GB peak against ~4 GB for "
                        "segment-level timings on an 11 GB card. Off, words are still timed "
                        "but only per segment, so a line can be attributed to the neighbouring "
                        "shot."
                    ),
                ),
                io.Float.Input(
                    "whisper_window_seconds", default=30.0, min=0.0, max=600.0, step=5.0, advanced=True,
                    tooltip=(
                        "Transcribe the track in windows of this many seconds, shifting each "
                        "window's timings back into track time. Word-timestamp memory grows "
                        "with the length of the whole input, not with 'whisper_chunk_length_s' "
                        "- measured on an 11 GB card at ~7 GB for 60 s and an out-of-memory "
                        "failure at 90 s - hence the 30 s default. 0, or any value below 5, "
                        "disables windowing and sends the whole track in one pass; windowing "
                        "is also skipped when the track is shorter than the window plus 5 s."
                    ),
                ),
                io.Boolean.Input(
                    "whisper_keep_loaded", default=True, advanced=True,
                    tooltip=(
                        "Keep the Whisper pipeline in memory after the run - keyed by model, "
                        "device and dtype - so the next queue does not reload several GB from "
                        "disk. Off unloads it and empties the CUDA cache as soon as the "
                        "transcript is done, freeing that VRAM for the rest of the workflow at "
                        "the price of a full reload next time. Only one pipeline is cached, so "
                        "changing 'whisper_model', 'whisper_device' or 'whisper_dtype' "
                        "replaces it anyway."
                    ),
                ),
                io.Boolean.Input(
                    "whisper_skip", default=False, advanced=True,
                    tooltip=(
                        "Skip transcription entirely, for an instrumental track. The "
                        "'transcript' output is then empty, no lyrics reach any shot, the "
                        "model is told to leave every dialogue field empty whatever "
                        "'include_dialogue' says - so the H3 <d> blocks normally disappear - "
                        "and the H3 language tag falls back to English. 'free_comfy_vram' and "
                        "'free_lmstudio_vram' also never run, because nothing needs the VRAM."
                    ),
                ),
                io.Boolean.Input(
                    "free_comfy_vram", default=True, advanced=True,
                    tooltip=(
                        "Call ComfyUI's unload_all_models() and empty its cache just before "
                        "Whisper loads, so a checkpoint another node left resident cannot push "
                        "the transcription into an out-of-memory error. Those models reload "
                        "the next time they are used. Does nothing when 'whisper_skip' is on."
                    ),
                ),
                io.Boolean.Input(
                    "free_lmstudio_vram", default=True, advanced=True,
                    tooltip=(
                        "Unload the LM Studio model before Whisper starts; it is loaded again "
                        "three steps later, at the 'preparing model' step just before the "
                        "writing passes, which needs 'lm_auto_load' on - with that off the "
                        "node never loads it back itself. Keep this on with a single GPU: an "
                        "11 GB card cannot hold the LLM and Whisper large-v3 (~2.9 GB of "
                        "weights, ~7 GB peak) at once. Shown only for llm_provider 'lmstudio', "
                        "and skipped when 'whisper_skip' is on."
                    ),
                ),
                io.Boolean.Input(
                    "analyze_music", default=True, advanced=True,
                    tooltip=(
                        "Measure tempo, the beat grid, section boundaries and an energy curve, "
                        "and hand them to the LLM as facts instead of asking it to imagine the "
                        "music. librosa does this when it is installed; otherwise a "
                        "numpy/scipy fallback runs and the analysis JSON records which one "
                        "under 'backend'. Off, the track becomes one section called 'Part 1' "
                        "at 0 BPM with no beats - which also leaves 'snap_cuts_to_beats' "
                        "nothing to snap to."
                    ),
                ),
                io.Boolean.Input(
                    "snap_cuts_to_beats", default=True, advanced=True,
                    tooltip=(
                        "Move each shot boundary onto the nearest section edge, or onto the "
                        "nearest beat when no section edge is close enough, so cuts land on "
                        "the music rather than on an even division of the track. A boundary "
                        "moves at most 42% of an even division onto a section edge and 35% "
                        "onto a beat, and only when the move still leaves every shot at or "
                        "above 'min_shot_seconds'. Does nothing with 'analyze_music' off: "
                        "there is then no beat grid, and the single section spans the whole "
                        "track, so no boundary has anything to move to."
                    ),
                ),
                io.Float.Input(
                    "audio_clip_padding", default=0.0, min=0.0, max=2.0, step=0.05, advanced=True,
                    tooltip=(
                        "Widen every shot's audio clip by this many seconds on both sides, "
                        "clamped to the track. It always affects the 'audio_clips' output; it "
                        "affects the slice sent to a video model as driving audio only when "
                        "'lipsync_audio' is on, the video provider is fal and the chosen fal "
                        "model declares an audio field (OpenRouter's video API has none). 0 "
                        "keeps the cut sample-accurate against the shot boundaries, which is "
                        "what lipsync needs - any padding puts the vocal out of step with the "
                        "frames. Padding also lengthens the clip, and a clip outside the "
                        "accepted window (2-15 s for a reference-audio list such as H3, 2-30 s "
                        "for a driving-audio field) is sent as no audio at all."
                    ),
                ),
                io.Int.Input(
                    "max_subjects", default=6, min=0, max=16, advanced=True,
                    tooltip=(
                        "Upper bound on the recurring characters, locations and props locked "
                        "for consistency: the model is told to write at most this many and the "
                        "list is then truncated to it. With 'render_subject_sheets' on, each "
                        "subject costs one paid reference image. Those sheets reach the video "
                        "model only with 'video_prompt_source' 'ref2va', and then only the "
                        "first 9 references (wired 'reference_images' plus sheets) are sent, "
                        "with the prompt's <Picture N> labels numbered against that same list; "
                        "on the default 'i2va' the sheets are used as references for the start "
                        "frames instead. 0 leaves no subjects, so no sheets are rendered and a "
                        "ref2va prompt has nothing to point at."
                    ),
                ),
                io.String.Input(
                    "negative_prompt_base", multiline=True, default=DEFAULT_NEGATIVE, advanced=True,
                    tooltip=(
                        "Merged with the art direction's own negative terms and each shot's - "
                        "split on commas and semicolons, dots trimmed off the ends, "
                        "de-duplicated case-insensitively - into the 'negative_prompts' "
                        "output. It only reaches fal image endpoints that declare a "
                        "'negative_prompt' field, or one whose schema could not be read, where "
                        "it is sent blind; either way it is the first field dropped when the "
                        "endpoint rejects the payload. OpenRouter's image API has no negative "
                        "field and no video path sends one at all, so on those it is written "
                        "to the output and otherwise ignored."
                    ),
                ),
                io.Boolean.Input(
                    "include_dialogue", default=True, advanced=True,
                    tooltip=(
                        "Put each shot's transcribed lyrics into the H3 <d>[Language] ...</d> "
                        "block, rendered as '<subject> (S1) sings: ...' - that block is what "
                        "makes a lipsync-capable model perform the line. Off, the per-shot "
                        "lyrics are withheld from the shot-writing stage and the model is told "
                        "to leave the dialogue field empty, so no <d> block appears in either "
                        "the i2va or the ref2va prompt; the full transcript still goes to the "
                        "first stage, which reads the track before any shot is written. When "
                        "Whisper returned no words (including under 'whisper_skip') the model "
                        "gets that same instruction, so in practice no <d> block is written - "
                        "but nothing downstream strips one if the model writes a line anyway."
                    ),
                ),
                io.String.Input(
                    "h3_style_directive", default="", advanced=True,
                    tooltip=(
                        "Extra clause appended to the style line of every H3 prompt, in both "
                        "the i2va and the ref2va form - for example '35 mm film grain, no lens "
                        "flare'. It is joined to the LLM's own visual_style with a comma, so "
                        "write attributes, not sentences; a leading field label such as "
                        "'style:' is stripped and a trailing full stop removed. Empty leaves "
                        "the style line exactly as the art direction wrote it."
                    ),
                ),
                io.Boolean.Input(
                    "save_json", default=True, advanced=True,
                    tooltip=(
                        "Write the whole run to "
                        "ComfyUI/output/music2prompts/<prefix>_<stamp>_analysis.json: the "
                        "music analysis, the transcript, the interpretation and art direction, "
                        "every subject, per shot its times, section, lyrics and the raw "
                        "content the model wrote, plus which providers and models were used "
                        "for rendering, how many images and sheets came back, and the file "
                        "path of every clip. The same text is on the 'analysis_json' output "
                        "whether this is on or off."
                    ),
                ),
                io.Boolean.Input(
                    "save_transcript", default=True, advanced=True,
                    tooltip=(
                        "Write <prefix>_<stamp>_transcript.txt next to the JSON: the detected "
                        "language, the shot count, the full transcript, then one block per "
                        "shot with its time range, section, the image prompt it was rendered "
                        "from, and the first line of its i2va video prompt (capped at 200 "
                        "characters), and the words sung inside that shot. A shot with no words "
                        "reads '(instrumental)'. The 'transcript' output carries only the "
                        "bare Whisper text, with no timings and no shot structure."
                    ),
                ),
                io.String.Input(
                    "filename_prefix", default="music2prompts", advanced=True,
                    tooltip=(
                        "Leading part of every file this run writes. One timestamp is taken "
                        "when the run starts and shared by the frames, subject sheets, clips, "
                        "transcript and JSON (<prefix>_<stamp>_frame001, _subject001, "
                        "_shot001.mp4, _transcript.txt, _analysis.json), so a run's files sort "
                        "together; the concatenated film takes a fresh timestamp at the moment "
                        "it is assembled, as <prefix>_<time>_final.mp4. Empty falls back to "
                        "'music2prompts' for the transcript, the JSON and the final film only "
                        "- the frames, sheets and clips get no fallback and are written with a "
                        "leading underscore instead."
                    ),
                ),
                io.Boolean.Input(
                    "verbose", default=False, advanced=True,
                    tooltip=(
                        "Log the number of characters each LLM reply came back with - which is "
                        "how a truncated reply is told apart from a refused one. On LM Studio "
                        "the line names the stage; the OpenAI-compatible providers name the "
                        "provider instead, so those lines are only told apart by their order, "
                        "and Anthropic logs nothing for a stage that comes back through its "
                        "structured tool call. The stage-by-stage progress lines, the render "
                        "counts and every warning are printed regardless of this switch. It is "
                        "also handed to the fal and OpenRouter media clients, where it "
                        "currently produces no extra output."
                    ),
                ),
io.Boolean.Input(
                    "save_cost_report", default=True, advanced=True,
                    tooltip=(
                        "Write <prefix>_<stamp>_cost.json and _cost.txt next to the JSON: "
                        "every billed call of the run with its model, what it billed and how "
                        "the price was arrived at, plus the per-model subtotals and the total "
                        "the node shows. The JSON also lists the assumptions behind the "
                        "figure. Costs nothing and needs no key - the numbers come from the "
                        "replies the providers already sent. It sits at the end of the list "
                        "rather than beside the other save_* switches because a widget "
                        "inserted in the middle would shift every stored value after it in "
                        "workflows saved before it existed."
                    ),
                ),
                io.Image.Input(
                    "reference_images", optional=True,
                    tooltip=(
                        "Optional look/identity references. Only the first 4 images of the "
                        "batch are used, each downscaled to a 768 px longest side if it is "
                        "larger; they are sent to the LLM to be described, and that "
                        "description is forced into the art direction and the subject bible. "
                        "The pixels themselves are prepended to the reference list of every "
                        "start-frame render and of ref2va clips, so with 'image_provider' at "
                        "'none' they still reach a ref2va video model while an i2va run keeps "
                        "only the written description - and their presence is what switches "
                        "the shots onto 'fal_image_edit_model'."
                    ),
                ),
            ],
            outputs=[
                io.Custom(pipe_module.PIPE_TYPE).Output(
                    display_name="pipe",
                    tooltip=(
                        "Everything this run wrote in words and numbers, on one wire: the "
                        "start-frame and reference image prompts, the subject names, both "
                        "MiniMax H3 prompt forms, the negatives, and every shot's index, "
                        "start, end and duration, plus the transcript and the full analysis "
                        "JSON. Feed it to 'Music2Video Pipe Expand' wherever you need one "
                        "of them as its own socket; that node passes the pipe through, so it "
                        "can be tapped as many times as you like along a chain. Only the "
                        "media stays on its own sockets here, because an IMAGE or a VIDEO "
                        "goes straight into a preview or a save node."
                    ),
                ),
                io.Audio.Output(
                    display_name="audio_clips",
                    is_output_list=True,
                    tooltip=(
                        "One AUDIO per shot, cut from the input track at that shot's "
                        "boundaries and widened on both sides by 'audio_clip_padding', with "
                        "the original sample rate and channel layout untouched. Always "
                        "produced, even on a prompts-only run. Send them to PreviewAudio / "
                        "SaveAudio or a lipsync node. The same clips are sent as driving audio "
                        "when 'lipsync_audio' is on, 'video_provider' is fal, the chosen model "
                        "declares an input for a driving audio track, and the clip itself "
                        "lands inside that field's length window (2-15 s for a list-shaped "
                        "reference field, 2-30 s otherwise). The clip is re-encoded to MP3 and "
                        "sent inline in the request as a data: URI; clips outside the window "
                        "go out without audio and a warning is logged."
                    ),
                ),
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip=(
                        "The rendered start frames, one IMAGE per shot in shot order. Empty "
                        "when 'image_provider' is 'none' (the default) - a prompts-only run "
                        "puts nothing here, and any other setting means one billed image "
                        "generation per shot. A shot whose render failed keeps its slot as a "
                        "black frame so the list stays aligned with 'shot_index'; if every "
                        "shot fails the node raises instead. Wire to PreviewImage / SaveImage, "
                        "or into an image-to-video node as the first frame."
                    ),
                ),
                io.Image.Output(
                    display_name="subject_images",
                    is_output_list=True,
                    tooltip=(
                        "The rendered subject reference sheets, one per subject. Empty unless "
                        "'image_provider' is set AND 'render_subject_sheets' is on - each "
                        "sheet is one more billed image on top of the per-shot renders. Sheets "
                        "that failed are dropped rather than padded, so this can be shorter "
                        "than 'reference_subjects'. These are the same images the run feeds "
                        "back as identity references for the start frames, and - when "
                        "'video_prompt_source' is ref2va - as the <Picture N> references sent "
                        "with the video request (the first nine references only, counting any "
                        "images wired into 'reference_images')."
                    ),
                ),
                io.Video.Output(
                    display_name="videos",
                    is_output_list=True,
                    tooltip=(
                        "The rendered per-shot clips as VIDEO, in shot order, opened from the "
                        "mp4 files written to ComfyUI/output/music2prompts (or ComfyUI's temp "
                        "folder when 'save_rendered_video' is off). Empty when "
                        "'video_provider' is 'none' - any other setting means one billed video "
                        "generation per shot - and empty on a ComfyUI too old to expose "
                        "VideoFromFile. Clips that failed are skipped rather than padded, so "
                        "this can be shorter than 'shot_index' - use 'final_video' if you need "
                        "the cut in one piece."
                    ),
                ),
                io.Video.Output(
                    display_name="final_video",
                    tooltip=(
                        "A single VIDEO: every rendered clip concatenated in shot order, each "
                        "trimmed or its last frame held to its shot's exact duration, "
                        "re-encoded through PyAV (libx264, yuv420p, mp4 with faststart). The "
                        "output size is taken from the clips themselves - the most common "
                        "frame size, ties going to the largest - and 'final_fit' decides "
                        "whether an odd-sized clip is letterboxed, stretched or cropped into "
                        "it. 'final_fps' sets the frame rate (0 = the highest rate found among "
                        "the clips), 'final_crf' the x264 quality, and 'final_audio' the "
                        "soundtrack. None unless 'video_provider' is set and 'concat_video' is "
                        "on, None if the muxing pass fails (a warning is logged and the "
                        "individual clips survive on 'videos'), and None on a ComfyUI too old "
                        "to expose VideoFromFile - the mp4 is written to "
                        "ComfyUI/output/music2prompts either way. Wire to SaveVideo."
                    ),
                ),
            ],
            # the gallery is addressed per node, so the run has to know which one it is
            hidden=[io.Hidden.unique_id],
        )

    # ------------------------------------------------------------------ execution

    @classmethod
    def execute(  # noqa: PLR0913 - a monolithic node by design
        cls,
        audio,
        instruction: str,
        lm_model: str,
        visual_style: str,
        aspect_ratio: str,
        clip_seconds: float,
        min_shot_seconds: float,
        max_shot_seconds: float,
        num_shots: int,
        creativity: float,
        dynamicity: float,
        word_influence: float,
        whisper_device: str,
        seed: int,
        llm_provider: str = "lmstudio",
        image_provider: str = "none",
        fal_image_model: str = "",
        fal_image_edit_model: str = "",
        openrouter_image_model: str = "",
        video_provider: str = "none",
        fal_video_model: str = "",
        openrouter_video_model: str = "",
        render_concurrency: int = 2,
        openrouter_model: str = "",
        openai_model: str = "",
        anthropic_model: str = "",
        openrouter_api_key: str = "",
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        fal_api_key: str = "",
        video_prompt_source: str = "i2va",
        render_subject_sheets: bool = False,
        live_preview: bool = True,
        prompt_expansion: str = "minimal",
        lipsync_audio: bool = True,
        style_anchor: bool = True,
        save_rendered_images: bool = True,
        save_rendered_video: bool = True,
        concat_video: bool = True,
        final_audio: str = "music",
        final_fit: str = "pad",
        final_fps: float = 0.0,
        final_crf: int = 20,
        render_timeout: int = 600,
        lm_url: str = DEFAULT_URL,
        lm_api_key: str = "",
        lm_model_override: str = "",
        lm_auto_download: bool = True,
        lm_auto_load: bool = True,
        lm_context_length: int = 32768,
        lm_unload_after: bool = False,
        lm_temperature: float = 0.8,
        lm_max_tokens: int = 4096,
        lm_timeout: int = 300,
        lm_retries: int = 2,
        lm_reasoning_effort: str = "none",
        shots_per_request: int = 4,
        guide_excerpt_chars: int = 0,
        whisper_model: str = "openai/whisper-large-v3",
        whisper_dtype: str = "float16",
        whisper_language: str = "auto",
        whisper_chunk_length_s: int = 30,
        whisper_batch_size: int = 1,
        whisper_word_timestamps: bool = True,
        whisper_window_seconds: float = 30.0,
        whisper_keep_loaded: bool = True,
        whisper_skip: bool = False,
        free_comfy_vram: bool = True,
        free_lmstudio_vram: bool = True,
        analyze_music: bool = True,
        snap_cuts_to_beats: bool = True,
        audio_clip_padding: float = 0.0,
        max_subjects: int = 6,
        negative_prompt_base: str = DEFAULT_NEGATIVE,
        include_dialogue: bool = True,
        h3_style_directive: str = "",
        save_json: bool = True,
        save_transcript: bool = True,
        save_cost_report: bool = True,
        filename_prefix: str = "music2prompts",
        verbose: bool = False,
        reference_images=None,
    ) -> io.NodeOutput:
        started = time.perf_counter()
        stamp = render_module.run_stamp()  # shared by every file this run writes
        provider = (llm_provider or "lmstudio").strip().lower()
        wants_images = (image_provider or "none").lower() not in {"", "none"}
        wants_video = (video_provider or "none").lower() not in {"", "none"}
        feed = preview_module.PreviewFeed(
            getattr(cls.hidden, "unique_id", None),
            filename_prefix,
            enabled=live_preview and (wants_images or wants_video),
        )
        feed.reset()
        # Deliberately not gated on live_preview or on a media provider: an LLM-only run
        # still spends money, and a node that shows nothing there looks broken.
        ledger = cost_module.CostLedger(getattr(cls.hidden, "unique_id", None))
        stages = BASE_STAGES + int(wants_images) + int(wants_video)
        stages += int(wants_images and render_subject_sheets)  # the sheets are their own pass
        progress = _ProgressReporter(stages, verbose)

        samples, sample_rate = audio_to_mono(audio)
        duration = len(samples) / float(sample_rate)
        log(f"track: {duration:.1f}s @ {sample_rate} Hz")

        model_key, api_key = cls._pick_model(
            provider,
            lm_model_override or lm_model,
            {"openrouter": openrouter_model, "openai": openai_model, "anthropic": anthropic_model},
            {
                "lmstudio": lm_api_key,
                "openrouter": openrouter_api_key,
                "openai": openai_api_key,
                "anthropic": anthropic_api_key,
            },
        )
        client = make_llm_client(
            provider,
            lm_url=lm_url,
            api_key=api_key,
            timeout=lm_timeout,
            retries=lm_retries,
            verbose=verbose,
            ledger=ledger,
        )
        local_llm = provider == "lmstudio"
        image_model = fal_image_model if image_provider == "fal" else openrouter_image_model
        video_model = fal_video_model if video_provider == "fal" else openrouter_video_model
        if wants_video and video_provider == "fal":
            cls._check_video_inputs(video_model, video_prompt_source, wants_images, render_subject_sheets)
        log(f"LLM: {provider}/{model_key}")

        # ---------------------------------------------------------- transcription
        progress.step("transcribing")
        transcription = {"text": "", "language": "English", "words": []}
        if not whisper_skip:
            if free_comfy_vram:
                asr_module.free_comfy_vram()
            if free_lmstudio_vram and local_llm:
                log(f"unloading '{model_key}' from LM Studio to make room for Whisper")
                client.unload(model_key)
            speech, _ = audio_to_mono(audio, target_sr=asr_module.WHISPER_SAMPLE_RATE)
            try:
                transcription = asr_module.transcribe(
                    speech,
                    sample_rate=asr_module.WHISPER_SAMPLE_RATE,
                    repo_id=whisper_model,
                    device_choice=whisper_device,
                    dtype_choice=whisper_dtype,
                    chunk_length_s=whisper_chunk_length_s,
                    batch_size=whisper_batch_size,
                    language=whisper_language,
                    word_timestamps=whisper_word_timestamps,
                    keep_loaded=whisper_keep_loaded,
                    window_seconds=whisper_window_seconds,
                )
                log(f"transcript: {len(transcription['words'])} words, language {transcription['language']}")
            except Exception as exc:
                warn(f"transcription failed ({exc}); continuing as an instrumental")
            if not whisper_keep_loaded:
                asr_module.unload_all()
        _interrupt_check()

        # ---------------------------------------------------------- music analysis
        progress.step("analysing music")
        analysis = music_dsp.analyze(samples, sample_rate, enabled=analyze_music)
        _interrupt_check()

        # ---------------------------------------------------------- shot planning
        progress.step("planning shots")
        clip_seconds, max_shot_seconds = cls._fit_shots_to_model(
            clip_seconds, max_shot_seconds, video_provider if wants_video else "none", video_model
        )
        slots = plan_shots(
            duration=duration,
            clip_seconds=clip_seconds,
            min_seconds=min_shot_seconds,
            max_seconds=max_shot_seconds,
            dynamicity=dynamicity,
            num_shots=num_shots,
            beats=analysis.get("beats"),
            sections=analysis.get("sections"),
            snap_to_beats=snap_cuts_to_beats,
        )
        slots = attach_lyrics(slots, transcription.get("words") or [])
        log(f"{len(slots)} shots planned ({min_shot_seconds:.1f}-{max_shot_seconds:.1f}s each)")
        _interrupt_check()

        # ---------------------------------------------------------- LLM
        progress.step(f"preparing model '{model_key}' ({provider})")
        if local_llm:
            client.ensure_model(
                model_key,
                auto_download=lm_auto_download,
                auto_load=lm_auto_load,
                context_length=lm_context_length,
                progress=lambda message: log(message),
            )

        runner = StageRunner(
            client=client,
            model=model_key,
            temperature=lm_temperature,
            max_tokens=lm_max_tokens,
            seed=int(seed) or None,
            guide_excerpt=load_h3_guide(guide_excerpt_chars),
            verbose=verbose,
            progress=log,
            reasoning_effort=lm_reasoning_effort,
        )

        reference_descriptions: list[str] = []
        reference_uris: list[str] = []
        if reference_images is not None:
            uris = reference_uris
            try:
                count = int(reference_images.shape[0])
            except Exception:
                count = 0
            for index in range(min(count, 4)):
                uri = image_tensor_to_data_uri(reference_images, index)
                if uri:
                    uris.append(uri)
            reference_descriptions = runner.describe_reference_images(uris)

        facts = music_dsp.compact_for_llm(analysis)
        progress.step("writing the treatment")
        interpretation = runner.interpret(
            facts, instruction, transcription.get("text", ""), word_influence
        )
        art = runner.art_direction(
            interpretation, instruction, visual_style, creativity, reference_descriptions
        )
        subjects = runner.subjects(
            interpretation, art, instruction, max_subjects, reference_descriptions
        )
        _interrupt_check()

        progress.step("writing shot content")
        content = runner.shot_content(
            slots,
            interpretation,
            art,
            subjects,
            instruction,
            dynamicity,
            word_influence,
            include_dialogue and bool(transcription.get("words")),
            batch_size=shots_per_request,
        )
        _interrupt_check()

        progress.step("writing prompts")
        image_prompts = runner.image_prompts(slots, content, art, subjects, aspect_ratio, shots_per_request)
        reference_prompts = runner.reference_prompts(subjects, art, aspect_ratio)

        if lm_unload_after and local_llm:
            client.unload(model_key)

        # ---------------------------------------------------------- assembly
        lyrics_language = transcription.get("language") or "English"
        base_negative = (negative_prompt_base or "").strip()
        extra_negative = ", ".join(str(item) for item in as_list(art.get("negative_extra")) if item)

        start_frames: list[str] = []
        i2va: list[str] = []
        ref2va: list[str] = []
        negatives: list[str] = []
        indices: list[int] = []
        starts: list[float] = []
        ends: list[float] = []
        durations: list[float] = []
        audio_clips: list[dict] = []
        shots_debug: list[dict] = []
        h3_shots: list[H3Shot] = []  # kept so ref2va can be rewritten once the references exist

        for slot in slots:
            item = content.get(slot.index) or {}
            shot = cls._build_shot(
                slot, item, subjects, art, lyrics_language, h3_style_directive, include_dialogue
            )
            h3_shots.append(shot)
            start_frames.append(image_prompts.get(slot.index) or cls._fallback_image_prompt(shot, art, aspect_ratio))
            i2va.append(render_i2va(shot))
            ref2va.append(render_ref2va(shot))
            negatives.append(
                cls._merge_negatives(base_negative, extra_negative, first_str(item, "negative_extra"))
            )
            indices.append(slot.index)
            starts.append(round(slot.start, 3))
            ends.append(round(slot.end, 3))
            durations.append(slot.duration)
            audio_clips.append(slice_audio(audio, slot.start, slot.end, audio_clip_padding))
            shots_debug.append(
                {
                    "shot": slot.index,
                    "start": slot.start,
                    "end": slot.end,
                    "duration": slot.duration,
                    "section": slot.section,
                    "lyrics": slot.lyrics,
                    "content": item,
                }
            )

        subject_names = [first_str(subject, "name", default=f"subject {i + 1}") for i, subject in enumerate(subjects)]
        subject_prompts = [
            reference_prompts.get(name)
            or reference_prompts.get(name.lower())
            or cls._fallback_reference_prompt(subjects[index], art, aspect_ratio)
            for index, name in enumerate(subject_names)
        ]

        # ---------------------------------------------------------- optional rendering
        images_out: list = []
        subject_images_out: list = []
        videos_out: list = []
        image_payloads: list[bytes | None] = []
        subject_payloads: list[bytes | None] = []
        video_paths: list[str] = []
        final_video = None
        final_info: dict = {}

        # the LLM bill is complete here, so the panel carries a real number before the
        # slow, expensive half of the run begins
        ledger.publish()
        sheet_of: dict[str, int] = {}  # subject name -> its 1-based position in the sent references
        identity_uris: list[str] = list(reference_uris)
        if wants_images:
            image_client = render_module.make_media_client(
                image_provider, fal_api_key if image_provider == "fal" else openrouter_api_key,
                timeout=render_timeout, verbose=verbose, ledger=ledger,
            )

            # The subject sheets go FIRST and then become references for every shot: that
            # is the whole mechanism behind keeping one face and one look across a film.
            # They are rendered by the plain text-to-image model, because an edit model
            # cannot draw a subject that does not exist yet.
            if render_subject_sheets and subject_prompts:
                progress.step(f"rendering {len(subject_prompts)} subject sheet(s) with {image_provider}")
                subject_payloads = render_module.render_images(
                    image_client,
                    image_model,
                    [
                        ImageRequest(
                            prompt=prompt,
                            negative=base_negative,
                            aspect_ratio=aspect_ratio,
                            seed=(int(seed) + 1000 + index) if seed else None,
                            references=list(reference_uris),
                            label=subject_names[index] if index < len(subject_names) else "",
                        )
                        for index, prompt in enumerate(subject_prompts)
                    ],
                    render_concurrency,
                    on_done=lambda index, data, total=len(subject_prompts): feed.publish(
                        "image",
                        1000 + index,
                        data,
                        label=subject_names[index] if index < len(subject_names) else f"subject {index + 1}",
                        total=total,
                    ),
                )
                subject_images_out = [
                    render_module.image_bytes_to_tensor(data) for data in subject_payloads if data
                ]
                for index, payload in enumerate(subject_payloads):
                    if not payload:
                        continue
                    identity_uris.append(render_module.data_uri(payload))
                    if index < len(subject_names):
                        sheet_of[subject_names[index]] = len(identity_uris)
                _interrupt_check()

            frame_model, frame_client = cls._frame_model(
                image_provider, image_model, fal_image_edit_model, bool(identity_uris), image_client
            )
            progress.step(f"rendering {len(start_frames)} start frame(s) with {image_provider}")

            def frame_request(index: int, prompt: str, extra: list[str]) -> ImageRequest:
                return ImageRequest(
                    prompt=prompt,
                    negative=negatives[index],
                    aspect_ratio=aspect_ratio,
                    seed=(int(seed) + index) if seed else None,
                    references=identity_uris + extra,
                    label=f"shot {index + 1}",
                )

            def publish_frame(index: int, data: bytes, total: int = len(start_frames)) -> None:
                feed.publish("image", index, data, label=f"shot {index + 1}", total=total)

            image_errors: list[Exception] = []
            anchor: list[str] = []
            first: list[bytes | None] = []
            wants_anchor = style_anchor and bool(identity_uris) and len(start_frames) > 1
            if wants_anchor:
                # Shot 1 alone first; every later shot then also sees it, which is what
                # holds the grade, the grain and the wardrobe together across the film.
                first = render_module.render_images(
                    frame_client, frame_model, [frame_request(0, start_frames[0], [])],
                    1, image_errors, on_done=publish_frame,
                )
                if first and first[0]:
                    anchor = [render_module.data_uri(first[0])]
                else:
                    warn("the style anchor failed to render; the remaining shots go out without it")

            rest = render_module.render_images(
                frame_client,
                frame_model,
                [
                    frame_request(index, prompt, anchor)
                    for index, prompt in enumerate(start_frames)
                    if not (wants_anchor and index == 0)
                ],
                render_concurrency,
                image_errors,
                on_done=lambda offset, data: publish_frame(offset + (1 if wants_anchor else 0), data),
            )
            image_payloads = (first + rest) if wants_anchor else rest
            ledger.publish()

            if not any(image_payloads):
                raise RuntimeError(
                    f"{PREFIX} every image render failed on {image_provider}/{frame_model}. "
                    f"First error: {image_errors[0] if image_errors else 'unknown'}"
                )
            # a failed shot keeps its slot as a black frame, so `images` stays aligned
            # with the shot list instead of silently shifting every later shot
            images_out = [
                render_module.image_bytes_to_tensor(data)
                if data
                else render_module.placeholder_image(aspect_ratio)
                for data in image_payloads
            ]
            if save_rendered_images:
                written = render_module.save_images(image_payloads, filename_prefix, "frame", stamp)
                written += render_module.save_images(
                    subject_payloads, filename_prefix, "subject", stamp
                )
                if written:
                    log(f"{len(written)} image(s) written to {os.path.dirname(written[0])}")
            log(f"{len(images_out)}/{len(start_frames)} start frames rendered")
            _interrupt_check()

        if wants_video:
            progress.step(f"rendering {len(slots)} clip(s) with {video_provider}")
            video_client = render_module.make_media_client(
                video_provider, fal_api_key if video_provider == "fal" else openrouter_api_key,
                timeout=render_timeout, verbose=verbose, ledger=ledger,
            )
            use_reference = (video_prompt_source or "i2va").lower() == "ref2va"
            audio_uris = cls._shot_audio(
                audio_clips, video_provider, video_model, lipsync_audio
            )
            # H3 numbers its references by the order they are sent, and the prompt has to
            # name them - <Picture 2> means "the second entry of this list". Send the same
            # list the sheets were numbered against, so the labels point at the right face.
            subject_uris = identity_uris[:9]
            if use_reference and (sheet_of or any(audio_uris)):
                for index, shot in enumerate(h3_shots):
                    for subject in shot.subjects:
                        subject.picture = sheet_of.get(subject.name, 0)
                    if index < len(audio_uris) and audio_uris[index]:
                        # naming the clip in the prompt is what ties it to the performance;
                        # an audio reference nothing refers to is just a file
                        shot.audio_reference = (
                            "this shot's own slice of the original track, carrying the vocal"
                        )
                ref2va = [render_ref2va(shot) for shot in h3_shots]
                bound = sum(1 for shot in h3_shots for s in shot.subjects if s.picture)
                log(f"{bound} subject reference(s) bound to a <Picture> label in the ref2va prompts")
            elif use_reference and not subject_uris:
                warn(
                    "ref2va was selected but there are no reference images to send: turn on "
                    "render_subject_sheets (with an image_provider), or wire reference_images."
                )
            video_requests: list[VideoRequest] = []
            for index, slot in enumerate(slots):
                frame = image_payloads[index] if index < len(image_payloads) else None
                video_requests.append(
                    VideoRequest(
                        prompt=(ref2va if use_reference else i2va)[index],
                        seconds=slot.duration,
                        aspect_ratio=aspect_ratio,
                        seed=(int(seed) + index) if seed else None,
                        first_frame="" if use_reference else (render_module.data_uri(frame) if frame else ""),
                        references=subject_uris if use_reference else [],
                        audio=audio_uris[index] if index < len(audio_uris) else "",
                        expansion=prompt_expansion,
                        label=f"shot {slot.index}",
                    )
                )
            video_errors: list[Exception] = []
            payloads = render_module.render_videos(
                video_client,
                video_model,
                video_requests,
                render_concurrency,
                video_errors,
                on_done=lambda index, data, total=len(video_requests): feed.publish(
                    "video", index, data, label=f"shot {index + 1}", total=total
                ),
            )
            if not any(payloads):
                raise RuntimeError(
                    f"{PREFIX} every clip failed on {video_provider}/{video_model}. "
                    f"First error: {video_errors[0] if video_errors else 'unknown'}"
                )
            ledger.publish()
            rendered_seconds = [slot.duration for slot, item in zip(slots, payloads) if item]
            # the clips are always written out: the VIDEO output and the final cut
            # both need real files. save_rendered_video only decides output vs temp.
            video_paths = render_module.save_videos(
                payloads,
                filename_prefix,
                temporary=not save_rendered_video,
                reuse=feed.paths("video"),
                stamp=stamp,
            )
            videos_out = cls._videos_from_paths(video_paths)
            log(f"{sum(1 for item in payloads if item)}/{len(video_requests)} clips rendered")
            _interrupt_check()

            if concat_video and video_paths:
                final_video, final_info = cls._concat(
                    video_paths,
                    rendered_seconds,
                    audio if final_audio == "music" else None,
                    final_audio,
                    final_fit,
                    final_fps,
                    final_crf,
                    filename_prefix,
                )
                _interrupt_check()

        debug = {
            "duration": round(duration, 3),
            "music": analysis,
            "transcription": {
                "language": transcription.get("language"),
                "text": transcription.get("text"),
                "word_count": len(transcription.get("words") or []),
            },
            "interpretation": interpretation,
            "art_direction": art,
            "subjects": subjects,
            "reference_image_descriptions": reference_descriptions,
            "shots": shots_debug,
            "rendering": {
                "image_provider": image_provider if wants_images else "none",
                "image_model": image_model if wants_images else "",
                "images_rendered": len(images_out),
                "subject_images_rendered": len(subject_images_out),
                "video_provider": video_provider if wants_video else "none",
                "video_model": video_model if wants_video else "",
                "video_prompt_source": video_prompt_source,
                "video_paths": video_paths,
                "concurrency": render_concurrency,
                "final_video": final_info,
            },
            "settings": {
                "llm_provider": provider,
                "lm_model": model_key,
                "clip_seconds": clip_seconds,
                "min_shot_seconds": min_shot_seconds,
                "max_shot_seconds": max_shot_seconds,
                "audio_clip_padding": audio_clip_padding,
                "dynamicity": dynamicity,
                "creativity": creativity,
                "word_influence": word_influence,
                "aspect_ratio": aspect_ratio,
            },
        }
        debug["cost"] = ledger.payload(final=True)["total"]
        analysis_json = json.dumps(debug, ensure_ascii=False, indent=2)
        if save_json:
            cls._save_text(analysis_json, filename_prefix, "analysis", "json", stamp)
        if save_cost_report:
            cls._save_text(ledger.report_json(), filename_prefix, "cost", "json", stamp)
            cls._save_text(ledger.report_text(), filename_prefix, "cost", "txt", stamp)
        if save_transcript:
            cls._save_text(
                cls._transcript_document(transcription, slots, i2va, start_frames),
                filename_prefix,
                "transcript",
                "txt",
                stamp,
            )

        log(f"done in {time.perf_counter() - started:.1f}s - {len(slots)} shots, {len(subject_names)} subjects")
        ledger.publish(final=True)
        # feed.ui() is {} whenever nothing was previewed, and `or None` below would then
        # drop the cost payload with it - so the two are merged before that test
        replay = feed.ui()
        replay.update(ledger.ui())
        return io.NodeOutput(
            pipe_module.pack(
                image_prompts_start=start_frames,
                image_prompts_reference=subject_prompts,
                reference_subjects=subject_names,
                video_prompts_i2va=i2va,
                video_prompts_ref2va=ref2va,
                negative_prompts=negatives,
                shot_index=indices,
                start_times=starts,
                end_times=ends,
                durations=durations,
                transcript=transcription.get("text", ""),
                analysis_json=analysis_json,
            ),
            audio_clips,
            images_out,
            subject_images_out,
            videos_out,
            final_video,
            ui=replay or None,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _pick_model(
        provider: str,
        local_model: str,
        cloud_models: dict[str, str],
        keys: dict[str, str],
    ) -> tuple[str, str]:
        """Model key and API key for the selected provider."""
        if provider == "lmstudio":
            return (local_model or "").strip(), keys.get("lmstudio", "")
        model = (cloud_models.get(provider) or "").strip()
        if not model or model.startswith("("):
            raise ValueError(
                f"{PREFIX} pick a model for '{provider}' ({provider}_model). The list was "
                "empty when the node was loaded - add the API key, then right-click the node "
                "and choose 'Refresh model lists'."
            )
        return model, keys.get(provider, "")

    @staticmethod
    def _fit_shots_to_model(
        clip_seconds: float, max_seconds: float, provider: str, model: str
    ) -> tuple[float, float]:
        """Aim the shot lengths at the clip lengths the video model can actually produce.

        Some endpoints offer a fixed menu - kling's is 5 or 10 seconds - and a shot that
        falls between two entries has to be covered by the longer one, so a 6 s shot buys
        a 10 s clip and throws four seconds away. Planning the shots on the endpoint's own
        grid keeps the cuts where the music puts them and stops the run paying for footage
        it will trim off.
        """
        if provider != "fal" or not model:
            return clip_seconds, max_seconds
        try:
            offered = render_module.fal_video_durations(model)
        except Exception:
            return clip_seconds, max_seconds
        if not offered:
            return clip_seconds, max_seconds
        # the longest length at or below what was asked for, so a shot is never planned
        # longer than the clip that has to fill it
        fits = [value for value in offered if value <= clip_seconds + 0.01]
        target = fits[-1] if fits else offered[0]
        ceiling = min(max_seconds, offered[-1])
        if abs(target - clip_seconds) > 0.01 or ceiling < max_seconds:
            log(
                f"'{model}' produces {', '.join(f'{value:g}s' for value in offered)} clips; "
                f"planning shots around {target:g}s (max {ceiling:g}s) so none is padded or wasted"
            )
        return target, max(target, ceiling)

    @staticmethod
    def _shot_audio(
        clips: list[dict], provider: str, model: str, wanted: bool
    ) -> list[str]:
        """Each shot's own audio, encoded, but only where there is somewhere to send it.

        Most image-to-video endpoints declare no audio input at all, and OpenRouter's video
        API has none whatsoever - encoding for those would burn CPU and inflate the request
        for nothing. Saying so out loud matters: a run that quietly drops the vocal looks
        exactly like a model that ignored it.
        """
        empty = [""] * len(clips)
        if not wanted or not clips:
            return empty
        if provider != "fal":
            warn(
                "OpenRouter's video API has no input for an audio track, so the shot audio "
                "cannot be sent. Use fal with a model that takes driving audio - for example "
                "fal-ai/wan/v2.7/image-to-video - if you want the performance to follow the vocal."
            )
            return empty
        field = render_module.fal_audio_field(model)
        if not field:
            warn(
                f"'{model}' declares no input for a driving audio track, so lipsync_audio does "
                "nothing here. fal-ai/wan/v2.7/image-to-video takes one (audio_url), and "
                "minimax/h3/reference-to-video takes one as a reference."
            )
            return empty
        # H3 states 2-15 s per reference clip; the driving-audio endpoints allow more, so the
        # tighter window is used for the list-shaped field only
        longest = 15.0 if field in render_module.AUDIO_LIST_FIELDS else 30.0
        uris: list[str] = []
        for index, clip in enumerate(clips):
            try:
                seconds = audio_io.duration(clip)
                if seconds < 2.0 or seconds > longest:
                    warn(
                        f"shot {index + 1} is {seconds:.1f}s, outside the 2-{longest:.0f}s window "
                        f"{model} accepts for audio; that shot goes out without it"
                    )
                    uris.append("")
                    continue
                uris.append(audio_io.data_uri(clip, "mp3"))
            except Exception as exc:
                warn(f"could not encode the audio of shot {index + 1} ({exc})")
                uris.append("")
        sent = sum(1 for uri in uris if uri)
        if sent:
            log(f"sending {sent}/{len(clips)} shot audio clip(s) to {model} as '{field}'")
        return uris

    @staticmethod
    def _frame_model(
        provider: str,
        image_model: str,
        edit_model: str,
        has_references: bool,
        client,
    ) -> tuple[str, object]:
        """Which model draws the start frames, now that there is an identity to keep.

        Most fal image models are text-to-image and declare no input for a reference
        image at all, so handing them one is a silent no-op - which is exactly why every
        shot used to come back as a different person. Only an edit endpoint can be told
        what to keep, so once references exist the run switches to the edit model.
        OpenRouter needs no switch: its image API takes ``input_references`` on almost
        every model it lists.
        """
        if provider != "fal" or not has_references:
            return image_model, client
        chosen = (edit_model or "").strip()
        if chosen and not chosen.startswith("("):
            if not render_module.fal_image_reference_field(chosen):
                warn(
                    f"'{chosen}' declares no field for a reference image, so the subject sheets "
                    "and the style anchor will be ignored. Pick a model whose id ends in /edit."
                )
            return chosen, client
        if not render_module.fal_image_reference_field(image_model):
            warn(
                f"'{image_model}' is a text-to-image model: it has no field for a reference image, "
                "so identity and style cannot carry between shots. Set fal_image_edit_model to an "
                "edit model (for example fal-ai/nano-banana-pro/edit) to keep one face and one look."
            )
        return image_model, client

    @staticmethod
    def _check_video_inputs(
        video_model: str,
        video_prompt_source: str,
        wants_images: bool,
        render_subject_sheets: bool,
    ) -> None:
        """Stop an image-to-video model that has no image coming, before anything is paid for.

        fal endpoints differ: MiniMax H3 will animate from text alone, Wan and Kling
        will not. Asked here, the run fails in a second instead of after the LLM and
        a full set of images have already been billed.
        """
        needs = render_module.fal_video_needs_image(video_model)
        if not needs:
            return
        use_reference = (video_prompt_source or "i2va").lower() == "ref2va"
        supplied = (wants_images and render_subject_sheets) if use_reference else wants_images
        if supplied:
            return
        source = "subject sheets" if use_reference else "a first frame"
        raise ValueError(
            f"{PREFIX} '{video_model}' requires {needs}, so every clip needs {source}. "
            f"Set image_provider (and {'render_subject_sheets' if use_reference else 'keep video_prompt_source on i2va'}), "
            "or pick a text-to-video model such as minimax/h3/text-to-video."
        )

    @classmethod
    def validate_inputs(  # noqa: PLR0913 - one argument per model dropdown, by design
        cls,
        lm_model=None,
        openrouter_model=None,
        openai_model=None,
        anthropic_model=None,
        fal_image_model=None,
        openrouter_image_model=None,
        fal_video_model=None,
        openrouter_video_model=None,
    ) -> bool:
        """Let the model dropdowns hold a value the schema snapshot has not seen yet.

        The lists refresh from the pack's HTTP route while ComfyUI runs, so the value
        the user picked can be newer than the options ComfyUI last read. Naming these
        inputs here - and only these - skips the "Value not in list" check for them
        without weakening validation anywhere else on the node.
        """
        return True

    @classmethod
    def _concat(
        cls,
        paths: list[str],
        seconds: list[float],
        audio,
        audio_mode: str,
        fit: str,
        fps: float,
        crf: int,
        prefix: str,
    ):
        """Glue the clips into one film. Returns (VIDEO or None, info dict)."""
        target = os.path.join(
            render_module.output_directory(),
            f"{prefix or 'music2prompts'}_{time.strftime('%Y%m%d-%H%M%S')}_final.mp4",
        )
        try:
            info = video_module.concat_clips(
                paths,
                target,
                audio=audio,
                audio_mode=audio_mode,
                clip_durations=seconds,
                fps=float(fps) or None,
                fit=fit,
                crf=int(crf),
            )
        except Exception as exc:
            warn(f"could not assemble the final video: {exc}")
            return None, {"error": str(exc)}
        return (cls._videos_from_paths([info["path"]]) or [None])[0], info

    @staticmethod
    def _videos_from_paths(paths: list[str]) -> list:
        """Wrap saved clips in ComfyUI's VIDEO type so SaveVideo/PreviewVideo accept them."""
        videos = []
        try:
            from comfy_api.input_impl import VideoFromFile  # type: ignore
        except Exception as exc:  # pragma: no cover - older ComfyUI
            warn(f"this ComfyUI has no VIDEO input type ({exc}); the clips are on disk only")
            return videos
        for path in paths:
            try:
                videos.append(VideoFromFile(path))
            except Exception as exc:
                warn(f"could not open {path} as VIDEO: {exc}")
        return videos

    @staticmethod
    def _build_shot(
        slot: ShotSlot,
        item: dict,
        subjects: list[dict],
        art: dict,
        lyrics_language: str,
        style_directive: str,
        include_dialogue: bool,
    ) -> H3Shot:
        wanted = {str(name).strip().lower() for name in as_list(item.get("subjects")) if str(name).strip()}
        selected: list[Subject] = []
        for subject in subjects:
            name = first_str(subject, "name")
            if not name:
                continue
            key = name.lower()
            if not wanted or key in wanted or any(key in candidate or candidate in key for candidate in wanted):
                selected.append(
                    Subject(
                        name=name,
                        kind=first_str(subject, "kind", default="character"),
                        description=first_str(subject, "description", default=name),
                        identity_lock=first_str(subject, "identity_lock"),
                    )
                )
        if not selected and subjects:
            first = subjects[0]
            selected = [
                Subject(
                    name=first_str(first, "name", default="main subject"),
                    kind=first_str(first, "kind", default="character"),
                    description=first_str(first, "description", default="the main on-screen subject"),
                    identity_lock=first_str(first, "identity_lock"),
                )
            ]

        speakers: list[Speaker] = []
        line = first_str(item, "dialogue")
        mode = first_str(item, "dialogue_mode", default="none").lower()
        if include_dialogue and line and mode != "none":
            speakers.append(
                Speaker(
                    description=first_str(item, "speaker", default="the performer"),
                    line=line,
                    language=lyrics_language,
                    mode=mode if mode in {"spoken", "sung", "voiceover"} else "sung",
                )
            )

        on_screen = [first_str(item, "on_screen_text")] if first_str(item, "on_screen_text") else []
        opening = first_str(item, "opening") or (
            f"the scene described for section {slot.section or 'the track'} is established in frame"
        )
        return H3Shot(
            index=slot.index,
            duration=slot.duration,
            style=first_str(art, "visual_style", default="Live-action, cinematic"),
            opening=opening,
            action=first_str(item, "action", default="the subject continues the motion begun in the first frame"),
            camera=first_str(item, "camera"),
            diegetic_sound=first_str(item, "diegetic_sound"),
            on_screen_text=on_screen,
            soundscape=first_str(item, "soundscape"),
            music=first_str(item, "music", default="N/A"),
            speakers=speakers,
            subjects=selected,
            extra_style_directive=style_directive,
        )

    @staticmethod
    def _merge_negatives(*sources: str) -> str:
        """Merge negative prompt fragments, dropping duplicates and empty parts."""
        seen: set[str] = set()
        merged: list[str] = []
        for source in sources:
            for raw in str(source or "").replace(";", ",").split(","):
                part = raw.strip().strip(".")
                key = part.lower()
                if part and key not in seen:
                    seen.add(key)
                    merged.append(part)
        return ", ".join(merged)

    @staticmethod
    def _fallback_image_prompt(shot: H3Shot, art: dict, aspect_ratio: str) -> str:
        pieces = [
            shot.opening,
            shot.action,
            first_str(art, "lighting"),
            first_str(art, "lens_and_texture"),
            ", ".join(str(color) for color in as_list(art.get("color_palette"))),
            first_str(art, "visual_style"),
            f"{aspect_ratio} framing",
        ]
        return ", ".join(piece for piece in pieces if piece).strip(", ")

    @staticmethod
    def _fallback_reference_prompt(subject: dict, art: dict, aspect_ratio: str) -> str:
        pieces = [
            first_str(subject, "reference_prompt_hint", "description", "name"),
            first_str(subject, "identity_lock"),
            "clean neutral background, even soft lighting, full subject visible, reference sheet",
            first_str(art, "visual_style"),
            f"{aspect_ratio} framing",
        ]
        return ", ".join(piece for piece in pieces if piece).strip(", ")

    @staticmethod
    def _save_text(payload: str, prefix: str, kind: str, extension: str, stamp: str = "") -> str:
        """Write one sidecar file next to the run's images and clips."""
        directory = render_module.output_directory()
        stamp = stamp or render_module.run_stamp()
        path = os.path.join(directory, f"{prefix or 'music2prompts'}_{stamp}_{kind}.{extension}")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload)
            log(f"{kind} written to {path}")
            return path
        except OSError as exc:
            warn(f"could not write {path}: {exc}")
            return ""

    @staticmethod
    def _transcript_document(
        transcription: dict, slots: list[ShotSlot], video_prompts: list[str], image_prompts: list[str]
    ) -> str:
        """The transcript as a readable document, cut into the same shots as the film.

        The bare Whisper text is on the node's own output; what is worth keeping on disk
        is the text lined up with the shots it belongs to, next to the prompt each shot
        was rendered from - that is what you read when a shot came out wrong.
        """
        words = as_list(transcription.get("words"))
        lines = [
            f"language: {transcription.get('language') or 'unknown'}",
            f"shots: {len(slots)}",
            "",
            "=== transcript ===",
            (transcription.get("text") or "").strip() or "(no speech detected)",
            "",
            "=== per shot ===",
        ]
        for index, slot in enumerate(slots):
            spoken = " ".join(
                # asr.transcribe writes each word as {"start", "end", "text"}; "word" is
                # what the raw transformers pipeline calls it, so both are accepted
                str(word.get("text") or word.get("word") or "").strip()
                for word in words
                if isinstance(word, dict)
                and slot.start <= float(word.get("start", -1)) < slot.end
            ).strip()
            lines += [
                "",
                f"[shot {slot.index}] {slot.start:.2f}s - {slot.end:.2f}s "
                f"({slot.duration:.2f}s, {slot.section or 'section n/a'})",
                f"  lyrics: {spoken or '(instrumental)'}",
            ]
            if index < len(image_prompts):
                lines.append(f"  image : {image_prompts[index]}")
            if index < len(video_prompts):
                lines.append(f"  video : {video_prompts[index].splitlines()[0][:200]}")
        return "\n".join(lines) + "\n"


class _ProgressReporter:
    """Thin wrapper around ComfyUI's progress bar that also logs."""

    def __init__(self, total: int, verbose: bool) -> None:
        self.verbose = verbose
        self.done = 0
        self.total = total
        self.bar = None
        try:
            from comfy.utils import ProgressBar  # type: ignore

            self.bar = ProgressBar(total)
        except Exception:
            self.bar = None

    def step(self, message: str) -> None:
        self.done += 1
        log(f"[{self.done}/{self.total}] {message}")
        if self.bar is not None:
            try:
                self.bar.update_absolute(min(self.done, self.total))
            except Exception:
                pass
