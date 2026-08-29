"""The Music2Prompts node: audio in, prompts out - and, optionally, rendered media."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from comfy_api.latest import io

from . import asr as asr_module
from . import model_cache
from . import music_dsp
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
            display_name="🎵 Music → Prompts (LM Studio + Whisper)",
            category="Music2Prompts",
            description=(
                "Analyses an audio track locally (Whisper large-v3 + librosa), directs it with an LLM "
                "(LM Studio by default, or OpenRouter / OpenAI / Anthropic) and returns ready-to-use "
                "prompts: start-frame image prompts, subject reference prompts and MiniMax H3 video "
                "prompts (I2VA and Ref2VA), plus shot timings and per-shot audio for lipsync. "
                "Image and video rendering through fal.ai or OpenRouter is optional and off by default - "
                "those providers bill per call."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="Track to convert into prompts."),
                io.String.Input(
                    "instruction",
                    multiline=True,
                    default="Cinematic music video. Describe the mood, story or visual world you want.",
                    tooltip="Your brief: story, mood, world, constraints.",
                ),
                io.Combo.Input(
                    "llm_provider",
                    options=list(LLM_PROVIDERS),
                    default="lmstudio",
                    tooltip=(
                        "Who writes the prompts. 'lmstudio' is local and free; the other three are "
                        "paid APIs. Only the model dropdown of the selected provider stays visible."
                    ),
                ),
                io.Combo.Input(
                    "lm_model",
                    options=options["lmstudio"],
                    tooltip="Model served by LM Studio. The list is read from the running server.",
                ),
                io.Combo.Input(
                    "openrouter_model", options=options["openrouter_llm"],
                    tooltip="Model used when llm_provider = openrouter.",
                ),
                io.Combo.Input(
                    "openai_model", options=options["openai_llm"],
                    tooltip="Model used when llm_provider = openai.",
                ),
                io.Combo.Input(
                    "anthropic_model", options=options["anthropic_llm"],
                    tooltip=(
                        "Model used when llm_provider = anthropic. Structured stages go through forced "
                        "tool use; temperature is not sent because current Claude models reject it."
                    ),
                ),
                io.String.Input(
                    "visual_style",
                    default="",
                    tooltip="Force a look, e.g. 'grainy 16mm night photography'. Empty = the model decides.",
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=ASPECT_RATIOS,
                    default="16:9",
                    tooltip="Framing used when writing the image prompts.",
                ),
                io.Float.Input(
                    "clip_seconds",
                    default=6.0,
                    min=1.0,
                    max=60.0,
                    step=0.5,
                    tooltip="Target length of one shot. MiniMax H3 accepts 5-15 s per clip.",
                ),
                io.Float.Input(
                    "min_shot_seconds", default=5.0, min=0.5, max=60.0, step=0.5,
                    tooltip="Shortest allowed shot. MiniMax H3 will not accept anything under 5 s.",
                ),
                io.Float.Input(
                    "max_shot_seconds", default=15.0, min=1.0, max=60.0, step=0.5,
                    tooltip="Longest allowed shot. MiniMax H3 will not accept anything over 15 s.",
                ),
                io.Int.Input(
                    "num_shots",
                    default=0,
                    min=0,
                    max=400,
                    tooltip="0 = derive the shot count from the track length and pacing.",
                ),
                io.Float.Input(
                    "creativity", default=0.7, min=0.0, max=1.0, step=0.05,
                    tooltip="0 = grounded and literal, 1 = bold and surreal.",
                ),
                io.Float.Input(
                    "dynamicity", default=0.6, min=0.0, max=1.0, step=0.05,
                    tooltip="0 = long calm shots, 1 = short kinetic cutting.",
                ),
                io.Float.Input(
                    "word_influence", default=0.6, min=-1.0, max=1.0, step=0.1,
                    tooltip="+1 = visualise the lyrics literally, -1 = ignore the words, use the vibe.",
                ),
                io.Combo.Input(
                    "whisper_device", options=DEVICES, default="auto",
                    tooltip="Where Whisper runs. 'auto' picks cuda:0 when available.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip="Forwarded to the LLM and to the image/video models for reproducible runs.",
                ),
                io.Combo.Input(
                    "image_provider",
                    options=list(MEDIA_PROVIDERS),
                    default="none",
                    tooltip=(
                        "Render the start frames here. 'none' = prompts only (free). "
                        "fal.ai and OpenRouter bill per generated image."
                    ),
                ),
                io.Combo.Input(
                    "fal_image_model",
                    options=options["fal_image"],
                    tooltip="Image model used when image_provider = fal.",
                ),
                io.Combo.Input(
                    "openrouter_image_model",
                    options=options["openrouter_image"],
                    tooltip="Image model used when image_provider = openrouter.",
                ),
                io.Combo.Input(
                    "video_provider",
                    options=list(MEDIA_PROVIDERS),
                    default="none",
                    tooltip=(
                        "Render the clips here. 'none' = prompts only (free). "
                        "fal.ai and OpenRouter bill per generated second of video."
                    ),
                ),
                io.Combo.Input(
                    "fal_video_model",
                    options=options["fal_video"],
                    tooltip=(
                        "Video model used when video_provider = fal. The MiniMax H3 endpoints match "
                        "the prompts this node writes; pick a 'reference-to-video' id for Ref2VA."
                    ),
                ),
                io.Combo.Input(
                    "openrouter_video_model",
                    options=options["openrouter_video"],
                    tooltip="Video model used when video_provider = openrouter.",
                ),
                io.Int.Input(
                    "render_concurrency", default=2, min=1, max=16,
                    tooltip="How many images/clips are rendered at the same time.",
                ),
                # ------------------------------------------------------------------ advanced
                io.String.Input(
                    "lm_url", default=DEFAULT_URL, advanced=True,
                    tooltip="LM Studio server address.",
                ),
                io.String.Input(
                    "lm_api_key", default="", advanced=True,
                    tooltip="Only needed when LM Studio is configured to require a token.",
                ),
                io.String.Input(
                    "lm_model_override", default="", advanced=True,
                    tooltip="Use this model key instead of the dropdown (handy when the server was offline).",
                ),
                io.Boolean.Input(
                    "lm_auto_download", default=True, advanced=True,
                    tooltip="Download the model in LM Studio when it is not installed yet.",
                ),
                io.Boolean.Input(
                    "lm_auto_load", default=True, advanced=True,
                    tooltip="Load the model (and reload it when the loaded context is too small).",
                ),
                io.Int.Input(
                    "lm_context_length", default=32768, min=4096, max=262144, step=4096, advanced=True,
                    tooltip="Context requested when loading the model.",
                ),
                io.Boolean.Input(
                    "lm_unload_after", default=False, advanced=True,
                    tooltip="Free the LLM from memory when the node finishes.",
                ),
                io.Float.Input(
                    "lm_temperature", default=0.8, min=0.0, max=2.0, step=0.05, advanced=True,
                    tooltip="Sampling temperature for prompt writing.",
                ),
                io.Int.Input(
                    "lm_max_tokens", default=4096, min=256, max=32768, step=256, advanced=True,
                    tooltip="Maximum tokens per LLM reply.",
                ),
                io.Int.Input(
                    "lm_timeout", default=300, min=30, max=3600, advanced=True,
                    tooltip="HTTP timeout per LLM request, in seconds.",
                ),
                io.Int.Input(
                    "lm_retries", default=2, min=0, max=5, advanced=True,
                    tooltip="Retries per stage. The last retry drops the JSON schema and parses loosely.",
                ),
                io.Combo.Input(
                    "lm_reasoning_effort",
                    options=["none", "low", "medium", "high", "default"],
                    default="none",
                    advanced=True,
                    tooltip=(
                        "Thinking budget. Keep 'none' for reasoning models such as Gemma 4 or Qwen3 - "
                        "otherwise they spend the whole token budget thinking and return nothing."
                    ),
                ),
                io.Int.Input(
                    "shots_per_request", default=4, min=1, max=16, advanced=True,
                    tooltip="How many shots the model writes per request. Lower is safer for small models.",
                ),
                io.Int.Input(
                    "guide_excerpt_chars", default=0, min=0, max=40000, step=1000, advanced=True,
                    tooltip=(
                        "Inject this many characters of the official MiniMax H3 guides (when "
                        "ComfyUI-MiniMaxH3-Easy is installed). Needs a large context; 0 = compact rules only."
                    ),
                ),
                io.String.Input(
                    "openrouter_api_key", default="", advanced=True,
                    tooltip="Empty = read OPENROUTER_API_KEY from the environment.",
                ),
                io.String.Input(
                    "openai_api_key", default="", advanced=True,
                    tooltip="Empty = read OPENAI_API_KEY from the environment.",
                ),
                io.String.Input(
                    "anthropic_api_key", default="", advanced=True,
                    tooltip="Empty = read ANTHROPIC_API_KEY from the environment.",
                ),
                io.String.Input(
                    "fal_api_key", default="", advanced=True,
                    tooltip="Empty = read FAL_KEY (or FAL_API_KEY) from the environment.",
                ),
                io.Combo.Input(
                    "video_prompt_source", options=list(VIDEO_PROMPT_SOURCES), default="i2va",
                    advanced=True,
                    tooltip=(
                        "Which prompt feeds the video model: 'i2va' uses the rendered start frame as "
                        "the first frame, 'ref2va' uses the rendered subject sheets as references."
                    ),
                ),
                io.Boolean.Input(
                    "render_subject_sheets", default=False, advanced=True,
                    tooltip=(
                        "Also render one reference image per subject. Required for ref2va video and "
                        "billed like any other image."
                    ),
                ),
                io.Boolean.Input(
                    "save_rendered_video", default=True, advanced=True,
                    tooltip="Write the clips into ComfyUI/output/music2prompts (needed for the VIDEO output).",
                ),
                io.Boolean.Input(
                    "concat_video", default=True, advanced=True,
                    tooltip=(
                        "Glue the rendered clips into one finished film on the final_video output. "
                        "Every clip is trimmed or held to the exact length of its shot."
                    ),
                ),
                io.Combo.Input(
                    "final_audio", options=list(AUDIO_MODES), default="music", advanced=True,
                    tooltip=(
                        "Soundtrack of the finished film: 'music' uses the track you fed in, "
                        "'clips' keeps the audio the video model generated, 'none' leaves it silent."
                    ),
                ),
                io.Combo.Input(
                    "final_fit", options=list(FIT_MODES), default="pad", advanced=True,
                    tooltip="What to do with a clip whose aspect differs: letterbox it, stretch it or crop it.",
                ),
                io.Float.Input(
                    "final_fps", default=0.0, min=0.0, max=120.0, step=1.0, advanced=True,
                    tooltip="Frame rate of the finished film. 0 = the fastest rate among the clips.",
                ),
                io.Int.Input(
                    "final_crf", default=20, min=0, max=51, advanced=True,
                    tooltip="x264 quality of the finished film: lower is better and bigger. 20 is a good default.",
                ),
                io.Int.Input(
                    "render_timeout", default=600, min=60, max=3600, advanced=True,
                    tooltip="Seconds to wait for one image or clip before giving up on it.",
                ),
                io.Combo.Input(
                    "whisper_model", options=list(asr_module.SUPPORTED_MODELS),
                    default="openai/whisper-large-v3", advanced=True,
                    tooltip="Downloaded on first use into ComfyUI/models/whisper.",
                ),
                io.Combo.Input(
                    "whisper_dtype", options=["float16", "float32"], default="float16", advanced=True,
                    tooltip="float16 on GPU (Turing-safe), float32 on CPU.",
                ),
                io.String.Input(
                    "whisper_language", default="auto", advanced=True,
                    tooltip="'auto' or an ISO code such as pl / en / de.",
                ),
                io.Int.Input(
                    "whisper_chunk_length_s", default=30, min=5, max=30, advanced=True,
                    tooltip="Chunk size used for long-form transcription.",
                ),
                io.Int.Input(
                    "whisper_batch_size", default=1, min=1, max=32, advanced=True,
                    tooltip=(
                        "Chunks decoded at once. Word-level timestamps peak around 7 GB even at 1 - "
                        "raise this only on a card with spare VRAM."
                    ),
                ),
                io.Boolean.Input(
                    "whisper_word_timestamps", default=True, advanced=True,
                    tooltip="Word-level timings so lyrics land in the right shot.",
                ),
                io.Float.Input(
                    "whisper_window_seconds", default=30.0, min=0.0, max=600.0, step=5.0, advanced=True,
                    tooltip=(
                        "Audio is transcribed in windows of this length because word-timestamp memory "
                        "grows with total audio length (~7 GB per 60 s). 0 disables windowing."
                    ),
                ),
                io.Boolean.Input(
                    "whisper_keep_loaded", default=True, advanced=True,
                    tooltip="Keep Whisper in memory between runs.",
                ),
                io.Boolean.Input(
                    "whisper_skip", default=False, advanced=True,
                    tooltip="Skip transcription entirely (instrumental tracks).",
                ),
                io.Boolean.Input(
                    "free_comfy_vram", default=True, advanced=True,
                    tooltip="Unload ComfyUI models before Whisper runs.",
                ),
                io.Boolean.Input(
                    "free_lmstudio_vram", default=True, advanced=True,
                    tooltip=(
                        "Unload the LM Studio model before Whisper runs. Keep this on with a single "
                        "GPU - an 11 GB card cannot hold the LLM and Whisper large-v3 at once."
                    ),
                ),
                io.Boolean.Input(
                    "analyze_music", default=True, advanced=True,
                    tooltip="Measure BPM, beat grid, sections and energy with librosa.",
                ),
                io.Boolean.Input(
                    "snap_cuts_to_beats", default=True, advanced=True,
                    tooltip="Move shot boundaries onto the nearest beat or section edge.",
                ),
                io.Float.Input(
                    "audio_clip_padding", default=0.0, min=0.0, max=2.0, step=0.05, advanced=True,
                    tooltip=(
                        "Widen every audio clip by this many seconds on both sides. 0 keeps the cut "
                        "sample-accurate against the shot boundaries (what lipsync wants)."
                    ),
                ),
                io.Int.Input(
                    "max_subjects", default=6, min=0, max=16, advanced=True,
                    tooltip="How many recurring characters/locations/props to lock for consistency.",
                ),
                io.String.Input(
                    "negative_prompt_base", multiline=True, default=DEFAULT_NEGATIVE, advanced=True,
                    tooltip="Prepended to every negative prompt.",
                ),
                io.Boolean.Input(
                    "include_dialogue", default=True, advanced=True,
                    tooltip="Put transcribed lyrics into the H3 <d>[Language] ...</d> blocks.",
                ),
                io.String.Input(
                    "h3_style_directive", default="", advanced=True,
                    tooltip="Extra style clause injected into every H3 prompt.",
                ),
                io.Boolean.Input(
                    "save_json", default=False, advanced=True,
                    tooltip="Write the full analysis to the ComfyUI output folder.",
                ),
                io.String.Input(
                    "filename_prefix", default="music2prompts", advanced=True,
                    tooltip="Prefix for the saved JSON file.",
                ),
                io.Boolean.Input(
                    "verbose", default=False, advanced=True,
                    tooltip="Log per-stage timing and payload sizes.",
                ),
                io.Image.Input(
                    "reference_images", optional=True,
                    tooltip="Optional look/identity references, described by the vision model.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="image_prompts_start", is_output_list=True),
                io.String.Output(display_name="image_prompts_reference", is_output_list=True),
                io.String.Output(display_name="reference_subjects", is_output_list=True),
                io.String.Output(display_name="video_prompts_i2va", is_output_list=True),
                io.String.Output(display_name="video_prompts_ref2va", is_output_list=True),
                io.String.Output(display_name="negative_prompts", is_output_list=True),
                io.Int.Output(display_name="shot_index", is_output_list=True),
                io.Float.Output(display_name="start_times", is_output_list=True),
                io.Float.Output(display_name="end_times", is_output_list=True),
                io.Float.Output(display_name="durations", is_output_list=True),
                io.Audio.Output(display_name="audio_clips", is_output_list=True),
                io.String.Output(display_name="transcript"),
                io.String.Output(display_name="analysis_json"),
                io.Image.Output(display_name="images", is_output_list=True),
                io.Image.Output(display_name="subject_images", is_output_list=True),
                io.Video.Output(display_name="videos", is_output_list=True),
                io.Video.Output(display_name="final_video"),
            ],
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
        save_json: bool = False,
        filename_prefix: str = "music2prompts",
        verbose: bool = False,
        reference_images=None,
    ) -> io.NodeOutput:
        started = time.perf_counter()
        provider = (llm_provider or "lmstudio").strip().lower()
        wants_images = (image_provider or "none").lower() not in {"", "none"}
        wants_video = (video_provider or "none").lower() not in {"", "none"}
        progress = _ProgressReporter(BASE_STAGES + int(wants_images) + int(wants_video), verbose)

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
        )
        local_llm = provider == "lmstudio"
        image_model = fal_image_model if image_provider == "fal" else openrouter_image_model
        video_model = fal_video_model if video_provider == "fal" else openrouter_video_model
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

        for slot in slots:
            item = content.get(slot.index) or {}
            shot = cls._build_shot(
                slot, item, subjects, art, lyrics_language, h3_style_directive, include_dialogue
            )
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

        if wants_images:
            progress.step(f"rendering {len(start_frames)} start frame(s) with {image_provider}")
            image_client = render_module.make_media_client(
                image_provider, fal_api_key if image_provider == "fal" else openrouter_api_key,
                timeout=render_timeout, verbose=verbose,
            )
            image_errors: list[Exception] = []
            image_payloads = render_module.render_images(
                image_client,
                image_model,
                [
                    ImageRequest(
                        prompt=prompt,
                        negative=negatives[index],
                        aspect_ratio=aspect_ratio,
                        seed=(int(seed) + index) if seed else None,
                        references=list(reference_uris),
                        label=f"shot {index + 1}",
                    )
                    for index, prompt in enumerate(start_frames)
                ],
                render_concurrency,
                image_errors,
            )
            if not any(image_payloads):
                raise RuntimeError(
                    f"{PREFIX} every image render failed on {image_provider}/{image_model}. "
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
            if render_subject_sheets and subject_prompts:
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
                )
                subject_images_out = [
                    render_module.image_bytes_to_tensor(data) for data in subject_payloads if data
                ]
            log(f"{len(images_out)}/{len(start_frames)} start frames rendered")
            _interrupt_check()

        if wants_video:
            progress.step(f"rendering {len(slots)} clip(s) with {video_provider}")
            video_client = render_module.make_media_client(
                video_provider, fal_api_key if video_provider == "fal" else openrouter_api_key,
                timeout=render_timeout, verbose=verbose,
            )
            use_reference = (video_prompt_source or "i2va").lower() == "ref2va"
            subject_uris = [
                render_module.data_uri(data) for data in subject_payloads if data
            ][:9]
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
                        label=f"shot {slot.index}",
                    )
                )
            video_errors: list[Exception] = []
            payloads = render_module.render_videos(
                video_client, video_model, video_requests, render_concurrency, video_errors
            )
            if not any(payloads):
                raise RuntimeError(
                    f"{PREFIX} every clip failed on {video_provider}/{video_model}. "
                    f"First error: {video_errors[0] if video_errors else 'unknown'}"
                )
            rendered_seconds = [slot.duration for slot, item in zip(slots, payloads) if item]
            # the clips are always written out: the VIDEO output and the final cut
            # both need real files. save_rendered_video only decides output vs temp.
            video_paths = render_module.save_videos(
                payloads, filename_prefix, temporary=not save_rendered_video
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
        analysis_json = json.dumps(debug, ensure_ascii=False, indent=2)
        if save_json:
            cls._save_json(analysis_json, filename_prefix)

        log(f"done in {time.perf_counter() - started:.1f}s - {len(slots)} shots, {len(subject_names)} subjects")
        return io.NodeOutput(
            start_frames,
            subject_prompts,
            subject_names,
            i2va,
            ref2va,
            negatives,
            indices,
            starts,
            ends,
            durations,
            audio_clips,
            transcription.get("text", ""),
            analysis_json,
            images_out,
            subject_images_out,
            videos_out,
            final_video,
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
    def _save_json(payload: str, prefix: str) -> None:
        try:
            import folder_paths  # type: ignore

            directory = folder_paths.get_output_directory()
        except Exception:
            directory = os.getcwd()
        name = f"{prefix or 'music2prompts'}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        path = os.path.join(directory, name)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload)
            log(f"analysis written to {path}")
        except OSError as exc:
            warn(f"could not write {path}: {exc}")


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
