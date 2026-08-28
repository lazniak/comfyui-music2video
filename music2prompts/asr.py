"""Local Whisper large-v3 transcription through HuggingFace transformers.

Runs on GPU by default (fp16, Turing-safe - never bf16) with CPU as an
explicit option. Model files are cached under ``ComfyUI/models/whisper``.
"""

from __future__ import annotations

import os
from typing import Any

from .util import PREFIX, log, sanitize_repo_id, warn

WHISPER_SAMPLE_RATE = 16000
SUPPORTED_MODELS = (
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-medium",
)

_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.model",
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer*",
]

_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}


def resolve_device(choice: str) -> tuple[str, str]:
    """Return ``(device, dtype)`` honoring the widget choice and hardware reality."""
    choice = (choice or "auto").strip().lower()
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
        device_count = torch.cuda.device_count() if has_cuda else 0
    except Exception:
        has_cuda, device_count = False, 0

    if choice == "cpu" or not has_cuda:
        if choice not in {"cpu", "auto"} and not has_cuda:
            warn("CUDA not available - falling back to CPU for Whisper.")
        return "cpu", "float32"

    if choice.startswith("cuda:"):
        try:
            index = int(choice.split(":", 1)[1])
        except ValueError:
            index = 0
        if index >= max(1, device_count):
            warn(f"{choice} not present ({device_count} CUDA devices); using cuda:0.")
            index = 0
        return f"cuda:{index}", "float16"
    return "cuda:0", "float16"


def model_directory(repo_id: str) -> str:
    """Where the weights live: ``ComfyUI/models/whisper/<repo--id>``."""
    base = None
    try:
        import folder_paths  # type: ignore

        base = os.path.join(folder_paths.models_dir, "whisper")
    except Exception:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "whisper")
    return os.path.join(base, sanitize_repo_id(repo_id))


def ensure_model_files(repo_id: str) -> str:
    """Download the model into the ComfyUI models folder if it is missing."""
    target = model_directory(repo_id)
    config = os.path.join(target, "config.json")
    weights_present = os.path.isdir(target) and any(
        name.endswith(".safetensors") or name.endswith(".bin") for name in os.listdir(target)
    ) if os.path.isdir(target) else False

    if os.path.isfile(config) and weights_present:
        return target

    log(f"downloading {repo_id} to {target} (first run only, several GB)...")
    from huggingface_hub import snapshot_download

    os.makedirs(target, exist_ok=True)
    try:
        snapshot_download(repo_id=repo_id, local_dir=target, allow_patterns=_ALLOW_PATTERNS)
    except Exception as exc:
        raise RuntimeError(
            f"{PREFIX} could not download '{repo_id}'. Check the internet connection or place the "
            f"model manually in {target}. Original error: {exc}"
        ) from exc
    return target


def free_comfy_vram() -> None:
    try:
        import comfy.model_management as mm  # type: ignore

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:  # pragma: no cover - only meaningful inside ComfyUI
        warn(f"could not free ComfyUI VRAM: {exc}")


def _build_pipeline(model_path: str, device: str, dtype: str, keep_loaded: bool):
    cache_key = (model_path, device, dtype)
    if keep_loaded and cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    import inspect

    import torch
    from transformers import pipeline

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    # transformers >= 5 renamed torch_dtype to dtype; passing the wrong one is silently
    # swallowed by **kwargs and the model loads in fp32 (twice the VRAM).
    dtype_kwarg = "dtype" if "dtype" in inspect.signature(pipeline).parameters else "torch_dtype"
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        **{dtype_kwarg: torch_dtype},
    )
    if keep_loaded:
        _PIPELINE_CACHE.clear()
        _PIPELINE_CACHE[cache_key] = asr
    return asr


def unload_all() -> None:
    _PIPELINE_CACHE.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe(
    samples: Any,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    repo_id: str = "openai/whisper-large-v3",
    device_choice: str = "auto",
    dtype_choice: str = "float16",
    chunk_length_s: int = 30,
    batch_size: int = 1,
    language: str = "auto",
    word_timestamps: bool = True,
    keep_loaded: bool = True,
    window_seconds: float = 30.0,
) -> dict:
    """Transcribe mono 16 kHz audio and return text, language and word timings.

    Word-level timestamps are produced by a DTW pass over the cross-attentions,
    whose memory grows with the length of the *whole* input (not with
    ``chunk_length_s``). Measured on an 11 GB card: ~7 GB for 60 s of audio and
    an out-of-memory failure at 90 s. So the audio is transcribed in windows and
    the timestamps are shifted back into track time.
    """
    device, auto_dtype = resolve_device(device_choice)
    dtype = "float32" if device == "cpu" else (dtype_choice if dtype_choice in {"float16", "float32"} else auto_dtype)

    model_path = ensure_model_files(repo_id)
    log(f"transcribing with {repo_id} on {device} ({dtype})")
    _build_pipeline(model_path, device, dtype, keep_loaded)

    generate_kwargs: dict[str, Any] = {"task": "transcribe"}
    if language and language.strip().lower() not in {"auto", ""}:
        generate_kwargs["language"] = language.strip().lower()

    duration = len(samples) / float(max(1, sample_rate))
    window = float(window_seconds or 0.0)
    context = {
        "model_path": model_path,
        "device": device,
        "dtype": dtype,
        "chunk_length_s": chunk_length_s,
        "batch_size": batch_size,
        "word_timestamps": word_timestamps,
        "keep_loaded": keep_loaded,
        "generate_kwargs": generate_kwargs,
        "sample_rate": sample_rate,
    }

    if window < 5.0 or duration <= window + 5.0:
        return _normalize(_transcribe_slice(samples, 0.0, context), language)

    windows = int(duration // window) + (1 if duration % window > 1.0 else 0)
    log(f"transcribing {duration:.0f}s in {windows} window(s) of {window:.0f}s")
    texts: list[str] = []
    words: list[dict] = []
    offset = 0.0
    while offset < duration - 1.0:
        end = min(duration, offset + window)
        piece = samples[int(offset * sample_rate) : int(end * sample_rate)]
        if len(piece) < sample_rate // 2:
            break
        part = _normalize(_transcribe_slice(piece, offset, context), language)
        if part["text"]:
            texts.append(part["text"])
        words.extend(part["words"])
        offset = end

    text = " ".join(texts).strip()
    return {"text": text, "language": _language_of(text, language), "words": words}


def _transcribe_slice(samples: Any, offset: float, context: dict) -> Any:
    """Run one window through the pipeline, stepping down on out-of-memory."""
    last_error: Exception | None = None
    ladder = _attempt_ladder(
        context["batch_size"], context["device"], context["dtype"], context["word_timestamps"]
    )
    for run_device, run_dtype, batch, timestamps in ladder:
        try:
            pipe = _build_pipeline(context["model_path"], run_device, run_dtype, context["keep_loaded"])
            options = {"return_timestamps": timestamps} if timestamps else {}
            # transformers consumes the input dict, so build a fresh one for every attempt
            result = pipe(
                {"raw": samples, "sampling_rate": int(context["sample_rate"])},
                chunk_length_s=int(context["chunk_length_s"]),
                batch_size=max(1, int(batch)),
                generate_kwargs=context["generate_kwargs"],
                **options,
            )
            if timestamps != "word" and context["word_timestamps"]:
                warn("word-level timestamps were unavailable; lyrics are aligned per segment instead")
            return _shift(result, offset)
        except Exception as exc:
            last_error = exc
            if _is_out_of_memory(exc):
                warn(
                    f"out of memory on {run_device} (batch {batch}, {timestamps or 'no'} timestamps); "
                    "freeing and stepping down"
                )
                unload_all()
                continue
            warn(f"whisper call failed ({timestamps or 'no'} timestamps, batch {batch}): {exc}")
            continue
    raise RuntimeError(f"{PREFIX} transcription failed: {last_error}")


def _shift(result: Any, offset: float) -> Any:
    """Move a window's timestamps back into track time."""
    if offset <= 0 or not isinstance(result, dict):
        return result
    for chunk in result.get("chunks") or []:
        stamp = chunk.get("timestamp")
        if isinstance(stamp, (list, tuple)) and len(stamp) == 2:
            start, end = stamp
            chunk["timestamp"] = (
                None if start is None else float(start) + offset,
                None if end is None else float(end) + offset,
            )
    return result


def _attempt_ladder(
    batch_size: int, device: str, dtype: str, word_timestamps: bool
) -> list[tuple[str, str, int, Any]]:
    """Progressively cheaper attempts, ordered by measured VRAM cost.

    On an 11 GB card whisper-large-v3 needs roughly 2.9 GB for the weights,
    ~7 GB peak with word-level timestamps (the DTW alignment) and ~4 GB with
    segment timestamps, so those are the rungs we step down through.
    """
    batch = max(1, int(batch_size))
    primary: Any = "word" if word_timestamps else True
    ladder: list[tuple[str, str, int, Any]] = [(device, dtype, batch, primary)]
    if batch > 1:
        ladder.append((device, dtype, 1, primary))
    if primary == "word":
        ladder.append((device, dtype, 1, True))
    ladder.append((device, dtype, 1, None))
    if device != "cpu":
        ladder.append(("cpu", "float32", 1, primary))
    return ladder


def _is_out_of_memory(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or exc.__class__.__name__ == "OutOfMemoryError"


def _normalize(result: Any, requested_language: str) -> dict:
    text = ""
    words: list[dict] = []
    if isinstance(result, dict):
        text = str(result.get("text") or "").strip()
        for chunk in result.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            stamp = chunk.get("timestamp") or (None, None)
            try:
                start = float(stamp[0]) if stamp[0] is not None else None
                end = float(stamp[1]) if stamp[1] is not None else start
            except (TypeError, ValueError, IndexError):
                continue
            if start is None:
                continue
            token = str(chunk.get("text") or "").strip()
            if token:
                words.append({"start": round(start, 3), "end": round(end if end else start, 3), "text": token})
    elif isinstance(result, str):
        text = result.strip()

    return {"text": text, "language": _language_of(text, requested_language), "words": words}


def _language_of(text: str, requested: str) -> str:
    if requested and requested.strip().lower() not in {"auto", ""}:
        return requested.strip()
    return _guess_language(text)


def _guess_language(text: str) -> str:
    """Very small heuristic used only for the ``[Language]`` tag in H3 dialogue."""
    if not text:
        return "English"
    lowered = text.lower()
    if any(char in lowered for char in "ąćęłńóśźż"):
        return "Polish"
    if any(char in lowered for char in "äöüß"):
        return "German"
    if any(char in lowered for char in "áéíóúñ¿¡"):
        return "Spanish"
    if any("一" <= char <= "鿿" for char in text):
        return "Chinese"
    if any("぀" <= char <= "ヿ" for char in text):
        return "Japanese"
    return "English"
