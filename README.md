# Music → Prompts (LM Studio + Whisper) — ComfyUI node

Turns an audio track into **ready-to-use generation prompts**. It renders nothing itself: no images,
no video, no cloud API, no extra UI. One node in, twelve outputs out — you wire them into whatever
image or video model you already run in ComfyUI.

Everything runs locally:

| Job | Engine |
|---|---|
| Lyrics + word-level timestamps | **Whisper large-v3** (transformers, GPU by default, CPU selectable) |
| BPM, beat grid, sections, energy | **librosa** when installed, otherwise a built-in numpy/scipy fallback |
| Treatment, art direction, shot writing | **LM Studio** (default model `google/gemma-4-e4b`) |
| MiniMax H3 prompt formatting | deterministic Python renderers — the model fills fields, the code builds the exact skeleton |

Built on the ComfyUI **V3 node schema**, so the 30+ secondary settings collapse into the native
*Advanced* section of the new node layout.

---

## Quick start (PL)

1. Odpal LM Studio → *Developer* → **Start Server** (domyślnie `http://127.0.0.1:1234`).
2. Skopiuj / zlinkuj ten katalog do `ComfyUI/custom_nodes/` i zrestartuj ComfyUI.
3. Dodaj node **🎵 Music → Prompts (LM Studio + Whisper)**, podłącz `LoadAudio`, wpisz brief.
4. Pierwsze uruchomienie ściąga Whisper large-v3 (~3 GB) do `ComfyUI/models/whisper/`.
5. Wyjścia to **listy** — podłącz `image_prompts_start` do dowolnego enkodera tekstu,
   a `video_prompts_i2va` / `video_prompts_ref2va` do node'a MiniMax H3.

Instalacja przez junction (Windows, bez kopiowania):

```bash
cmd /c mklink /J "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\comfyui-music2prompts" "D:\code\comfyui-ultimate-node"
```

---

## Requirements

Nothing beyond a normal ComfyUI install: `requests`, `numpy`, `scipy`, `transformers`,
`huggingface_hub`. `librosa` is optional (better beat/section detection); without it the node uses
its own numpy/scipy DSP.

If you do install extras, install them **into ComfyUI's own environment** and use `--no-user`,
otherwise on Windows the package can land in `%APPDATA%\Python\Python313\site-packages`, which
shadows the ComfyUI venv and silently breaks torch:

```bash
"D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe" -m pip install --no-user librosa
```

LM Studio 0.4+ is required for model management (`/api/v1/models/...`); inference itself uses the
OpenAI-compatible `/v1/chat/completions` endpoint, so older builds still work for generation.

---

## Inputs

### Main

| Widget | Default | Meaning |
|---|---|---|
| `audio` | — | AUDIO from `LoadAudio` (or anything producing AUDIO) |
| `instruction` | — | Your brief: story, mood, world, constraints |
| `lm_model` | first model found | Model list is read live from the running LM Studio |
| `visual_style` | empty | Force a look; empty means the model chooses |
| `aspect_ratio` | `16:9` | Framing used when writing image prompts |
| `clip_seconds` | `6.0` | Target shot length — MiniMax H3 accepts 5–15 s |
| `min_shot_seconds` | `5.0` | Shortest allowed shot (H3 refuses anything below 5 s) |
| `max_shot_seconds` | `15.0` | Longest allowed shot (H3 refuses anything above 15 s) |
| `num_shots` | `0` | `0` = derive the count from track length and pacing |
| `creativity` | `0.7` | 0 grounded → 1 surreal |
| `dynamicity` | `0.6` | 0 long calm shots → 1 short kinetic cutting |
| `word_influence` | `0.6` | +1 literal lyrics → −1 pure atmosphere |
| `whisper_device` | `auto` | `auto`, `cuda:0`, `cuda:1`, `cpu` |
| `seed` | `0` | Forwarded to LM Studio |
| `reference_images` | optional socket | Described by the vision model and locked into every prompt |

### Advanced (collapsed by default)

**LM Studio** — `lm_url`, `lm_api_key`, `lm_model_override`, `lm_auto_download`, `lm_auto_load`,
`lm_context_length`, `lm_unload_after`, `lm_temperature`, `lm_max_tokens`, `lm_timeout`,
`lm_retries`, `lm_reasoning_effort`, `shots_per_request`, `guide_excerpt_chars`.

> `lm_reasoning_effort` defaults to **none** on purpose. Reasoning models (Gemma 4, Qwen3, …)
> otherwise spend the whole token budget on hidden thinking and return an empty message.

**Whisper** — `whisper_model` (large-v3 / large-v3-turbo / medium), `whisper_dtype`,
`whisper_language`, `whisper_chunk_length_s`, `whisper_batch_size`, `whisper_word_timestamps`,
`whisper_window_seconds`, `whisper_keep_loaded`, `whisper_skip`, `free_comfy_vram`,
`free_lmstudio_vram`.

> **VRAM, measured on an 11 GB RTX 2080 Ti.** Weights in fp16 take 2.9 GB. Word-level timestamps add
> a DTW pass whose memory grows with the length of the *whole* input — ~7 GB for 60 s of audio and an
> out-of-memory failure at 90 s — so the node transcribes in windows of `whisper_window_seconds`
> (default 30 s) and shifts the timestamps back into track time. If it still runs out of memory it
> steps down automatically: smaller batch → segment timestamps → CPU.
>
> `free_lmstudio_vram` (on by default) unloads the LM Studio model before Whisper runs, because one
> 11 GB card cannot hold both. `whisper-large-v3-turbo` is much lighter if you prefer speed.

**Music & shots** — `analyze_music`, `snap_cuts_to_beats`, `audio_clip_padding`.

**Prompting** — `max_subjects`, `negative_prompt_base`, `include_dialogue`, `h3_style_directive`.

**Debug** — `save_json`, `filename_prefix`, `verbose`.

---

## Outputs

Ten of the twelve outputs are **lists**. Wire them into any node that iterates a list, or index them.

| # | Output | Aligned with | Contents |
|---|---|---|---|
| 1 | `image_prompts_start` | shots | Natural-language cinematic prompt for the first frame of each shot (Flux / Qwen-Image / Z-Image style) |
| 2 | `image_prompts_reference` | subjects | Clean reference-sheet prompt per recurring subject |
| 3 | `reference_subjects` | subjects | Subject names, same order as #2 |
| 4 | `video_prompts_i2va` | shots | MiniMax H3 **image-to-video** prompt (first frame = the image from #1) |
| 5 | `video_prompts_ref2va` | shots | MiniMax H3 **reference-to-video** prompt (six-section format) |
| 6 | `negative_prompts` | shots | Base negatives + per-shot additions, de-duplicated |
| 7 | `shot_index` | shots | 1…N |
| 8 | `start_times` | shots | Seconds |
| 9 | `end_times` | shots | Seconds |
| 10 | `durations` | shots | Seconds, inside the `min_shot_seconds`…`max_shot_seconds` window |
| 11 | `audio_clips` | shots | **AUDIO** cut sample-accurately to each shot — feed straight into lipsync |
| 12 | `transcript` | — | Full transcription (empty for instrumentals) |
| 13 | `analysis_json` | — | Everything: BPM, beats, sections, treatment, art direction, subject bible, per-shot fields |

### Wiring examples

* **Start frames** → `image_prompts_start` into a text encoder, `negative_prompts` into the negative
  encoder. `durations` / `start_times` drive whatever timing you need downstream.
* **MiniMax H3, image-to-video** → generate the image from `image_prompts_start[i]`, then feed
  `video_prompts_i2va[i]` into `MiniMaxH3Easy.prompt` with `mode = image`, `keyframe_role = first`,
  `seconds = durations[i]`.
* **MiniMax H3, reference-to-video** → generate the subject references from
  `image_prompts_reference`, feed them as media, and use `video_prompts_ref2va[i]` with
  `mode = reference`.
* **Lipsync / audio-driven video** → `audio_clips[i]` is the exact slice of the track between
  `start_times[i]` and `end_times[i]`, at the original sample rate and channel count, so it lines up
  with the shot it belongs to. Wire it into any audio-driven node (S2V, OmniHuman, talking-head,
  `PreviewAudio`, `SaveAudio`) alongside `image_prompts_start[i]` for the same shot. Set the shot
  length the lipsync model expects with `min_shot_seconds` / `max_shot_seconds`, and use the advanced
  `audio_clip_padding` if a model clips the first or last syllable.

---

## Pipeline

```
AUDIO ─┬─ Whisper large-v3 ──── words + timestamps ──┐
       └─ librosa / numpy DSP ─ BPM, beats, sections ┤
                                                     ├─ shot planner (Python: no LLM does timing)
                                                     │      cuts snapped to beats, 5–15 s each
                                                     ▼
              LM Studio ── treatment → art direction → subject bible → per-shot content
                                                     ▼
              deterministic renderers ── I2VA / Ref2VA skeletons, image prompts, negatives
```

The language model is never asked to "listen" or to count time; it receives measured numbers and
fixed shot boundaries and only writes creative content. The MiniMax H3 skeletons are assembled in
Python from the model's structured JSON, so the format is exact regardless of model size.

### MiniMax H3 formats

*I2VA* — an alignment line, then the three core fields:

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

*Ref2VA* — six sections in fixed order: `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`, with `<Subject N>` labels
renumbered per shot so no label is ever undefined.

Camera language is validated against the official vocabulary (Push In, Truck Left, Arc Shot,
Tracking Shot, …) and label-style input is rewritten into a proper sentence. Dialogue and lyrics stay
verbatim in their original language inside `<d>[Language] …</d>`.

When `ComfyUI-MiniMaxH3-Easy` is installed next to this pack, `guide_excerpt_chars` can inject part
of the official H3 guides into the system prompt (needs a large context window).

---

## Tests

```bash
python -m pytest tests -q
```

Formatting and timing tests are dependency-free. The end-to-end plumbing test needs ComfyUI on
`PYTHONPATH`; without pytest you can run the same checks directly:

```bash
python tests/run_pipeline_check.py
```

---

## Notes & limits

* One LM Studio request per stage plus one per shot batch — expect a few minutes per track on a
  small local model (≈5 min for 45 s of audio with `gemma-4-e4b` on a 2080 Ti).
* Whisper large-v3 needs ≈2.9 GB VRAM in fp16 for the weights and up to ~7 GB during word-timestamp
  alignment; `free_comfy_vram` and `free_lmstudio_vram` make room before it loads.
* fp16 only on GPU (bf16 is deliberately not used — Turing cards do not support it); CPU forces fp32.
* Shots always tile the whole track with no gaps, and every shot stays inside the
  `min_shot_seconds`…`max_shot_seconds` window (5–15 s by default, which is what MiniMax H3 accepts).
  `audio_clips` tile the track the same way, so concatenating them reproduces the input audio.
* Prompts are always written in English; lyrics, dialogue and on-screen text keep their original
  language.
