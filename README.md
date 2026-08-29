# Music → Prompts (LM Studio + Whisper) — ComfyUI node

Turns an audio track into **ready-to-use generation prompts** — and, if you want, into the images,
the clips, and the finished cut-together film. One node in, seventeen outputs out.

The analysis is always local and free. The prompt writing is local by default and can be moved to a
cloud LLM. Rendering is **off by default** and only happens when you pick a provider for it.

| Job | Engine |
|---|---|
| Lyrics + word-level timestamps | **Whisper large-v3** (transformers, GPU by default, CPU selectable) |
| BPM, beat grid, sections, energy | **librosa** when installed, otherwise a built-in numpy/scipy fallback |
| Treatment, art direction, shot writing | **LM Studio** (default, `google/gemma-4-e4b`) or **OpenRouter** / **OpenAI** / **Anthropic** |
| MiniMax H3 prompt formatting | deterministic Python renderers — the model fills fields, the code builds the exact skeleton |
| Optional image rendering | **fal.ai** or **OpenRouter** (`image_provider`, default `none`) |
| Optional video rendering | **fal.ai** or **OpenRouter** (`video_provider`, default `none`) |
| Assembling the final film | **PyAV** — clips trimmed to their shots, music muxed in, no ffmpeg binary needed |

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

## Providers

Every provider gets **its own model dropdown**. The lists are fetched in the background just after
ComfyUI starts (never while a node is being built, so nothing blocks) and are served from one shared
cache with a TTL. A small frontend script keeps only the dropdown of the selected provider on screen
— `llm_provider = openrouter` shows `openrouter_model` and hides the rest — and pulls fresh lists
from the pack's own route.

**Refreshing without a restart:** right-click the node → **Refresh model lists**. Start LM Studio,
load a model, export a key — then refresh and it appears. The lists are *monotonic*: a model that was
offered once is never removed, so a saved workflow keeps validating even when a provider is
temporarily unreachable.

| Provider | Used for | Key | Endpoint |
|---|---|---|---|
| LM Studio | LLM (default) | usually none | `/v1/chat/completions` + `/api/v1/models/...` |
| OpenRouter | LLM, images, video | `OPENROUTER_API_KEY` | `/chat/completions`, `/images`, `/videos` |
| OpenAI | LLM | `OPENAI_API_KEY` | `/chat/completions` with strict JSON schema |
| Anthropic | LLM | `ANTHROPIC_API_KEY` | `/v1/messages`, structured stages via forced tool use |
| fal.ai | images, video | `FAL_KEY` (or `FAL_API_KEY`) | queue API `https://queue.fal.run/<model>` |

Keys come from the node widgets first and from those environment variables when the widget is empty;
they are never written into the outputs or into `analysis_json`. Seeds are folded into the 32-bit
range the media providers accept (ComfyUI's seed widget goes far higher, and OpenRouter rejects it
outright).

> **These are paid, per-call APIs.** LM Studio and the whole analysis are free; every other provider
> bills you. Image and video rendering therefore stays `none` until you choose otherwise, and one
> run renders one image per shot (plus one per subject with `render_subject_sheets`) and one clip per
> shot. A 3-minute track cut into 6-second shots is 30 shots — check the provider's price per image
> and per second of video before you start it.

### Rendering

* `image_provider` + `fal_image_model` / `openrouter_image_model` render the start frames. The result
  arrives on the `images` output, aligned with the shots.
* `video_provider` + `fal_video_model` / `openrouter_video_model` render the clips. With
  `video_prompt_source = i2va` the rendered start frame is sent as the first frame; with `ref2va` the
  rendered subject sheets are sent as references (turn on `render_subject_sheets`), which is what the
  MiniMax H3 `reference-to-video` endpoints expect.
* `render_concurrency` decides how many images or clips are in flight at once. Failures do not stop
  the run: the failing entry is logged and skipped, everything else still comes back.
* `live_preview` (on by default) shows each image and clip **inside the node the moment it is
  rendered**, with arrows to page back and forth - a batch of twelve shots takes minutes, and this is
  what lets you spot a bad prompt at shot 1 instead of paying for the rest. The gallery clears at the
  start of the next run. It comes back if you reopen the finished job from the queue sidebar, but not
  after a plain F5: node previews live only in the browser's memory, and the preview files sit in
  `ComfyUI/temp/`, which is wiped when ComfyUI restarts. The real results are on the `images` /
  `videos` outputs, and `save_rendered_video` keeps the clips in `output/`.
* fal payloads are built from each endpoint's own published schema, so a model is sent the field
  names it actually has (`start_image_url` for Wan, `image_url` for MiniMax H3) and only the options
  it declares. Picking an image-to-video model with `image_provider = none` is refused before the run
  starts, rather than after the LLM and the images have been billed.
* Clips are written to `ComfyUI/output/music2prompts/` and returned on the `videos` output (the
  regular VIDEO type, so `SaveVideo` / `PreviewVideo` accept them).
* `concat_video` (on by default) glues them into **one finished film** on the `final_video` output.
  Every clip is placed on a single grid — one size, one frame rate — and trimmed or frozen on its
  last frame so it lasts exactly as long as its shot; the film therefore stays in sync with the
  track. Clips whose aspect differs are letterboxed (`final_fit`), and `final_audio` decides the
  soundtrack: `music` (your track), `clips` (the audio the video model generated) or `none`.
* When a single shot fails to render, its slot in `images` stays filled with a black frame so the
  list still lines up with the shots. When *every* shot fails, the node raises with the provider's
  own error instead of returning empty lists — an empty list wired into another node makes ComfyUI
  fail with a bare `IndexError: list index out of range`.
* Do not wire `images` / `videos` / `final_video` while the matching provider is `none`: those
  outputs are then empty, and the same `IndexError` follows.

---

## Inputs

### Main

| Widget | Default | Meaning |
|---|---|---|
| `audio` | — | AUDIO from `LoadAudio` (or anything producing AUDIO) |
| `instruction` | — | Your brief: story, mood, world, constraints |
| `llm_provider` | `lmstudio` | `lmstudio` (local, free), `openrouter`, `openai`, `anthropic` |
| `lm_model` | first model found | Model list is read live from the running LM Studio |
| `openrouter_model` / `openai_model` / `anthropic_model` | first model found | One live list per cloud provider |
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
| `seed` | `0` | Forwarded to the LLM and to the image/video models |
| `image_provider` | `none` | `none`, `fal`, `openrouter` — `none` means prompts only |
| `fal_image_model` / `openrouter_image_model` | first model found | Image model per provider |
| `video_provider` | `none` | `none`, `fal`, `openrouter` |
| `fal_video_model` / `openrouter_video_model` | first model found | Video model per provider |
| `render_concurrency` | `2` | Images/clips rendered at the same time |
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

**Cloud keys & rendering** — `openrouter_api_key`, `openai_api_key`, `anthropic_api_key`,
`fal_api_key` (all empty = read from the environment), `video_prompt_source` (`i2va` / `ref2va`),
`render_subject_sheets`, `live_preview`, `save_rendered_video`, `render_timeout`.

**Final film** — `concat_video`, `final_audio` (`music` / `clips` / `none`), `final_fit`
(`pad` / `stretch` / `crop`), `final_fps` (0 = the fastest rate among the clips), `final_crf`.

**Music & shots** — `analyze_music`, `snap_cuts_to_beats`, `audio_clip_padding`.

**Prompting** — `max_subjects`, `negative_prompt_base`, `include_dialogue`, `h3_style_directive`.

**Debug** — `save_json`, `filename_prefix`, `verbose`.

---

## Outputs

Fourteen of the seventeen outputs are **lists** (everything except `transcript` and `analysis_json`). Wire them into any node that iterates a list, or index them.

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
| 13 | `analysis_json` | — | Everything: BPM, beats, sections, treatment, art direction, subject bible, per-shot fields, and what was rendered |
| 14 | `images` | shots | **IMAGE** — rendered start frames (empty while `image_provider = none`) |
| 15 | `subject_images` | subjects | **IMAGE** — rendered reference sheets (`render_subject_sheets`) |
| 16 | `videos` | shots | **VIDEO** — rendered clips, also written to `ComfyUI/output/music2prompts/` |
| 17 | `final_video` | — | **VIDEO** — every clip cut together in shot order, with the music |

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

* Existing workflows keep working: the three new outputs were appended, so the old sockets did not move.
* One LLM request per stage plus one per shot batch — expect a few minutes per track on a
  small local model (≈5 min for 45 s of audio with `gemma-4-e4b` on a 2080 Ti).
* Whisper large-v3 needs ≈2.9 GB VRAM in fp16 for the weights and up to ~7 GB during word-timestamp
  alignment; `free_comfy_vram` and `free_lmstudio_vram` make room before it loads.
* fp16 only on GPU (bf16 is deliberately not used — Turing cards do not support it); CPU forces fp32.
* Shots always tile the whole track with no gaps, and every shot stays inside the
  `min_shot_seconds`…`max_shot_seconds` window (5–15 s by default, which is what MiniMax H3 accepts).
  `audio_clips` tile the track the same way, so concatenating them reproduces the input audio.
* Prompts are always written in English; lyrics, dialogue and on-screen text keep their original
  language.
