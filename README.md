<p align="center">
  <img src="assets/banner.png" alt="Music2Video" width="900">
</p>

# Music2Video

Turns an audio track into **ready-to-use generation prompts** — and, if you want, into the images,
the clips, and the finished cut-together film. One node in; the prompts and timings come out on one pipe, the media on their own sockets.

The analysis is always local and free. The prompt writing is local by default and can be moved to a
cloud LLM. Rendering is **off by default** and only happens when you pick a provider for it.

| Job | Engine |
|---|---|
| Lyrics + word-level timestamps | **Whisper large-v3** (transformers, GPU by default, CPU selectable) |
| BPM, beat grid, sections, energy | **librosa** when installed, otherwise a built-in numpy/scipy fallback |
| Treatment, art direction, shot writing | **LM Studio** (default, `google/gemma-4-e4b`) or **OpenRouter** / **OpenAI** / **Anthropic** |
| MiniMax H3 prompt formatting | deterministic Python renderers — the model fills fields, the code builds the exact skeleton |
| Optional image rendering | **fal.ai** or **OpenRouter** (`image_provider`, default `pipe-steps`) |
| Optional video rendering | **fal.ai** or **OpenRouter** (`video_provider`, default `pipe-steps`) |
| Assembling the final film | **PyAV** — clips trimmed to their shots, music muxed in, no ffmpeg binary needed |

Built on the ComfyUI **V3 node schema**, so the 30+ secondary settings collapse into the native
*Advanced* section of the new node layout.

---

## Install

**ComfyUI-Manager** — search for **Music2Video**, hit Install, restart.

**comfy-cli**

```bash
comfy node install music2video
```

**By hand**

```bash
git clone https://github.com/lazniak/comfyui-music2video ComfyUI/custom_nodes/music2video
```

Registry page: [registry.comfy.org/nodes/music2video](https://registry.comfy.org/nodes/music2video).
Needs **ComfyUI ≥ 0.3.48** — the first release carrying the V3 node schema
(`comfy_api.latest`) the pack is built on. Nothing else beyond a normal ComfyUI install.

---

## Quick start (PL)

1. Odpal LM Studio → *Developer* → **Start Server** (domyślnie `http://127.0.0.1:1234`).
2. Skopiuj / zlinkuj ten katalog do `ComfyUI/custom_nodes/` i zrestartuj ComfyUI.
3. Dodaj node **🎵 Music2Video**, podłącz `LoadAudio`, wpisz brief.
4. Pierwsze uruchomienie ściąga Whisper large-v3 (~3 GB) do `ComfyUI/models/whisper/`.
5. Wszystkie teksty i liczby wychodzą jednym kablem `pipe`. Dodaj node
   **🎵 Music2Video Pipe Expand**, podłącz do niego `pipe`, i tam masz osobne wyjścia:
   `image_prompts_start` do enkodera tekstu, `video_prompts_i2va` / `video_prompts_ref2va`
   do node'a MiniMax H3. Media (`images`, `videos`, `final_video`, `audio_clips`) zostają
   jako własne gniazda na głównym node'zie.

Instalacja przez junction (Windows, bez kopiowania):

```bash
cmd /c mklink /J "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\music2video" "D:\code\comfyui-music2video"
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
> bills you. Image and video rendering therefore stays `pipe-steps` until you choose otherwise, and one
> run renders one image per shot (plus one per subject with `render_subject_sheets`) and one clip per
> shot. A 3-minute track cut into 6-second shots is 30 shots — check the provider's price per image
> and per second of video before you start it.

### Rendering

* `image_provider` + `fal_image_model` / `openrouter_image_model` render the start frames. The result
  arrives on the `images` output, aligned with the shots.

#### Keeping one face and one look

Most fal image models are **text-to-image**: their API declares no field for a reference image at all,
so handing them one does nothing. That is why independently rendered shots come back as different
people in a different style - the model was never told what to keep. Three things address it:

* **Subject sheets first.** With `render_subject_sheets` on, the sheets are rendered before the shots
  and then travel *with* every shot as references.
* **`fal_image_edit_model`.** Once there are references, the shots switch to this model. It must be an
  edit endpoint (`fal-ai/nano-banana-pro/edit`, `fal-ai/flux-2-pro/edit`, `fal-ai/qwen-image-edit-2511`,
  `bytedance/seedream/v5/pro/edit`, …) - the model list now includes fal's `image-to-image` category,
  which is where they all live. Leave it and the node warns instead of silently dropping the identity.
* **`style_anchor`** (on by default) renders shot 1 alone, then hands it to every later shot as another
  reference. This is what holds the grade, the grain and the wardrobe together. It costs one
  serialised render.

OpenRouter needs none of this: its image API takes `input_references` on almost every model it lists,
so the same references go out on the model you already picked.

#### Lip sync

`lipsync_audio` (on by default) sends each shot's own slice of the track to the video model - but only
where the endpoint actually declares an input for a driving audio track, and it says so in the log when
it does not:

| endpoint | field | what the audio does |
| --- | --- | --- |
| `fal-ai/wan/v2.7/image-to-video` | `audio_url` | driving audio - the performance follows it |
| `minimax/h3/reference-to-video` | `reference_audio_urls` | a multimodal reference, named `<Audio 1>` in the prompt |
| talking-head models (`fal-ai/infinitalk`, `fal-ai/kling-video/ai-avatar/*`, …) | `audio_url` | drives the face |
| `minimax/h3/image-to-video`, kling i2v, veo, hailuo | *none* | nothing to send it to |
| `alibaba/wan-3.0-prime/image-to-video` | `audio` is a **boolean** | "include generated audio" - not an input |

The clip is encoded as MP3 (~95 KB for six seconds against 1 MB as WAV) and travels inline. On the
ref2va path the prompt is rewritten to name it: `<Audio 1> is this shot's own slice of the original
track…` plus a `fully_copy` retention line, because a reference nothing in the prompt refers to is just
an attached file. The same rewrite binds each subject to the image that defines it -
`<Subject 1> … whose appearance comes from <Picture 1>` - which is what the MiniMax H3 format requires
and what was missing before.

OpenRouter's video API has no audio input of any kind; the node says so rather than pretending.

#### Who writes the final prompt

Both MiniMax H3 endpoints rewrite the prompt before generating (`prompt_expansion_mode` defaults to
`balanced`, and the wan endpoints do the same behind `enable_prompt_expansion`). That rewrite happens
per request, so every shot's look is re-invented independently by fal's own writer - working directly
against one art direction. `prompt_expansion` defaults to `minimal`, which keeps the prompt this node
assembled; `rich` hands the endpoint's writer the wheel, and `model default` sends nothing.
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
  it declares. Picking an image-to-video model with `image_provider = pipe-steps` is refused before the run
  starts, rather than after the LLM and the images have been billed.
* Clips are written to the run's project folder (see **Where the files go**) and returned on the `videos` output (the
  regular VIDEO type, so `SaveVideo` / `PreviewVideo` accept them).
* `concat_video` (on by default) glues them into **one finished film** on the `final_video` output.
  Every clip is placed on a single grid — one size, one frame rate — and trimmed or frozen on its
  last frame so it lasts exactly as long as its shot; the film therefore stays in sync with the
  track. Clips whose aspect differs are letterboxed (`final_fit`), and `final_audio` decides the
  soundtrack: `music` (your track), `clips` (the audio the video model generated), `mix` (both
  summed) or `none`.
* When a single shot fails to render, its slot in `images` stays filled with a black frame so the
  list still lines up with the shots. When *every* shot fails, the node raises with the provider's
  own error instead of returning empty lists.
* **Wiring a media output on a run that renders nothing is safe.** ComfyUI slices every input once
  per downstream execution (`v[i if len(v) > i else -1]`), so an empty list reaches `v[-1]` and the
  whole prompt dies with a bare `IndexError: list index out of range` — a traceback that names
  neither the node nor the wire. Instead of an empty list, each media socket hands out a silent
  execution blocker: the node wired to it is skipped, the log says which socket went empty and why,
  and the rest of the graph — prompts, timings, audio — finishes normally. The same applies to every
  list output of **Pipe Expand**: `reference_subjects` on an instrumental with no subjects blocks
  its consumer rather than crashing the run.

---

## Where the files go

Everything one run writes — frames, subject sheets, clips, the final film, the transcript,
the analysis JSON and the cost report — lands in one folder:

```
ComfyUI/output/music2prompts/<project_name>_v<iteration>/
```

`project_name` is one folder name, not a path: separators and the characters Windows
refuses become `-`, so nothing typed there can write outside the output folder; letters
with diacritics are kept. An empty name falls back to `music2video`.

`iteration` ships with ComfyUI's **increment** control, so it counts itself up after every
run and each take gets its own folder — `myclip_v001`, `myclip_v002`, … Set the control to
`fixed` to keep re-running into one folder; filenames still carry the run's timestamp, so
even then nothing is overwritten.

---

## Inputs

### Main

| Widget | Default | Meaning |
|---|---|---|
| `audio` | — | AUDIO from `LoadAudio` (or anything producing AUDIO) |
| `project_name` | `music2video` | Folder this run writes into (see **Where the files go**) |
| `iteration` | `1` | Take number, appended as `_v001`. Set to **increment**, so every run gets its own folder |
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
| `image_provider` | `pipe-steps` | `pipe-steps`, `fal`, `openrouter` — `pipe-steps` renders nothing: the shots leave as prompts on the pipe |
| `fal_image_model` / `openrouter_image_model` | first model found | Image model per provider |
| `video_provider` | `pipe-steps` | `pipe-steps`, `fal`, `openrouter` |
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
>
> A window is handed to Whisper whole, which is the model's own long-form path.
> `whisper_chunk_length_s` only comes into play when a slice is genuinely longer than a chunk —
> setting `whisper_window_seconds` to 0 hands the whole track over at once, and then transformers'
> chunking is what keeps it inside the card. That chunking is approximate at the seams (word timings
> can drift), and the node says so once when it happens.

### Handing the card back

A graph that renders after this node needs the whole card, and the error it gets when it
cannot have it — `Allocation on device 0 would exceed allowed memory` — names neither this
node nor the model still sitting there. So, in order, a run:

1. unloads ComfyUI's own models (`free_comfy_vram`) and the LM Studio model
   (`free_lmstudio_vram`) **before** Whisper loads;
2. drops the Whisper pipeline as soon as the transcript is out, unless
   `whisper_keep_loaded` is on — it is **off** by default, because large-v3 holds ~2.9 GB;
3. unloads the LM Studio model once the prompts are written (`lm_unload_after`, **on** by
   default) and waits until LM Studio reports it gone, so the memory is really back;
4. empties the caching allocator on the way out — freeing a model is not enough, the
   blocks stay reserved and ComfyUI cannot use what it cannot see.

**Cancel is part of this.** Pressing Cancel used to leave both models resident, so the run
after it hit the same out-of-memory error — see *Stopping a run* below.

Two things this node cannot do for you:

* **Pick a model that fits.** A 27B LLM on an 11 GB card is mostly running on the CPU: the
  writing stages take ten minutes instead of one, and LM Studio may hold VRAM the whole
  time. An 8–14B local model, or `llm_provider = openrouter` for a few cents a run, is the
  single biggest speed-up available here.
* **Control LM Studio's own idle behaviour.** Set a JIT TTL there if you want it to let go
  without being asked.

**Cloud keys & rendering** — `openrouter_api_key`, `openai_api_key`, `anthropic_api_key`,
`fal_api_key` (all empty = read from the environment), `video_prompt_source` (`i2va` / `ref2va`),
`render_subject_sheets`, `style_anchor`, `lipsync_audio`, `prompt_expansion`, `live_preview`,
`save_rendered_images`, `save_rendered_video`, `save_transcript`, `render_timeout`.

**Final film** — `concat_video`, `final_audio` (`music` / `clips` / `mix` / `none`), `final_fit`
(`pad` / `stretch` / `crop`), `final_fps` (0 = the fastest rate among the clips), `final_crf`.

**Music & shots** — `analyze_music`, `snap_cuts_to_beats`, `audio_clip_padding`.

**Prompting** — `max_subjects`, `negative_prompt_base`, `include_dialogue`, `h3_style_directive`.

**Debug** — `save_json`, `filename_prefix`, `verbose`.

### Stopping a run

Cancel takes effect now, not at the end of whatever is running:

* **Whisper** is given a stopping rule, so a cancel ends the decode inside the current
  window instead of after it — and the windows after it never start.
* **The LLM** is called from a worker thread. An HTTP request already in flight cannot be
  aborted, so the reply is left to arrive unread; the model is then unloaded, which ends
  the generation at the far end too.
* **Renders** stop between polls (a quarter second), and the shots still queued behind the
  cancel are never submitted — nothing is billed for a clip nobody wants any more.
* **The film** stops between clips.

Then the run hands the card back on its way out — the same unloads a finished run does,
plus the caching allocator — so the next node in the graph gets the memory. A run that
fails does the same.

One consequence worth knowing: the flag ComfyUI sets is deliberately *not* cleared when
this node notices it. The node renders and polls on several threads at once, and clearing
it — which ComfyUI's own helper does — would mean the first thread to see the cancel hides
it from the rest. ComfyUI resets the flag itself at the start of the next prompt.

---

## Outputs

The node has six sockets. Every text and number leaves on one of them.

| # | Output | Type | Contents |
|---|---|---|---|
| 1 | `pipe` | **M2P_PIPE** | Every prompt, name, timing, the transcript and the analysis JSON — see the table below |
| 2 | `audio_clips` | **AUDIO**, per shot | Cut sample-accurately to each shot — feed straight into lipsync. Also carried inside the pipe |
| 3 | `images` | **IMAGE**, per shot | Rendered start frames (empty while `image_provider = pipe-steps`) |
| 4 | `subject_images` | **IMAGE**, per subject | Rendered reference sheets (`render_subject_sheets`) |
| 5 | `videos` | **VIDEO**, per shot | Rendered clips, also written to the run's project folder |
| 6 | `final_video` | **VIDEO** | Every clip cut together in shot order, with the music |

### The pipe, and taking it apart

Twelve string and number sockets on one node was a wall of noodles, and they almost always
travel together. They now leave as one `pipe`; **🎵 Music2Video Pipe Expand** hands them
back wherever one is actually needed. That node also passes the pipe straight through, so
several can sit along one wire — expand next to the sampler for the prompts, again next to
a text preview for the transcript. It calls nothing and costs nothing.

The media stayed on their own sockets on purpose: an IMAGE or a VIDEO normally goes
straight into a preview or a save node, so hiding it behind an expander would cost a node
and buy nothing.

Everything in the pipe except `transcript` and `analysis_json` is a **list**.

| Field | Aligned with | Contents |
|---|---|---|
| `image_prompts_start` | shots | Natural-language cinematic prompt for the first frame of each shot (Flux / Qwen-Image / Z-Image style) |
| `image_prompts_reference` | subjects | Clean reference-sheet prompt per recurring subject |
| `reference_subjects` | subjects | Subject names, same order as `image_prompts_reference` |
| `video_prompts_i2va` | shots | MiniMax H3 **image-to-video** prompt (first frame = the image from `image_prompts_start`) |
| `video_prompts_ref2va` | shots | MiniMax H3 **reference-to-video** prompt (six-section format) |
| `negative_prompts` | shots | Base negatives + per-shot additions, de-duplicated |
| `shot_index` | shots | 1…N |
| `start_times` | shots | Seconds |
| `end_times` | shots | Seconds |
| `durations` | shots | Seconds, inside the `min_shot_seconds`…`max_shot_seconds` window |
| `transcript` | — | Full transcription (empty for instrumentals) |
| `audio_clips` | shots | The same per-shot **AUDIO** as socket 2, so a lipsync graph can take it off the pipe |
| `clip_prefixes` | shots | The `filename_prefix` for each shot's clip — `music2prompts/<project>_v003/<prefix>_<stamp>_shot001` — so a clip rendered elsewhere in the graph saves into this run's folder, in shot order |
| `final_video_name` | — | The `filename_prefix` for the finished film. Leave the Concat node's own `filename_prefix` empty and it takes this off the pipe |
| `analysis_json` | — | Everything: BPM, beats, sections, treatment, art direction, subject bible, per-shot fields, what was rendered, and what it cost |

> **Upgrading:** a workflow saved before the pipe existed loses the links to those twelve
> sockets, and the links to the media sockets move up with them. Re-wire the media
> outputs, then take the rest off a Pipe Expand node.

### Cutting clips you rendered yourself

The main node only assembles what **it** rendered. If the clips come from somewhere else in
the graph — an LTX or Wan subgraph, a sampler, a load-from-disk node — the film is put
together by **🎵 Music2Video Concat**:

| Input | |
|---|---|
| `videos` | The clips, as a list. It takes the whole list at once rather than running per clip |
| `pipe` | *optional* — read for the shot durations only, so each clip is trimmed or held to last exactly as long as its shot |
| `audio` | *optional* — the source track, for the two modes that use it |
| `audio_mode` | `source audio` (the track alone), `mix` (track + clip audio, balanced by `music_gain` / `clip_gain`), `video audio` (only what the models generated), `silent` |
| `fit`, `fps`, `width`, `height`, `crf` | The grid every clip is placed on. `0` takes it from the clips |

Out come the finished `video`, the `path` it was written to, and its `duration`.

**Wire the pipe.** Without it the clips keep whatever length they came back at, and the film
drifts out of sync with the track as the errors accumulate; with it every cut lands within
half a frame of where the beat grid put it. If a render failed the clip list is shorter than
the shot list, the two no longer line up, and the node says so and falls back to the clips'
own lengths rather than misaligning the whole film.

**Leave `filename_prefix` empty** and the pipe names the film: it lands in that run's project
folder, under that run's name, next to the clips it was cut from and the transcript. Type
something there and yours wins, subfolders included (`films/tour`, the same convention every
ComfyUI save node uses). The pipe also carries `clip_prefixes` — one per shot — so the SaveVideo
node inside your render subgraph can write each clip into the same folder with the same run
stamp, in shot order.

Nothing is generated and nothing is billed — one muxing pass with PyAV, no ffmpeg binary.

### Sizing the frames

**🎵 Music2Video Resolution** is ComfyUI's own *Resolution Selector* — same arithmetic, same
preset list, verified against it across every preset, megapixel and multiple — with two things
added:

* **`custom`** in the ratio dropdown, and a `custom_ratio` box to type one no preset covers:
  `21:10`, `5:4`, `2.39` (cinemascope, same as `2.39:1`), or `1920x1080`, which is read as the
  ratio it reduces to. Only the shape is taken from it — `megapixels` decides the size.
* a third output, **`aspect_ratio`**, carrying that ratio in plain `W:H` form. Wire it into the
  main node's `aspect_ratio` input and the shape of the latent and the shape the prompts are
  written for cannot drift apart. A ratio arriving down a wire is not held to that widget's six
  presets, so a custom one reaches the prompts and the render payloads as typed.

### Wiring examples

All of these start with `pipe` → **Music2Video Pipe Expand**; the field names below are
that node's outputs.

* **Start frames** → `image_prompts_start` into a text encoder, `negative_prompts` into the negative
  encoder. `durations` / `start_times` drive whatever timing you need downstream.
* **MiniMax H3, image-to-video** → generate the image from `image_prompts_start[i]`, then feed
  `video_prompts_i2va[i]` into `MiniMaxH3Easy.prompt` with `mode = image`, `keyframe_role = first`,
  `seconds = durations[i]`.
* **MiniMax H3, reference-to-video** → generate the subject references from
  `image_prompts_reference`, feed them as media, and use `video_prompts_ref2va[i]` with
  `mode = reference`.
* **Rendering the clips outside this pack** → `video_prompts_i2va[i]` (or `ref2va`) into whatever
  video node you use, collect its VIDEO outputs into a list, then `videos` + `pipe` + `audio` into
  **Music2Video Concat**. That is the same assembly the main node does for its own renders.
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
