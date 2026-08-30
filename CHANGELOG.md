# Changelog

The section matching the version in `pyproject.toml` is published to the Comfy Registry
as that version's release notes, so keep the heading format `## <version>`.

## 1.2.0

- **Cancel stops the run now**, instead of at the end of whatever was running. Whisper is
  given a stopping rule so a cancel ends the decode inside the current window; the LLM
  call runs on a worker thread, so an HTTP request already in flight no longer holds the
  node (the model is unloaded straight after, which ends the generation at the far end);
  renders stop between polls and the shots queued behind the cancel are never submitted,
  so nothing is billed for them; the film stops between clips. A cancelled - or failed -
  run then hands the card back the way a finished one does: LM Studio unloaded, Whisper
  dropped, the caching allocator emptied. Cancelling used to leave both models resident
  and the next node in the graph ran out of memory.
- Fixed: every transcription printed *"Using `chunk_length_s` is very experimental with seq2seq
  models"*. A window shorter than one chunk was being chunked into exactly one chunk - a no-op
  that cost Whisper its own long-form path, the one it was trained for. Windows are now handed
  to the model whole; `whisper_chunk_length_s` is only used when a slice really is longer than a
  chunk (`whisper_window_seconds = 0`), and the node then says once, in its own words, that word
  timings can drift around the seams.
- New node: **🎵 Music2Video Concat**. Takes a list of VIDEO from anywhere in the graph -
  an LTX or Wan subgraph, a sampler, a load-from-disk node - and cuts it into one film,
  which the main node could only do for clips it had rendered itself. Optional `pipe`
  input supplies the shot durations, so every cut lands within half a frame of the beat
  grid instead of drifting; optional `audio` input carries the source track.
- Its `audio_mode` decides the soundtrack: `source audio` (the track alone), `mix` (the
  track and the clips' own audio summed, balanced by `music_gain` and `clip_gain` and
  clipped so the sum cannot wrap), `video audio` (only what the models generated) or
  `silent`.
- `mix` is a new mode in the muxer generally, so the main node's `final_audio` offers it
  too.
- Two new inputs at the top of the main node: **`project_name`** and **`iteration`**.
  Everything a run writes now lands in
  `ComfyUI/output/music2prompts/<project_name>_v<iteration>/` instead of one shared
  folder. `iteration` carries ComfyUI's increment control, so every run gets its own
  take folder. The name is sanitised down to a single path component, so nothing typed
  there can write outside the output folder.
- Fixed: the main node's `final_audio` = `mix` was not given the track, so it silently
  fell back to the clips' own audio.
- **Memory is handed back at the end of a run**, so the sampler or video model that runs
  next in the graph can have the card. `lm_unload_after` now defaults to **on** and waits
  until LM Studio reports the model gone; `whisper_keep_loaded` defaults to **off**; and
  the caching allocator is emptied on the way out, because a freed model whose blocks are
  still reserved is memory ComfyUI cannot see.
- Fixed: the LM Studio load request sent `context_length` nested under `config`, which
  that API rejects with HTTP 400. The run then fell back to loading the model at whatever
  context LM Studio defaults to, silently, and long prompts were truncated for the rest of
  the run. It is now sent where the API looks for it, and a context smaller than the one
  asked for is reported.

## 1.1.0

- The per-shot audio clips now travel on the pipe as well, as a list, and **Music2Video
  Pipe Expand** hands them back as AUDIO. They stay on their own socket too: a lipsync
  node is often nowhere near an expander.
- `image_provider` and `video_provider` no longer call the off position "none". It is
  **`pipe-steps`**, which says what happens rather than what does not: the run still
  produces every prompt, timing and audio slice on the pipe, it just renders nothing and
  bills nothing. A workflow saved before this keeps working - the value is rewritten on
  load, and the old spelling is still accepted everywhere the node reads it.
- **A run that renders nothing no longer takes the graph down.** ComfyUI slices every
  input once per downstream execution (`v[i if len(v) > i else -1]`), so an empty list
  wired into any node reached `v[-1]` and killed the whole prompt with a bare
  `IndexError: list index out of range` - after all the LLM work was already paid for.
  Each media socket now hands out a silent execution blocker instead: the node wired to it
  is skipped, the log says which socket went empty and why, and the prompts, timings and
  audio still come out. Every list output of Pipe Expand does the same.

## 1.0.0

First public release.

- One node from a track to a finished film. Whisper large-v3 transcribes the lyrics with
  word-level timing; librosa (or a built-in numpy/scipy fallback) reads the BPM, the beat
  grid and the sections; an LLM - LM Studio locally by default, or OpenRouter / OpenAI /
  Anthropic - writes the treatment, the art direction, a bible of the recurring subjects
  and every shot, with the cuts snapped to the beat.
- Prompts out: start-frame image prompts, subject reference-sheet prompts, MiniMax H3
  image-to-video and reference-to-video prompts in their exact six-section format,
  negatives, per-shot timings, and sample-accurate AUDIO slices for lipsync - all on one
  pipe, taken apart by "Music2Video Pipe Expand" wherever a value is needed.
- Optional rendering through fal.ai or OpenRouter, then every clip trimmed to its shot and
  cut together under the original music with PyAV. No ffmpeg binary needed.
- Every frame and clip appears in the node as it lands, and the cost meter under the
  gallery reports in USD what each model actually billed - figures billed by OpenRouter,
  computed from fal's own price catalogue.
- The analysis is always local and free. Nothing is rendered until you pick a provider.
- Requires ComfyUI 0.3.48 or newer (the V3 node schema).
