# Changelog

The section matching the version in `pyproject.toml` is published to the Comfy Registry
as that version's release notes, so keep the heading format `## <version>`.

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
