/**
 * Music2Video - node-side behaviour for the provider widgets.
 *
 * Four jobs, all deliberately small:
 *
 * 1. Show only the widgets that belong to the selected providers. Every provider has
 *    its own model dropdown (ComfyUI builds a combo's options once, on the server), so
 *    hiding the others is what makes the node readable. Hiding sets BOTH `widget.hidden`
 *    (read by the canvas renderer) and `widget.options.hidden` (the only flag the Vue
 *    renderer reads). It never swaps `widget.type`: an unknown type falls back to the
 *    legacy widget component, which draws nothing but still reserves a row - that is
 *    exactly the empty gap this used to leave behind.
 *
 * 2. Refresh the model lists from the pack's own route, so a model added in LM Studio
 *    or a key exported after ComfyUI started shows up without a restart. The values of
 *    hidden widgets are still serialized and still sent to the backend, so nothing here
 *    changes what python receives.
 *
 * 3. Give the pipe wire a colour of its own, so the one line carrying every prompt and
 *    timing is not another grey noodle among the media links.
 *
 * 4. Rewrite renamed widget values on load, so a workflow saved before a rename still
 *    passes the server's combo validation.
 */

import { app } from "../../scripts/app.js";

const NODE_ID = "Music2PromptsLM";
const ROUTE = "/music2prompts/models";
const PIPE_TYPE = "M2P_PIPE";

// widget -> the provider value that keeps it on screen
const RULES = [
  ["llm_provider", { lm_model: "lmstudio", openrouter_model: "openrouter", openai_model: "openai", anthropic_model: "anthropic" }],
  ["llm_provider", { lm_url: "lmstudio", lm_api_key: "lmstudio", lm_model_override: "lmstudio" }],
  ["llm_provider", { lm_auto_download: "lmstudio", lm_auto_load: "lmstudio", lm_context_length: "lmstudio" }],
  ["llm_provider", { lm_unload_after: "lmstudio", free_lmstudio_vram: "lmstudio" }],
  ["llm_provider", { openrouter_api_key: "openrouter", openai_api_key: "openai", anthropic_api_key: "anthropic" }],
  ["image_provider", { fal_image_model: "fal", fal_image_edit_model: "fal", openrouter_image_model: "openrouter" }],
  ["video_provider", { fal_video_model: "fal", openrouter_video_model: "openrouter" }],
];

// only meaningful once something is actually rendered
const RENDER_ONLY = [
  "render_concurrency", "video_prompt_source", "render_subject_sheets", "save_rendered_video", "render_timeout",
  "live_preview", "save_rendered_images", "style_anchor",
];
// only meaningful once clips are rendered
const VIDEO_ONLY = ["concat_video", "final_audio", "final_fit", "final_fps", "final_crf", "lipsync_audio", "prompt_expansion"];

// which cached list feeds which dropdown
const LISTS = {
  lm_model: "lmstudio",
  openrouter_model: "openrouter_llm",
  openai_model: "openai_llm",
  anthropic_model: "anthropic_llm",
  fal_image_model: "fal_image",
  fal_image_edit_model: "fal_image",
  openrouter_image_model: "openrouter_image",
  fal_video_model: "fal_video",
  openrouter_video_model: "openrouter_video",
};

let cache = null;
let inFlight = null;

async function fetchLists(force = false) {
  if (!force && cache) return cache;
  if (!force && inFlight) return inFlight;
  const url = force ? `${ROUTE}?force=1` : ROUTE;
  inFlight = fetch(url, { cache: "no-store" })
    .then((response) => (response.ok ? response.json() : null))
    .then((data) => {
      if (data && !data.error) cache = data;
      inFlight = null;
      return cache;
    })
    .catch((error) => {
      console.warn("[Music2Video] could not refresh model lists:", error);
      inFlight = null;
      return cache;
    });
  return inFlight;
}

function widgetsOf(node) {
  const found = {};
  for (const widget of node.widgets || []) found[widget.name] = widget;
  return found;
}

/** Vue nodes install an accessor on `widgets`; reassigning through it re-snapshots. */
function invalidate(node) {
  const descriptor = Object.getOwnPropertyDescriptor(node, "widgets");
  if (descriptor && typeof descriptor.set === "function") {
    node.widgets = [...(node.widgets || [])];
  }
  node.setDirtyCanvas?.(true, true);
}

function setVisible(widget, visible) {
  if (!widget) return false;
  let changed = false;
  // undo the old type-swap trick, which left an empty row behind
  if (widget.origType !== undefined) {
    widget.type = widget.origType;
    delete widget.origType;
    changed = true;
  }
  if (!widget.options) widget.options = {};
  const hidden = !visible;
  if (widget.hidden !== hidden || widget.options.hidden !== hidden) changed = true;
  widget.hidden = hidden;
  widget.options.hidden = hidden;
  return changed;
}

/** Does this provider value actually render anything? "none" is the pre-rename spelling. */
function renders(value) {
  return !["pipe-steps", "none", ""].includes(String(value ?? ""));
}

function refresh(node) {
  const widgets = widgetsOf(node);
  let changed = false;
  for (const [providerName, mapping] of RULES) {
    const provider = widgets[providerName];
    if (!provider) continue;
    for (const [name, wanted] of Object.entries(mapping)) {
      changed = setVisible(widgets[name], String(provider.value) === wanted) || changed;
    }
  }

  const images = widgets.image_provider && renders(widgets.image_provider.value);
  const videos = widgets.video_provider && renders(widgets.video_provider.value);
  for (const name of RENDER_ONLY) changed = setVisible(widgets[name], !!(images || videos)) || changed;
  for (const name of VIDEO_ONLY) changed = setVisible(widgets[name], !!videos) || changed;

  const usesFal = widgets.image_provider?.value === "fal" || widgets.video_provider?.value === "fal";
  changed = setVisible(widgets.fal_api_key, !!usesFal) || changed;
  if (widgets.image_provider?.value === "openrouter" || widgets.video_provider?.value === "openrouter") {
    changed = setVisible(widgets.openrouter_api_key, true) || changed;
  }

  if (changed) invalidate(node);
  return changed;
}

/** Put the freshly fetched ids into the dropdowns, keeping the current pick valid. */
function applyLists(node, lists) {
  if (!lists) return;
  for (const widget of node.widgets || []) {
    const kind = LISTS[widget.name];
    const values = kind && lists[kind];
    if (!Array.isArray(values) || values.length === 0) continue;
    const next = [...values];
    if (widget.value && !next.includes(widget.value)) next.unshift(widget.value);
    if (!widget.options) widget.options = {};
    widget.options.values = next;
    if (!widget.value || String(widget.value).startsWith("(")) widget.value = next[0];
  }
  invalidate(node);
}

/** Rewrite values that were renamed after a workflow could already have been saved.
 *
 * `image_provider` / `video_provider` used to call the off position "none". The server
 * validates a combo against its current option list, so a graph saved back then would
 * fail with "Value not in list" before any of our code runs. Rewriting on load costs
 * nothing and keeps yesterday's workflow working.
 */
const RENAMED = {
  image_provider: { none: "pipe-steps" },
  video_provider: { none: "pipe-steps" },
};

function migrate(node) {
  const widgets = widgetsOf(node);
  for (const name of Object.keys(RENAMED)) {
    const widget = widgets[name];
    const replacement = widget && RENAMED[name][String(widget.value)];
    if (replacement) widget.value = replacement;
  }
}

function watch(node, name) {
  const widget = widgetsOf(node)[name];
  if (!widget) return;
  const previous = widget.callback;
  widget.callback = function (...args) {
    const result = previous?.apply(this, args);
    refresh(node);
    return result;
  };
}

app.registerExtension({
  name: "music2prompts.providerWidgets",

  /** Colour the pipe link, and only that one.
   *
   * `LGraphCanvas.link_type_colors` is the frontend's own map from socket type to wire
   * colour; writing one key into it leaves every other link alone. Guarded because it is
   * not part of any published contract - a missing map costs the colour, never the node.
   */
  setup() {
    try {
      const canvas = app.canvas?.constructor;
      const colours = canvas?.link_type_colors ?? window.LGraphCanvas?.link_type_colors;
      if (colours && !colours[PIPE_TYPE]) colours[PIPE_TYPE] = "#d4a05a";
    } catch (error) {
      console.warn("[Music2Video] could not colour the pipe link:", error);
    }
  },

  async nodeCreated(node) {
    if (node.comfyClass !== NODE_ID) return;
    try {
      for (const name of ["llm_provider", "image_provider", "video_provider"]) watch(node, name);
      migrate(node);
      refresh(node);
      fetchLists().then((lists) => applyLists(node, lists));

      const onConfigure = node.onConfigure;
      node.onConfigure = function (...args) {
        const result = onConfigure?.apply(this, args);
        migrate(this);
        refresh(this);
        return result;
      };

      const getExtraMenuOptions = node.getExtraMenuOptions;
      node.getExtraMenuOptions = function (canvas, options) {
        const result = getExtraMenuOptions?.apply(this, arguments);
        options?.push({
          content: "Refresh model lists",
          callback: () => {
            fetchLists(true).then((lists) => {
              applyLists(this, lists);
              refresh(this);
            });
          },
        });
        return result;
      };
    } catch (error) {
      console.warn("[Music2Video] node behaviour disabled:", error);
    }
  },
});
