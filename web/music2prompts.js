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
 *
 * 5. On the concat node, hide the two mix gains unless the audio mode is actually "mix",
 *    and on the resolution node, the typed ratio unless the preset is "custom".
 *
 * 6. Put a saved workflow's widget values back where they belong when new widgets have
 *    been inserted above them since it was saved.
 *
 * 7. Shrink the node back onto its visible widgets. Nothing in the frontend does this -
 *    its own helper only ever grows - so a node with eighty-odd widgets, most of them
 *    hidden or advanced, keeps the height it had when they were all on show.
 */

import { app } from "../../scripts/app.js";

const NODE_ID = "Music2PromptsLM";
const CONCAT_ID = "Music2VideoConcat";
const RESOLUTION_ID = "Music2VideoResolution";
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

/** Shrink the node back onto the widgets it is actually showing.
 *
 * The frontend leaves a hidden widget out of `computeSize` (`isWidgetVisible` skips
 * anything with `hidden` set, and anything advanced while the advanced section is
 * collapsed) - but nothing shrinks the node afterwards. Its own helper only grows:
 * `expandToFitContent()` is `Math.max` on both axes, and it is what the advanced toggle
 * calls. So the height stays whatever it was when every widget was on show, and on a node
 * with eighty-odd of them that is most of the screen. The multiline `instruction` widget
 * then absorbs the difference, because a DOM widget is given the node's spare height - a
 * text box tall enough to push the rest of the node off the canvas.
 *
 * The width is only ever grown: it is the one dimension a user sets deliberately.
 */
function fit(node) {
  try {
    const size = node.computeSize?.();
    if (!size) return;
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
  } catch (error) {
    console.warn("[Music2Video] could not fit the node to its widgets:", error);
  }
}

/** Fit once the graph has finished applying the size saved in the workflow. */
function fitLater(node) {
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => fit(node));
  else fit(node);
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

  if (changed) {
    invalidate(node);
    fit(node);
  }
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

/** Put a saved workflow's values back on the right widgets after ones were inserted.
 *
 * `project_name` and `iteration` (and the control widget litegraph adds beside an int
 * with control_after_generate) went in above everything else, after workflows had
 * already been saved. Litegraph replays `widgets_values` positionally, so those files
 * would land every value three rows off - the exact failure that once fed a boolean into
 * `filename_prefix`. When the file names its values, use the names; otherwise recognise
 * the older, shorter array by its length and slide it down to where it belongs.
 */
const FIRST_LEGACY_WIDGET = "instruction";

function realign(node, info) {
  if (!info) return;
  const widgets = (node.widgets || []).filter((widget) => widget.serialize !== false);
  const named = info.widgets_values_named;
  if (named) {
    for (const widget of widgets) {
      if (widget.name in named) widget.value = named[widget.name];
    }
    return;
  }
  const values = info.widgets_values;
  if (!Array.isArray(values)) return;
  const shift = widgets.findIndex((widget) => widget.name === FIRST_LEGACY_WIDGET);
  if (shift <= 0 || values.length !== widgets.length - shift) return;  // already current
  values.forEach((value, index) => {
    widgets[shift + index].value = value;
  });
}

function migrate(node) {
  const widgets = widgetsOf(node);
  for (const name of Object.keys(RENAMED)) {
    const widget = widgets[name];
    const replacement = widget && RENAMED[name][String(widget.value)];
    if (replacement) widget.value = replacement;
  }
}

/** The concat node: the two mix gains mean nothing unless the mode is "mix". */
function refreshConcat(node) {
  const widgets = widgetsOf(node);
  const mixing = String(widgets.audio_mode?.value ?? "") === "mix";
  let changed = false;
  for (const name of ["music_gain", "clip_gain"]) {
    changed = setVisible(widgets[name], mixing) || changed;
  }
  if (changed) {
    invalidate(node);
    fit(node);
  }
  return changed;
}

/** The resolution node: the typed ratio means nothing unless the preset is "custom". */
function refreshResolution(node) {
  const widget = widgetsOf(node).custom_ratio;
  const custom = String(widgetsOf(node).aspect_ratio?.value ?? "") === "custom";
  if (setVisible(widget, custom)) {
    invalidate(node);
    fit(node);
  }
}

function watch(node, name, apply = refresh) {
  const widget = widgetsOf(node)[name];
  if (!widget) return;
  const previous = widget.callback;
  widget.callback = function (...args) {
    const result = previous?.apply(this, args);
    apply(node);
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
    if (node.comfyClass === CONCAT_ID) {
      try {
        watch(node, "audio_mode", refreshConcat);
        refreshConcat(node);
        const onConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
          const result = onConfigure?.apply(this, args);
          refreshConcat(this);
          fitLater(this);
          return result;
        };
      } catch (error) {
        console.warn("[Music2Video] concat node behaviour disabled:", error);
      }
      return;
    }
    if (node.comfyClass === RESOLUTION_ID) {
      try {
        watch(node, "aspect_ratio", refreshResolution);
        refreshResolution(node);
        const onConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
          const result = onConfigure?.apply(this, args);
          refreshResolution(this);
          fitLater(this);
          return result;
        };
      } catch (error) {
        console.warn("[Music2Video] resolution node behaviour disabled:", error);
      }
      return;
    }
    if (node.comfyClass !== NODE_ID) return;
    try {
      for (const name of ["llm_provider", "image_provider", "video_provider"]) watch(node, name);
      migrate(node);
      refresh(node);
      fetchLists().then((lists) => applyLists(node, lists));

      const onConfigure = node.onConfigure;
      node.onConfigure = function (...args) {
        const result = onConfigure?.apply(this, args);
        realign(this, args[0]);  // before migrate: it reads the provider values
        migrate(this);
        refresh(this);
        // the saved height was applied by configure() a moment ago, and it was saved
        // from a node that had never been shrunk either
        fitLater(this);
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
