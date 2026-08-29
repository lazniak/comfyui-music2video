/**
 * Music2Prompts - show only the widgets that belong to the selected providers.
 *
 * ComfyUI builds a node's dropdown options once, on the server, so every provider
 * gets its own model widget. This extension keeps exactly one of them on screen:
 * pick `llm_provider = openrouter` and only `openrouter_model` remains, and the
 * same for the image and video providers. Nothing else is touched - no custom UI,
 * no extra panels; if the frontend is too old to hide widgets, all of them simply
 * stay visible and the node still works.
 */

import { app } from "../../scripts/app.js";

const NODE_ID = "Music2PromptsLM";

// widget -> the provider value that keeps it visible
const RULES = [
  ["llm_provider", { lm_model: "lmstudio", openrouter_model: "openrouter", openai_model: "openai", anthropic_model: "anthropic" }],
  ["llm_provider", { lm_url: "lmstudio", lm_api_key: "lmstudio", lm_model_override: "lmstudio" }],
  ["llm_provider", { lm_auto_download: "lmstudio", lm_auto_load: "lmstudio", lm_context_length: "lmstudio" }],
  ["llm_provider", { lm_unload_after: "lmstudio", free_lmstudio_vram: "lmstudio" }],
  ["llm_provider", { openrouter_api_key: "openrouter", openai_api_key: "openai", anthropic_api_key: "anthropic" }],
  ["image_provider", { fal_image_model: "fal", openrouter_image_model: "openrouter" }],
  ["video_provider", { fal_video_model: "fal", openrouter_video_model: "openrouter" }],
];

// widgets that only matter once something is actually rendered
const RENDER_ONLY = ["video_prompt_source", "render_subject_sheets", "save_rendered_video", "render_timeout", "render_concurrency"];

function widgetsOf(node) {
  const found = {};
  for (const widget of node.widgets || []) found[widget.name] = widget;
  return found;
}

function setVisible(widget, visible) {
  if (!widget || widget.hidden === !visible) return;
  widget.hidden = !visible;
  // older frontends collapse hidden widgets by type instead of the flag
  if (visible) {
    if (widget.origType !== undefined) widget.type = widget.origType;
  } else if (widget.origType === undefined) {
    widget.origType = widget.type;
    widget.type = "hidden";
  }
}

function refresh(node) {
  const widgets = widgetsOf(node);
  for (const [providerName, mapping] of RULES) {
    const provider = widgets[providerName];
    if (!provider) continue;
    for (const [name, wanted] of Object.entries(mapping)) {
      setVisible(widgets[name], String(provider.value) === wanted);
    }
  }
  const renders =
    (widgets.image_provider && widgets.image_provider.value !== "none") ||
    (widgets.video_provider && widgets.video_provider.value !== "none");
  for (const name of RENDER_ONLY) setVisible(widgets[name], !!renders);

  // fal shares one key between images and video
  const usesFal =
    (widgets.image_provider && widgets.image_provider.value === "fal") ||
    (widgets.video_provider && widgets.video_provider.value === "fal");
  setVisible(widgets.fal_api_key, !!usesFal);
  const usesOpenRouterMedia =
    (widgets.image_provider && widgets.image_provider.value === "openrouter") ||
    (widgets.video_provider && widgets.video_provider.value === "openrouter");
  if (usesOpenRouterMedia) setVisible(widgets.openrouter_api_key, true);

  node.setDirtyCanvas?.(true, true);
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
  async nodeCreated(node) {
    if (node.comfyClass !== NODE_ID) return;
    try {
      for (const name of ["llm_provider", "image_provider", "video_provider"]) watch(node, name);
      refresh(node);
      const onConfigure = node.onConfigure;
      node.onConfigure = function (...args) {
        const result = onConfigure?.apply(this, args);
        refresh(node);
        return result;
      };
    } catch (error) {
      console.warn("[Music2Prompts] widget visibility disabled:", error);
    }
  },
});
