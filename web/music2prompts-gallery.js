/**
 * Music2Prompts - the live gallery inside the node.
 *
 * A render is slow and paid for per shot, so the node shows each image and clip the
 * moment it lands instead of at the end of the batch. The backend writes every result
 * into ComfyUI's temp folder and sends one `music2prompts/preview` event per item; this
 * widget collects them and lets you page through with the arrows.
 *
 * Two things keep it out of the way of the rest of the node:
 *
 * * the widget is marked `serialize = false`, so it never enters `widgets_values`
 *   and cannot disturb the ~74 real inputs of a saved workflow;
 * * it stays hidden until the first result arrives, so a node that renders nothing
 *   looks exactly as it did before.
 *
 * After the run the same items come back in the node's `ui` payload, which is what
 * refills the gallery when the page is reloaded.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "Music2PromptsLM";
const EVENT = "music2prompts/preview";
const WIDGET = "m2p_gallery";

/** Every live gallery, so an event can find the node it belongs to.
 *
 * Not keyed by node id: `addDOMWidget` is called from `nodeCreated`, and until
 * `LGraphNode.configure` runs a node's id is still -1, so any map built there would
 * file every node under the same key. The node is looked up in the graph instead.
 */
const galleries = new Set();

const CSS = `
.m2p-gallery { display: flex; flex-direction: column; gap: 4px; width: 100%; height: 100%;
  font-family: inherit; font-size: 11px; color: var(--descrip-text, #b0b0b0); }
.m2p-stage { position: relative; flex: 1 1 auto; min-height: 120px; border-radius: 6px;
  background: var(--comfy-input-bg, #202020); overflow: hidden;
  display: flex; align-items: center; justify-content: center; }
.m2p-stage img, .m2p-stage video { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
.m2p-empty { opacity: 0.6; padding: 8px; text-align: center; }
.m2p-arrow { position: absolute; top: 50%; transform: translateY(-50%); border: 0; cursor: pointer;
  width: 26px; height: 44px; font-size: 18px; line-height: 1; color: #fff;
  background: rgba(0, 0, 0, 0.45); border-radius: 4px; }
.m2p-arrow:hover { background: rgba(0, 0, 0, 0.75); }
.m2p-arrow[disabled] { opacity: 0.25; cursor: default; }
.m2p-arrow.prev { left: 4px; } .m2p-arrow.next { right: 4px; }
.m2p-bar { display: flex; align-items: center; gap: 6px; padding: 0 2px; }
.m2p-bar .grow { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.m2p-live { color: #6cd06c; }
.m2p-dot { width: 6px; height: 6px; border-radius: 50%; background: #6cd06c;
  display: inline-block; margin-right: 4px; animation: m2p-pulse 1.2s ease-in-out infinite; }
@keyframes m2p-pulse { 50% { opacity: 0.25; } }
`;

function styleOnce() {
  if (document.getElementById("m2p-gallery-css")) return;
  const style = document.createElement("style");
  style.id = "m2p-gallery-css";
  style.textContent = CSS;
  document.head.appendChild(style);
}

function viewURL(item) {
  const params = new URLSearchParams({
    filename: item.filename,
    subfolder: item.subfolder || "",
    type: item.type || "temp",
  });
  return api.apiURL(`/view?${params}`);
}

class Gallery {
  constructor(node) {
    this.node = node;
    this.items = [];
    this.cursor = 0;
    this.follow = true; // jump to each new result until the user pages back
    this.expected = 0;
    this.build();
  }

  build() {
    styleOnce();
    const root = document.createElement("div");
    root.className = "m2p-gallery";
    root.innerHTML = `
      <div class="m2p-stage">
        <div class="m2p-empty">Rendered images and clips appear here.</div>
        <button class="m2p-arrow prev" title="Previous">&#8249;</button>
        <button class="m2p-arrow next" title="Next">&#8250;</button>
      </div>
      <div class="m2p-bar">
        <span class="grow"></span>
        <span class="count"></span>
      </div>`;
    this.root = root;
    this.stage = root.querySelector(".m2p-stage");
    this.empty = root.querySelector(".m2p-empty");
    this.prev = root.querySelector(".prev");
    this.next = root.querySelector(".next");
    this.label = root.querySelector(".grow");
    this.count = root.querySelector(".count");

    this.prev.onclick = () => this.step(-1);
    this.next.onclick = () => this.step(1);
    // the canvas would otherwise read a click on the video as a drag of the node
    for (const type of ["pointerdown", "wheel", "dblclick"]) {
      root.addEventListener(type, (event) => event.stopPropagation());
    }

    this.widget = this.node.addDOMWidget(WIDGET, "music2prompts_gallery", root, {
      hideOnZoom: false,
      getMinHeight: () => (this.widget?.hidden ? 0 : 220),
    });
    // LGraphNode.configure walks the widgets and skips `serialize === false` when it
    // replays widgets_values - the flag lives on the widget, not in its options.
    this.widget.serialize = false;
    this.widget.serializeValue = () => undefined;
    if (!this.widget.options) this.widget.options = {};
    this.setVisible(false);
  }

  setVisible(visible) {
    if (!this.widget) return;
    this.widget.hidden = !visible;
    this.widget.options.hidden = !visible;
    this.root.style.display = visible ? "" : "none";
    const descriptor = Object.getOwnPropertyDescriptor(this.node, "widgets");
    if (descriptor && typeof descriptor.set === "function") {
      this.node.widgets = [...(this.node.widgets || [])];
    }
    this.node.setDirtyCanvas?.(true, true);
  }

  reset(total = 0) {
    this.items = [];
    this.cursor = 0;
    this.follow = true;
    this.expected = Number(total) || 0;
    this.stage.querySelectorAll("img, video").forEach((element) => element.remove());
    this.empty.style.display = "";
    this.setVisible(false);
    this.draw();
  }

  add(item) {
    if (!item || !item.filename) return;
    // the same shot can be re-announced (a reload replays the ui payload)
    const key = `${item.kind}:${item.index}:${item.filename}`;
    if (this.items.some((existing) => existing.key === key)) return;
    this.items.push({ ...item, key, url: viewURL(item) });
    this.items.sort((left, right) => left.index - right.index || left.kind.localeCompare(right.kind));
    if (item.total) this.expected = Math.max(this.expected, Number(item.total) || 0);
    if (this.follow) this.cursor = this.items.findIndex((entry) => entry.key === key);
    this.setVisible(true);
    this.draw();
  }

  fill(items) {
    this.items = [];
    this.cursor = 0;
    for (const item of items || []) this.add(item);
  }

  step(direction) {
    if (!this.items.length) return;
    this.cursor = Math.min(this.items.length - 1, Math.max(0, this.cursor + direction));
    this.follow = this.cursor === this.items.length - 1;
    this.draw();
  }

  draw() {
    const item = this.items[this.cursor];
    this.prev.disabled = this.cursor <= 0;
    this.next.disabled = this.cursor >= this.items.length - 1;
    this.prev.style.display = this.next.style.display = this.items.length > 1 ? "" : "none";

    const pending = this.expected && this.items.length < this.expected;
    this.count.innerHTML = this.items.length
      ? `${this.cursor + 1} / ${this.items.length}${
          pending ? ` <span class="m2p-live"><span class="m2p-dot"></span>${this.expected}</span>` : ""
        }`
      : "";
    this.label.textContent = item ? `${item.kind === "video" ? "clip" : "image"} - ${item.label || ""}` : "";

    if (!item) {
      this.empty.style.display = "";
      return;
    }
    this.empty.style.display = "none";
    const wanted = item.kind === "video" ? "VIDEO" : "IMG";
    let element = this.stage.querySelector("img, video");
    if (!element || element.tagName !== wanted) {
      element?.remove();
      element = document.createElement(wanted === "VIDEO" ? "video" : "img");
      if (wanted === "VIDEO") {
        element.controls = true;
        element.loop = true;
        element.muted = true;
        element.playsInline = true;
        element.preload = "metadata";
      }
      this.stage.appendChild(element);
    }
    if (element.getAttribute("src") !== item.url) element.setAttribute("src", item.url);
  }
}

function galleryOf(node) {
  if (!node._m2pGallery || node._m2pGallery.node !== node) {
    node._m2pGallery = new Gallery(node);
    galleries.add(node._m2pGallery);
  }
  return node._m2pGallery;
}

/** The gallery of the node the backend named, whatever type its id arrived as. */
function galleryFor(id) {
  const graph = app.graph;
  const node = graph?.getNodeById?.(Number(id)) ?? graph?.getNodeById?.(id);
  if (node?._m2pGallery) return node._m2pGallery;
  for (const gallery of galleries) {
    if (String(gallery.node?.id) === String(id)) return gallery;
  }
  return null;
}

app.registerExtension({
  name: "music2prompts.gallery",

  setup() {
    api.addEventListener(EVENT, (event) => {
      const detail = event.detail || {};
      const gallery = galleryFor(detail.node);
      if (!gallery) return;
      if (detail.reset) gallery.reset(detail.total);
      else gallery.add(detail);
    });
  },

  async nodeCreated(node) {
    if (node.comfyClass !== NODE_ID) return;
    try {
      const gallery = galleryOf(node);

      // after a reload the results come back with the node's stored ui payload
      const onExecuted = node.onExecuted;
      node.onExecuted = function (message) {
        const result = onExecuted?.apply(this, arguments);
        if (message?.m2p_preview?.length) {
          gallery.fill(message.m2p_preview);
          gallery.setVisible(true);
        }
        return result;
      };

      const onRemoved = node.onRemoved;
      node.onRemoved = function () {
        galleries.delete(gallery);
        delete node._m2pGallery;
        return onRemoved?.apply(this, arguments);
      };
    } catch (error) {
      console.warn("[Music2Prompts] gallery disabled:", error);
    }
  },
});
