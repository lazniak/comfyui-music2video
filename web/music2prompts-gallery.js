/**
 * Music2Prompts - the live gallery inside the node.
 *
 * A render is slow and paid for per shot, so the node shows each image and clip the
 * moment it lands instead of at the end of the batch. The backend writes every result
 * into ComfyUI's temp folder and sends one `music2prompts/preview` event per item; this
 * widget collects them and lets you page through with the controls under the media.
 *
 * Two rules govern the input handling here, both established by measurement:
 *
 * 1. **Navigate on `pointerdown`, never on `click`.** Inside a node the graph canvas
 *    expects to own the pointer, so hosts call `preventDefault()` on `pointerdown` to keep
 *    a drag over a widget moving the node - VideoHelperSuite does exactly that, forwarding
 *    to `app.canvas._mousedown_callback`. A cancelled `pointerdown` never produces the
 *    compatibility `click`, so an `onclick` control inside a node is dead on arrival while
 *    `pointerdown` listeners still fire.
 * 2. **Keep the controls out of the media box.** A `<video controls>` answers pointer
 *    events from its own shadow DOM, so anything floating over it competes with the play
 *    button for the same pixels. The controls sit in their own row underneath.
 *
 * Two more things keep the widget out of the way of the rest of the node:
 *
 * * it is marked `serialize = false` (and `options.serialize = false`), so it never enters
 *   `widgets_values` or the API prompt;
 * * it stays hidden until the first result arrives, so a node that renders nothing looks
 *   exactly as it did before.
 *
 * After the run the same items come back in the node's `ui` payload, so reopening the
 * finished job from the queue sidebar refills the gallery. A plain F5 does not: node
 * previews live only in the frontend's memory, and the files are in the temp folder,
 * which ComfyUI wipes on its next start.
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
.m2p-stage { position: relative; flex: 1 1 auto; min-height: 110px; border-radius: 6px;
  background: var(--comfy-input-bg, #202020); overflow: hidden;
  display: flex; align-items: center; justify-content: center; }
.m2p-stage img, .m2p-stage video { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
.m2p-empty { opacity: 0.6; padding: 8px; text-align: center; }
.m2p-row { display: flex; align-items: center; gap: 6px; }
.m2p-step { flex: 0 0 auto; width: 28px; height: 22px; border: 0; border-radius: 4px; cursor: pointer;
  font-size: 15px; line-height: 1; color: #eee; background: var(--comfy-input-bg, #303030);
  touch-action: none; user-select: none; }
.m2p-step:hover { background: #4a4a4a; }
.m2p-step[data-off="1"] { opacity: 0.3; cursor: default; }
.m2p-label { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.m2p-count { flex: 0 0 auto; font-variant-numeric: tabular-nums; }
.m2p-strip { display: flex; gap: 3px; overflow-x: auto; overflow-y: hidden; min-height: 30px;
  scrollbar-width: thin; }
.m2p-thumb { flex: 0 0 auto; width: 40px; height: 26px; border-radius: 3px; cursor: pointer;
  background: var(--comfy-input-bg, #303030) center/cover no-repeat; border: 1px solid transparent;
  position: relative; touch-action: none; }
.m2p-thumb[data-current="1"] { border-color: #6cd06c; }
.m2p-thumb[data-kind="video"]::after { content: "\\25B6"; position: absolute; right: 2px; bottom: 0;
  font-size: 8px; color: #fff; text-shadow: 0 0 3px #000; }
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

/** Wire a control so that it actually works inside a node.
 *
 * `pointerdown` is the event that survives a host cancelling the default action, and
 * `stopPropagation` keeps the same press from also starting a node drag underneath. The
 * `click` path stays as a fallback for hosts that leave `pointerdown` alone, guarded so a
 * single press never runs the handler twice.
 */
function onPress(element, handler) {
  element.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    element._m2pHandled = true;
    handler(event);
  });
  element.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!element._m2pHandled) handler(event);
    element._m2pHandled = false;
  });
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
      <div class="m2p-stage"><div class="m2p-empty">Rendered images and clips appear here.</div></div>
      <div class="m2p-row">
        <button class="m2p-step prev" title="Previous (Left arrow)">&#8249;</button>
        <button class="m2p-step next" title="Next (Right arrow)">&#8250;</button>
        <span class="m2p-label"></span>
        <span class="m2p-count"></span>
      </div>
      <div class="m2p-strip"></div>`;
    this.root = root;
    this.stage = root.querySelector(".m2p-stage");
    this.empty = root.querySelector(".m2p-empty");
    this.prev = root.querySelector(".prev");
    this.next = root.querySelector(".next");
    this.label = root.querySelector(".m2p-label");
    this.count = root.querySelector(".m2p-count");
    this.strip = root.querySelector(".m2p-strip");

    onPress(this.prev, () => this.step(-1));
    onPress(this.next, () => this.step(1));
    // scroll the thumbnails instead of zooming the graph out from under them
    this.strip.addEventListener("wheel", (event) => {
      if (this.strip.scrollWidth <= this.strip.clientWidth) return;
      event.stopPropagation();
      event.preventDefault();
      this.strip.scrollLeft += event.deltaY + event.deltaX;
    });
    root.tabIndex = 0;
    root.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.stopPropagation();
      this.step(event.key === "ArrowLeft" ? -1 : 1);
    });

    this.widget = this.node.addDOMWidget(WIDGET, "music2prompts_gallery", root, {
      hideOnZoom: false,
      getMinHeight: () => (this.widget?.hidden ? 0 : 250),
      // options.serialize keeps the gallery out of the API prompt, where it would
      // arrive at the node as an input nobody declared
      serialize: false,
    });
    // ...and widget.serialize (a different flag, on the instance) is the one
    // LGraphNode.configure reads when it replays widgets_values.
    this.widget.serialize = false;
    this.widget.serializeValue = () => undefined;
    if (!this.widget.options) this.widget.options = {};
    // addWidget already chains node.onRemoved -> widget.onRemove; keep that and stop
    // a clip that is still playing when the node goes away
    const onRemove = this.widget.onRemove;
    this.widget.onRemove = () => {
      this.stage?.querySelector("video")?.pause();
      onRemove?.call(this.widget);
    };
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
    this.strip.replaceChildren();
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
    this.show(this.cursor + direction);
  }

  show(index) {
    if (!this.items.length) return;
    this.cursor = Math.min(this.items.length - 1, Math.max(0, index));
    this.follow = this.cursor === this.items.length - 1;
    this.draw();
  }

  /** The thumbnail strip, rebuilt only when the set of items changed. */
  drawStrip() {
    if (this.strip.childElementCount !== this.items.length) {
      this.strip.replaceChildren(
        ...this.items.map((item, index) => {
          const thumb = document.createElement("div");
          thumb.className = "m2p-thumb";
          thumb.dataset.kind = item.kind;
          thumb.title = `${item.kind === "video" ? "clip" : "image"} - ${item.label || index + 1}`;
          // a video frame cannot be a CSS background, so only stills carry a picture
          if (item.kind !== "video") thumb.style.backgroundImage = `url("${item.url}")`;
          onPress(thumb, () => this.show(index));
          return thumb;
        })
      );
    }
    for (const [index, thumb] of [...this.strip.children].entries()) {
      thumb.dataset.current = index === this.cursor ? "1" : "0";
    }
  }

  draw() {
    const item = this.items[this.cursor];
    // `disabled` would stop the button dispatching anything at all; a data flag keeps the
    // element live so a press is always seen, and only the handler decides to do nothing
    this.prev.dataset.off = this.cursor <= 0 ? "1" : "0";
    this.next.dataset.off = this.cursor >= this.items.length - 1 ? "1" : "0";

    const pending = this.expected && this.items.length < this.expected;
    this.count.innerHTML = this.items.length
      ? `${this.cursor + 1} / ${this.items.length}${
          pending ? ` <span class="m2p-live"><span class="m2p-dot"></span>${this.expected}</span>` : ""
        }`
      : "";
    this.label.textContent = item ? `${item.kind === "video" ? "clip" : "image"} - ${item.label || ""}` : "";
    this.drawStrip();

    if (!item) {
      this.empty.style.display = "";
      return;
    }
    this.empty.style.display = "none";
    const wanted = item.kind === "video" ? "VIDEO" : "IMG";
    let element = this.stage.querySelector("img, video");
    if (!element || element.tagName !== wanted) {
      element?.pause?.();
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
    if (element.getAttribute("src") !== item.url) {
      element.pause?.();
      element.setAttribute("src", item.url);
    }
  }
}

function galleryOf(node) {
  if (!node._m2pGallery || node._m2pGallery.node !== node) {
    node._m2pGallery = new Gallery(node);
    galleries.add(node._m2pGallery);
  }
  return node._m2pGallery;
}

/** The gallery of the node the backend named, whatever type its id arrived as.
 *
 * Inside a subgraph the executing id is a colon path ("12:34") rather than a plain
 * node id, and that node is not in `app.graph` at all - so the last segment is
 * matched against the galleries that exist.
 */
function galleryFor(id) {
  const graph = app.graph;
  const node = graph?.getNodeById?.(Number(id)) ?? graph?.getNodeById?.(id);
  if (node?._m2pGallery) return node._m2pGallery;
  const local = String(id).split(":").pop();
  for (const gallery of galleries) {
    const own = String(gallery.node?.id);
    if (own === String(id) || own === local) return gallery;
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
