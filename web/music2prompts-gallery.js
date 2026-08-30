/**
 * Music2Video - the live gallery inside the node.
 *
 * A render is slow and paid for per shot, so the node shows each image and clip the
 * moment it lands instead of at the end of the batch. The backend writes every result
 * into ComfyUI's temp folder and sends one `music2prompts/preview` event per item; this
 * widget collects them and lets you page through with the controls under the media.
 *
 * Three rules govern the input handling here. All three were measured; an earlier
 * revision of this file justified the first two with a browser fact that is simply
 * false, so the reasoning is spelled out rather than asserted.
 *
 * 1. **Swallow the press before the graph sees it.** `stopPropagation` on `pointerdown`
 *    and `mousedown` stops a node drag starting under the control, and - the part that
 *    matters - stops an ancestor calling `setPointerCapture` between down and up.
 *    That capture is what actually kills a `click` inside a node: litegraph's
 *    `CanvasPointer` and the Vue node drag both take the pointer. `preventDefault()` on
 *    `pointerdown` does *not*: it suppresses only the compatibility mouse events
 *    (`mousedown`/`mousemove`/`mouseup`), and `click` still fires. Every working
 *    in-node control in the ecosystem relies on exactly this - KJNodes' `hdr_preview`
 *    and `fast_preview_batch`, mickmumpitz's `dataset_reviewer` - a plain `click`
 *    handler plus `stopPropagation` on the press.
 * 2. **Act on release, whichever path delivers it.** `click` is the one both a mouse
 *    and the keyboard produce (Enter and Space on a `<button>`), so it stays the
 *    primary path. `pointerup` acts as a fallback for a host that does swallow `click`,
 *    and a timestamp - not a sticky flag - keeps a single press from stepping twice:
 *    the compatibility `click` arrives in the same tick as the `pointerup` that
 *    produced it, a keyboard `click` never does.
 * 3. **Keep the controls out of the media box.** Not because an overlay cannot be hit:
 *    a positioned button paints - and hit-tests - above a non-positioned `<video>` flex
 *    item whatever the DOM order, and a `<video controls>` keeps its UA control bar
 *    inside its own border box. Both were measured. It is that nothing in the ecosystem
 *    overlays a control on a `<video controls>` inside a node (VideoHelperSuite turns
 *    the native controls off entirely; core's `ImagePreview.vue` makes the media
 *    `pointer-events: none`), and that letterboxing, small targets and Chrome's
 *    click-to-play on the bare video area all fight an overlay for no gain.
 *
 * What the arrows actually did before: they were hidden outright whenever the gallery
 * held a single item, and the end arrows used `disabled` - which eats the press
 * silently rather than letting it through. Both are gone.
 *
 * Two more things keep the widget out of the way of the rest of the node:
 *
 * * it is marked `serialize = false` (and `options.serialize = false`), so it never enters
 *   `widgets_values` or the API prompt;
 * * it stays hidden until the first result arrives, so a node that renders nothing looks
 *   exactly as it did before.
 *
 * Under the gallery sits a second widget with what the run cost. It is its own DOM
 * widget rather than a section of this one because the gallery hides itself whenever no
 * preview has arrived - which is exactly a prompts-only run, and exactly the start of
 * every run, both moments when a running total is worth seeing.
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
const COST_EVENT = "music2prompts/cost";
const WIDGET = "m2p_gallery";
const COST_WIDGET = "m2p_cost";

/** Every live gallery, so an event can find the node it belongs to.
 *
 * Not keyed by node id: `addDOMWidget` is called from `nodeCreated`, and until
 * `LGraphNode.configure` runs a node's id is still -1, so any map built there would
 * file every node under the same key. The node is looked up in the graph instead.
 */
const galleries = new Set();
const panels = new Set();

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

.m2p-cost { display: flex; flex-direction: column; width: 100%; height: 100%; gap: 2px;
  font-family: inherit; font-size: 11px; color: var(--descrip-text, #b0b0b0);
  background: var(--comfy-input-bg, #202020); border-radius: 6px; padding: 4px 6px;
  box-sizing: border-box; }
.m2p-cost-head { display: flex; align-items: baseline; gap: 6px; }
.m2p-cost-title { flex: 0 0 auto; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7; }
.m2p-cost-total { flex: 1 1 auto; text-align: right; font-size: 13px; color: #eee;
  font-variant-numeric: tabular-nums; }
.m2p-cost-rows { display: flex; flex-direction: column; gap: 1px; overflow-y: auto;
  max-height: 64px; scrollbar-width: thin; }
.m2p-cost-row { display: flex; align-items: baseline; gap: 6px; }
.m2p-cost-row[data-prov="estimated"], .m2p-cost-row[data-prov="unknown"] { opacity: 0.75; }
.m2p-cost-model { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.m2p-cost-qty { flex: 0 0 auto; opacity: 0.6; font-variant-numeric: tabular-nums; }
.m2p-cost-usd { flex: 0 0 auto; min-width: 62px; text-align: right;
  font-variant-numeric: tabular-nums; }
.m2p-cost-note { opacity: 0.55; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
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
 * The press is stopped before the graph canvas can see it - that is what keeps the node
 * from being dragged out from under the control and keeps an ancestor from capturing the
 * pointer, which is the one thing that would stop the `click` arriving. The action then
 * runs on `click`, so the keyboard reaches it too.
 *
 * `pointerup` is a fallback for a host that swallows `click` anyway. The two are told
 * apart by time, not by a flag: the compatibility `click` shares the `pointerup`'s tick,
 * a keyboard `click` has no `pointerup` before it at all. A flag would go stale the first
 * time a press is released off the control and no `click` ever comes to clear it.
 */
function onPress(element, handler) {
  let released = -Infinity;
  let pressed = false;
  const swallow = (event) => event.stopPropagation();
  element.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    pressed = true;
  });
  // the legacy canvas listens for mouse events as well as pointer ones
  element.addEventListener("mousedown", swallow);
  element.addEventListener("pointerup", (event) => {
    if (event.button !== 0 || !pressed) return;
    event.stopPropagation();
    pressed = false;
    released = event.timeStamp;
    handler(event);
  });
  element.addEventListener("click", (event) => {
    event.stopPropagation();
    if (event.timeStamp - released > 100) handler(event);
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

/** How well a figure is known, in one character in front of it.
 *
 * The distinction is the whole point of the panel: `billed` is what the provider said it
 * charged, `computed` is its live unit price times units it reported, `estimated` is a
 * price table that ships in the node. Presenting those as the same number would be the
 * one failure worth avoiding here.
 */
const MARK = { billed: "", computed: "~", estimated: "\u2248", unknown: "", free: "" };

const WORDS = {
  billed: "charged by the provider",
  computed: "live unit price x units billed",
  estimated: "built-in price table x tokens used",
  unknown: "no price available",
  free: "runs on your own machine",
};

class CostPanel {
  constructor(node) {
    this.node = node;
    this.rows = [];
    this.build();
  }

  build() {
    styleOnce();
    const root = document.createElement("div");
    root.className = "m2p-cost";
    root.innerHTML = `
      <div class="m2p-cost-head">
        <span class="m2p-cost-title">cost</span>
        <span class="m2p-cost-total"></span>
      </div>
      <div class="m2p-cost-rows"></div>
      <div class="m2p-cost-note"></div>`;
    this.root = root;
    this.total = root.querySelector(".m2p-cost-total");
    this.list = root.querySelector(".m2p-cost-rows");
    this.note = root.querySelector(".m2p-cost-note");

    // scroll the rows rather than zooming the graph out from under them
    this.list.addEventListener("wheel", (event) => {
      if (this.list.scrollHeight <= this.list.clientHeight) return;
      event.stopPropagation();
      event.preventDefault();
      this.list.scrollTop += event.deltaY;
    });

    this.widget = this.node.addDOMWidget(COST_WIDGET, "music2prompts_cost", root, {
      hideOnZoom: false,
      getMinHeight: () => (this.widget?.hidden ? 0 : 30 + 15 * Math.min(this.rows.length, 4) + 16),
      serialize: false,
    });
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

  reset() {
    this.rows = [];
    this.list.replaceChildren();
    this.total.textContent = "";
    this.note.textContent = "";
    this.setVisible(false);
  }

  /** Replace the whole table. The backend always sends the running total, never a delta,
   * so accumulating here would double every row when the `ui` payload replays it. */
  fill(payload) {
    if (!payload) return;
    const data = Array.isArray(payload) ? payload[0] : payload;
    if (!data) return;
    this.rows = data.models || [];
    this.draw(data);
    // a run that only ever spent nothing is worth showing at the end, not mid-flight
    const spent = this.rows.some((row) => row.provenance !== "free");
    if (spent || data.final) this.setVisible(true);
  }

  draw(data) {
    const total = data.total || {};
    const pending = !data.final;
    this.total.textContent =
      `${MARK[total.provenance] || ""}${total.display || "$0.00"}` +
      (pending ? " ..." : "");
    this.total.className = `m2p-cost-total${pending ? " m2p-live" : ""}`;
    this.total.title = WORDS[total.provenance] || "";

    this.list.replaceChildren(
      ...this.rows.map((row) => {
        const line = document.createElement("div");
        line.className = "m2p-cost-row";
        line.dataset.prov = row.provenance || "unknown";

        const model = document.createElement("span");
        model.className = "m2p-cost-model";
        // written as text, never as HTML: every string here comes off the wire
        model.textContent = row.model || row.provider;

        const quantity = document.createElement("span");
        quantity.className = "m2p-cost-qty";
        quantity.textContent =
          row.units && row.unit_usd
            ? `${row.units} ${row.unit} x $${row.unit_usd}`
            : row.tokens
              ? `${row.tokens.input}+${row.tokens.output} tok`
              : `${row.calls} call${row.calls === 1 ? "" : "s"}`;

        const money = document.createElement("span");
        money.className = "m2p-cost-usd";
        money.textContent = `${MARK[row.provenance] || ""}${row.display}`;

        line.title =
          `${row.provider} - ${row.calls} call${row.calls === 1 ? "" : "s"} - ` +
          `${WORDS[row.provenance] || row.provenance}` +
          (row.note ? ` - ${row.note}` : "");
        line.append(model, quantity, money);
        return line;
      })
    );

    const notes = [];
    if (total.unpriced_calls) notes.push(`${total.unpriced_calls} unpriced`);
    if (total.possibly_billed_calls) {
      notes.push(`${total.possibly_billed_calls} refused after running`);
    }
    if (Number(total.failed_usd || 0) > 0) notes.push(`${total.failed_display} on failed calls`);
    notes.push(total.note || "");
    this.note.textContent = notes.filter(Boolean).join(" \u00b7 ");
    this.note.title = this.note.textContent;
  }
}

function costOf(node) {
  if (!node._m2pCost || node._m2pCost.node !== node) {
    node._m2pCost = new CostPanel(node);
    panels.add(node._m2pCost);
  }
  return node._m2pCost;
}

function costFor(id) {
  const graph = app.graph;
  const node = graph?.getNodeById?.(Number(id)) ?? graph?.getNodeById?.(id);
  if (node?._m2pCost) return node._m2pCost;
  const local = String(id).split(":").pop();
  for (const panel of panels) {
    const own = String(panel.node?.id);
    if (own === String(id) || own === local) return panel;
  }
  return null;
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

    // its own event type: the preview listener drops anything without a filename
    api.addEventListener(COST_EVENT, (event) => {
      const detail = event.detail || {};
      costFor(detail.node)?.fill(detail);
    });
  },

  async nodeCreated(node) {
    if (node.comfyClass !== NODE_ID) return;
    try {
      const gallery = galleryOf(node);
      // widget order is creation order, so this lands under the previews
      const cost = costOf(node);

      // after a reload the results come back with the node's stored ui payload
      const onExecuted = node.onExecuted;
      node.onExecuted = function (message) {
        const result = onExecuted?.apply(this, arguments);
        if (message?.m2p_preview?.length) {
          gallery.fill(message.m2p_preview);
          gallery.setVisible(true);
        }
        if (message?.m2p_cost) cost.fill(message.m2p_cost);
        return result;
      };

      const onRemoved = node.onRemoved;
      node.onRemoved = function () {
        galleries.delete(gallery);
        panels.delete(cost);
        delete node._m2pGallery;
        delete node._m2pCost;
        return onRemoved?.apply(this, arguments);
      };
    } catch (error) {
      console.warn("[Music2Video] gallery disabled:", error);
    }
  },
});
