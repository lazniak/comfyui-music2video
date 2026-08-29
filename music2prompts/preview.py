"""Live previews: every finished image and clip shows up in the node while the run continues.

A render can take minutes per shot, so waiting for the whole batch before seeing
anything is the difference between spotting a bad prompt at shot 1 and paying for
twelve of them. Each result is written to ComfyUI's temp folder as soon as it lands
and announced over the websocket the frontend is already connected to; the node's
gallery widget picks the event up and shows it.

The same entries are returned in the node's ``ui`` payload at the end, so the gallery
survives a page reload - that is how ComfyUI's own preview nodes persist.
"""

from __future__ import annotations

import os
import time

from .util import PREFIX, warn

EVENT = "music2prompts/preview"
SUBFOLDER = "music2prompts"

EXTENSIONS = {"image": "png", "video": "mp4"}


def _send(payload: dict) -> bool:
    """Push one event to the browser. ``send_sync`` is safe from a worker thread."""
    try:
        import server  # type: ignore

        instance = server.PromptServer.instance
    except Exception:
        return False
    if instance is None:
        return False
    try:
        instance.send_sync(EVENT, payload)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"could not send a preview event: {exc}")
        return False


class PreviewFeed:
    """Writes each finished render to the temp folder and tells one node about it.

    Nothing here is allowed to break a run: a failed write or a closed socket costs
    the preview, never the render that was paid for.
    """

    def __init__(self, node_id, prefix: str = "music2prompts", enabled: bool = True) -> None:
        self.node_id = str(node_id) if node_id is not None else ""
        self.prefix = prefix or "music2prompts"
        self.enabled = bool(enabled) and bool(self.node_id)
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.items: list[dict] = []
        self._directory = ""

    # ------------------------------------------------------------------ writing

    def directory(self) -> str:
        if not self._directory:
            from .render import output_directory

            self._directory = output_directory(SUBFOLDER, temporary=True)
        return self._directory

    def _write(self, kind: str, index: int, payload: bytes) -> str:
        extension = EXTENSIONS.get(kind, "bin")
        name = f"{self.prefix}_{self.stamp}_{kind}{index + 1:03d}.{extension}"
        path = os.path.join(self.directory(), name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    # ------------------------------------------------------------------ events

    def reset(self, total: int = 0) -> None:
        """Clear the gallery at the start of a run, so it never mixes two takes."""
        self.items = []
        if self.enabled:
            _send({"node": self.node_id, "reset": True, "total": int(total)})

    def publish(self, kind: str, index: int, payload: bytes, label: str = "", total: int = 0) -> str:
        """Write one result and announce it. Returns the path, or "" if it was skipped."""
        if not self.enabled or not payload:
            return ""
        try:
            path = self._write(kind, index, payload)
        except OSError as exc:
            warn(f"could not write a {kind} preview: {exc}")
            return ""
        item = {
            "node": self.node_id,
            "kind": kind,
            "index": int(index),
            "total": int(total),
            "label": label,
            "filename": os.path.basename(path),
            "subfolder": SUBFOLDER,
            "type": "temp",
        }
        self.items.append(item)
        _send(item)
        return path

    def paths(self, kind: str) -> dict[int, str]:
        """Files already written for one kind, by shot index - so nothing is written twice."""
        found = {}
        for item in self.items:
            if item["kind"] == kind:
                found[item["index"]] = os.path.join(self.directory(), item["filename"])
        return found

    def ui(self) -> dict:
        """The payload that keeps the gallery filled after a page reload."""
        return {"m2p_preview": list(self.items)} if self.items else {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PreviewFeed {PREFIX} node={self.node_id} items={len(self.items)}>"
