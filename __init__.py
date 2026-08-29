"""ComfyUI entry point for the Music2Prompts node pack (V3 schema).

Imports of ``comfy_api`` happen inside ``comfy_entrypoint`` so this module can
also be imported outside ComfyUI (tests, tooling) without blowing up.
"""

from __future__ import annotations

__version__ = "0.2.0"

# served to the browser: hides the model widgets of providers you did not select
WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]


async def comfy_entrypoint():
    try:
        from comfy_api.latest import ComfyExtension, io
    except ImportError as exc:  # pragma: no cover - only on very old ComfyUI
        raise ImportError(
            "[Music2Prompts] requires the ComfyUI V3 node API (comfy_api.latest). "
            "Update ComfyUI to a recent version and restart."
        ) from exc

    from .music2prompts.node import Music2PromptsLM

    class Music2PromptsExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [Music2PromptsLM]

    return Music2PromptsExtension()
