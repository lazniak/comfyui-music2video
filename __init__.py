"""ComfyUI entry point for the Music2Video node pack (V3 schema).

Imports of ``comfy_api`` happen inside ``comfy_entrypoint`` so this module can
also be imported outside ComfyUI (tests, tooling) without blowing up.
"""

from __future__ import annotations

__version__ = "1.1.0"

# served to the browser: hides the widgets of providers you did not select, refreshes
# the model dropdowns from the route registered below, and shows each rendered image
# and clip in the node as it arrives
WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]


def _bootstrap() -> None:
    """Register the model-list route and start filling the cache in the background.

    Both are best-effort: outside ComfyUI (tests) there is no server to register on,
    and a failure here must never cost the user the whole node pack.
    """
    try:
        from .music2prompts import http_routes, model_cache
    except ImportError:
        return  # imported outside a package context (tests, tooling) - nothing to do
    try:
        http_routes.register()
        model_cache.warm()
    except Exception as exc:  # pragma: no cover - defensive
        import logging

        logging.getLogger("music2prompts").warning(
            "[Music2Video] startup hook failed (%s); the node still works, "
            "but the model dropdowns will only show their static fallbacks",
            exc,
        )


_bootstrap()


async def comfy_entrypoint():
    try:
        from comfy_api.latest import ComfyExtension, io
    except ImportError as exc:  # pragma: no cover - only on very old ComfyUI
        raise ImportError(
            "[Music2Video] requires the ComfyUI V3 node API (comfy_api.latest). "
            "Update ComfyUI to a recent version and restart."
        ) from exc

    from .music2prompts.expand import Music2PromptsPipeExpand
    from .music2prompts.node import Music2PromptsLM

    class Music2PromptsExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [Music2PromptsLM, Music2PromptsPipeExpand]

    return Music2PromptsExtension()
