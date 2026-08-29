"""The pack's own HTTP route, used by the node's JS to refresh the model dropdowns.

``GET /music2prompts/models``            -> every list, as ``{kind: [ids]}``
``GET /music2prompts/models?kind=fal_video``  -> one list, as a bare JSON array
``&force=1``                             -> re-probe now, ignoring the cache TTL

The probing runs in a worker thread: a blocking HTTP call inside an aiohttp handler
would stall ComfyUI's whole event loop, progress websocket included.
"""

from __future__ import annotations

import asyncio

from . import model_cache
from .util import log, warn

ROUTE = "/music2prompts/models"
_REGISTERED = False


async def _handle(request):
    from aiohttp import web

    kind = (request.query.get("kind") or "").strip()
    force = request.query.get("force") in {"1", "true", "yes"}

    try:
        if kind:
            if kind not in model_cache.KINDS:
                return web.json_response({"error": f"unknown kind '{kind}'"}, status=400)
            values = await asyncio.to_thread(model_cache.resolve, kind, force)
            return web.json_response(values)

        kinds = model_cache.KINDS
        results = await asyncio.gather(
            *(asyncio.to_thread(model_cache.resolve, item, force) for item in kinds)
        )
        return web.json_response(dict(zip(kinds, results)))
    except Exception as exc:  # never take the server down over a dropdown
        warn(f"model route failed: {exc}")
        return web.json_response({"error": str(exc)}, status=500)


def register() -> bool:
    """Attach the route to ComfyUI's server. Safe to call more than once."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from server import PromptServer  # type: ignore
    except Exception as exc:
        warn(f"could not import ComfyUI's server ({exc}); model lists will not refresh in the UI")
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None or not hasattr(instance, "routes"):
        warn("ComfyUI's PromptServer is not up yet; model lists will not refresh in the UI")
        return False
    try:
        instance.routes.get(ROUTE)(_handle)
    except Exception as exc:
        warn(f"could not register {ROUTE} ({exc}); model lists will not refresh in the UI")
        return False
    _REGISTERED = True
    log(f"model list route ready at {ROUTE}")
    return True
