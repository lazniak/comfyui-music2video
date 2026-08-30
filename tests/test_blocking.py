"""An empty output must not reach ComfyUI's list expansion.

ComfyUI slices every input once per downstream execution::

    return {k: v[i if len(v) > i else -1] for k, v in d.items()}   # execution.py

so a wire carrying an empty list falls through to ``v[-1]`` and raises IndexError
inside the executor - a traceback naming neither the node nor the wire. The node hands
out a silent ExecutionBlocker instead, which skips whatever is wired to that socket and
lets the rest of the graph finish.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.util import blocked_when_empty  # noqa: E402


@pytest.fixture
def blocker(monkeypatch):
    """Stand in for ComfyUI's ExecutionBlocker, which only exists inside ComfyUI."""

    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

    package = types.ModuleType("comfy_execution")
    module = types.ModuleType("comfy_execution.graph_utils")
    module.ExecutionBlocker = ExecutionBlocker
    package.graph_utils = module
    monkeypatch.setitem(sys.modules, "comfy_execution", package)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph_utils", module)
    return ExecutionBlocker


def test_something_that_has_content_is_handed_straight_back(blocker):
    values = [1, 2, 3]
    assert blocked_when_empty(values, "images") is values


def test_an_empty_list_leaves_as_a_blocker_wrapped_the_way_a_list_output_is(blocker):
    result = blocked_when_empty([], "images", "'image_provider' is 'pipe-steps'")
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], blocker)


def test_a_missing_single_value_leaves_as_a_bare_blocker(blocker):
    """final_video is not a list output, so it must not be handed out inside one."""
    result = blocked_when_empty(None, "final_video", "'concat_video' is off")
    assert isinstance(result, blocker)


def test_the_block_is_silent_so_a_prompts_only_run_still_succeeds(blocker):
    assert blocked_when_empty([], "videos")[0].message is None, (
        "a message would turn the default, render-nothing run into a failed one"
    )


def test_the_log_says_which_socket_went_empty_and_why(blocker, caplog):
    with caplog.at_level(logging.WARNING):
        blocked_when_empty([], "subject_images", "'render_subject_sheets' is off")
    assert "subject_images" in caplog.text
    assert "'render_subject_sheets' is off" in caplog.text


def test_outside_comfyui_the_value_is_returned_untouched(monkeypatch):
    """Tests and tooling import this package with no executor to block with."""
    monkeypatch.setitem(sys.modules, "comfy_execution.graph_utils", None)
    assert blocked_when_empty([], "images") == []
