"""End-to-end plumbing test with a stubbed LM Studio.

Skipped automatically outside a ComfyUI environment (needs ``comfy_api`` and torch).
Run it with ComfyUI's own python, e.g.::

    <ComfyUI>/.venv/Scripts/python.exe -m pytest tests/test_pipeline_offline.py -q

The same checks are available without pytest through ``tests/run_pipeline_check.py``.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("torch")
pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")

from fake_lmstudio import SUBJECTS  # noqa: E402
from run_pipeline_check import CHECKS, run_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def outputs():
    return run_pipeline()


@pytest.mark.parametrize("name,check", list(CHECKS.items()))
def test_pipeline(name, check, outputs):
    check(outputs)


def test_subject_outputs_match_the_bible(outputs):
    assert len(outputs.args[2]) == len(SUBJECTS)
    json.loads(outputs.args[11])
