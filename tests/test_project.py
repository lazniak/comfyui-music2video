"""One folder per project and per take, so a session's files stay together.

The folder name comes from a text widget, so the sanitising is the interesting part:
anything a user can type has to end up as one path component inside the output folder.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import render as render_module  # noqa: E402
from music2prompts.render import project_folder  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def node():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts.node import Music2PromptsLM

    return Music2PromptsLM


# --------------------------------------------------------------------------- the name


def test_the_take_number_is_padded_so_the_folders_sort():
    assert project_folder("clip", 1) == "clip_v001"
    assert project_folder("clip", 42) == "clip_v042"
    assert project_folder("clip", 0) == "clip_v000"


def test_a_take_past_the_padding_simply_widens():
    assert project_folder("clip", 1234) == "clip_v1234"


@pytest.mark.parametrize(
    "name",
    ["a/b", "a\\b", "../../etc", "con:trol", 'quo"te', "pipe|d", "star*", "quest?"],
)
def test_nothing_typed_here_can_write_outside_the_output_folder(name):
    """One path component, always: separators and the characters Windows refuses go."""
    folder = project_folder(name, 1)
    assert os.sep not in folder and "/" not in folder and ".." not in folder
    assert folder == os.path.basename(folder)


def test_letters_with_diacritics_are_kept():
    """The author writes Polish; mangling their project names would be its own bug."""
    assert project_folder("Mój klip", 2) == "Mój klip_v002"


def test_a_name_with_nothing_usable_in_it_falls_back_to_the_pack():
    assert project_folder("", 1) == "music2video_v001"
    assert project_folder("   ", 1) == "music2video_v001"
    assert project_folder("...", 1) == "music2video_v001"
    assert project_folder("///", 1) == "music2video_v001"


def test_a_very_long_name_is_cut_rather_than_hitting_the_path_limit():
    folder = project_folder("x" * 500, 1)
    assert len(folder) == 64 + len("_v001")


def test_a_take_number_that_is_not_a_number_does_not_take_the_run_down():
    assert project_folder("clip", "nonsense") == "clip_v000"
    assert project_folder("clip", -5) == "clip_v000"


# --------------------------------------------------------------------------- the files


@pytest.fixture
def output(tmp_path, monkeypatch):
    """Stand in for ComfyUI's output folder, keeping the subfolder argument honest."""
    def directory(subfolder=render_module.SUBFOLDER, temporary=False):
        path = os.path.join(str(tmp_path), "temp" if temporary else "output", subfolder)
        os.makedirs(path, exist_ok=True)
        return path

    monkeypatch.setattr(render_module, "output_directory", directory)
    return tmp_path


def test_frames_and_clips_of_one_take_land_in_that_take_s_folder(output):
    folder = os.path.join(render_module.SUBFOLDER, project_folder("my clip", 3))

    image = render_module.save_images([PNG], "take", "frame", "STAMP", folder)[0]
    clip = render_module.save_videos([b"mp4"], "take", stamp="STAMP", folder=folder)[0]

    assert os.path.basename(os.path.dirname(image)) == "my clip_v003"
    assert os.path.dirname(image) == os.path.dirname(clip), "one folder, not two"


def test_without_a_folder_everything_still_lands_where_it_used_to(output):
    """A caller that never heard of projects keeps writing into the pack's own folder."""
    image = render_module.save_images([PNG], "take", "frame", "STAMP")[0]
    assert os.path.basename(os.path.dirname(image)) == render_module.SUBFOLDER


def test_the_sidecars_follow_the_renders_into_the_same_folder(output):
    folder = os.path.join(render_module.SUBFOLDER, project_folder("my clip", 3))
    path = node()._save_text("hello", "take", "analysis", "json", "STAMP", folder)
    assert os.path.basename(os.path.dirname(path)) == "my clip_v003"
    assert open(path, encoding="utf-8").read() == "hello"


def test_the_take_number_is_what_keeps_two_runs_apart(output):
    folder_one = os.path.join(render_module.SUBFOLDER, project_folder("clip", 1))
    folder_two = os.path.join(render_module.SUBFOLDER, project_folder("clip", 2))
    first = render_module.save_images([PNG], "take", "frame", "S", folder_one)[0]
    second = render_module.save_images([PNG], "take", "frame", "S", folder_two)[0]
    assert os.path.dirname(first) != os.path.dirname(second)


# --------------------------------------------------------------------------- the widgets


def test_the_two_widgets_sit_at_the_top_where_they_were_asked_for():
    schema = node().define_schema()
    ids = [item.id for item in schema.inputs]
    assert ids[:4] == ["audio", "project_name", "iteration", "instruction"]


def test_the_take_number_counts_itself_up_after_every_run():
    """control_after_generate = increment is the whole point: no two runs share a folder."""
    schema = node().define_schema()  # skips first when ComfyUI is not on sys.path
    from comfy_api.latest import io  # noqa: PLC0415 - only importable with ComfyUI present

    iteration = next(item for item in schema.inputs if item.id == "iteration")
    assert iteration.control_after_generate == io.ControlAfterGenerate.increment
    assert iteration.default == 1
