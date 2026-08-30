"""Cancel has to mean now.

The node holds the graph for minutes at a time - a Whisper window, a local 27B model
writing a stage, a queue of renders being polled - and every one of those used to run to
completion after the button was pressed. Worse, the run that finally stopped left the
models resident, so the next thing the graph loaded hit an out-of-memory error that named
neither of them.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import asr as asr_module  # noqa: E402
from music2prompts import render as render_module  # noqa: E402
from music2prompts import util as util_module  # noqa: E402
from music2prompts.util import (  # noqa: E402
    interrupted,
    raise_if_interrupted,
    run_cancellable,
    sleep_unless_interrupted,
)


class Cancelled(BaseException):
    """ComfyUI's InterruptProcessingException is a BaseException too - that is the point.

    It travels up through the retry ladders in this pack (all of which catch ``Exception``)
    without being mistaken for a failure worth retrying.
    """


class FakeComfy:
    """The two things this pack asks comfy.model_management for."""

    InterruptProcessingException = Cancelled

    def __init__(self, pressed: bool = False) -> None:
        self.pressed = pressed
        self.asked = 0

    def processing_interrupted(self) -> bool:
        self.asked += 1
        return self.pressed


@pytest.fixture
def comfy(monkeypatch):
    fake = FakeComfy()
    monkeypatch.setattr(util_module, "_MODEL_MANAGEMENT", fake)
    return fake


# --------------------------------------------------------------------------- the flag


def test_nothing_is_raised_while_the_run_is_still_wanted(comfy):
    raise_if_interrupted()
    assert interrupted() is False


def test_the_flag_is_left_set_so_every_thread_sees_the_cancel(comfy):
    """ComfyUI's own throw_exception_if_processing_interrupted() clears it as it raises.

    With renders running four at a time that would mean one worker eats the cancel and
    the other three carry on paying for clips nobody will watch. The executor resets the
    flag itself at the start of the next prompt, so leaving it set costs nothing.
    """
    comfy.pressed = True
    for _ in range(3):
        with pytest.raises(Cancelled):
            raise_if_interrupted()
    assert comfy.pressed is True
    assert interrupted() is True


def test_a_cancel_is_not_mistaken_for_a_failure_worth_retrying(comfy):
    """Every stage in this pack sits inside `except Exception`. None may swallow this."""
    comfy.pressed = True

    def ladder():
        try:
            raise_if_interrupted()
        except Exception:  # noqa: BLE001 - exactly the handler under test
            return "swallowed"
        return "never reached"

    with pytest.raises(Cancelled):
        ladder()


def test_outside_comfyui_the_checks_are_inert(monkeypatch):
    """The pure-python modules are unit tested without ComfyUI, and must stay runnable."""
    monkeypatch.setattr(util_module, "_MODEL_MANAGEMENT", False)
    raise_if_interrupted()
    assert interrupted() is False
    sleep_unless_interrupted(0.0)


# --------------------------------------------------------------------------- waiting


def test_a_wait_ends_when_the_cancel_comes_not_when_the_timer_does(comfy):
    """fal is polled every two seconds and OpenRouter every three."""
    comfy.pressed = True
    started = time.perf_counter()
    with pytest.raises(Cancelled):
        sleep_unless_interrupted(30.0)
    assert time.perf_counter() - started < 1.0


def test_a_wait_nobody_cancelled_is_still_a_wait(comfy):
    started = time.perf_counter()
    sleep_unless_interrupted(0.3)
    assert time.perf_counter() - started >= 0.28


def test_the_cancel_is_noticed_part_way_through_a_long_wait(comfy):
    def press() -> None:
        time.sleep(0.2)
        comfy.pressed = True

    threading.Thread(target=press, daemon=True).start()
    started = time.perf_counter()
    with pytest.raises(Cancelled):
        sleep_unless_interrupted(30.0)
    assert time.perf_counter() - started < 2.0


# --------------------------------------------------------------------------- blocking calls


def test_a_blocking_call_hands_its_answer_back(comfy):
    assert run_cancellable(lambda: 7) == 7


def test_a_blocking_call_reports_its_own_failure_on_the_calling_thread(comfy):
    def work():
        raise ValueError("the model refused")

    with pytest.raises(ValueError, match="refused"):
        run_cancellable(work)


def test_the_node_stops_without_waiting_for_the_request_it_is_inside(comfy):
    """requests cannot abort a call in flight; the reply is left to arrive unread.

    What matters is that the node is out of the way at once - and the cancel path then
    unloads the model, which ends the generation at the far end.
    """
    finished = threading.Event()

    def work():
        time.sleep(5.0)
        finished.set()
        return "too late"

    comfy.pressed = True
    started = time.perf_counter()
    with pytest.raises(Cancelled):
        run_cancellable(work)
    assert time.perf_counter() - started < 1.5
    assert not finished.is_set(), "it must not have waited for the call to come back"


# --------------------------------------------------------------------------- fan-out


def test_the_renders_queued_behind_a_cancel_are_never_paid_for(comfy):
    """One shot per job, one bill per job: the ones not started yet must stay unstarted."""
    ran: list[int] = []

    def job(index: int):
        def run():
            ran.append(index)
            if index == 0:
                comfy.pressed = True
            return index

        return run

    with pytest.raises(Cancelled):
        render_module._in_parallel([job(index) for index in range(5)], 1, "clip")
    assert ran == [0]


def test_a_finished_fan_out_still_returns_everything(comfy):
    jobs = [(lambda value=value: value) for value in range(4)]
    assert render_module._in_parallel(jobs, 2, "clip") == [0, 1, 2, 3]


# --------------------------------------------------------------------------- whisper


def test_whisper_is_told_to_stop_mid_window(comfy):
    """A 30 s window decodes token by token, and a run is many windows long."""
    pytest.importorskip("transformers", reason="the stopping rule is a transformers object")

    class Pipe:
        class model:
            @staticmethod
            def generate(input_features=None, stopping_criteria=None, **kwargs):
                return None

    criteria = asr_module._cancel_criteria(Pipe())
    assert criteria is not None
    assert criteria[0](None, None) is False
    comfy.pressed = True
    assert criteria[0](None, None) is True


def test_a_transformers_without_stopping_criteria_is_left_alone(comfy):
    """Passing an argument generate() does not take would cost us the transcription."""
    pytest.importorskip("transformers")

    class Pipe:
        class model:
            @staticmethod
            def generate(input_features=None, **kwargs):
                return None

    assert asr_module._cancel_criteria(Pipe()) is None


# --------------------------------------------------------------------------- giving the card back


def node_module():
    pytest.importorskip("comfy_api", reason="needs a ComfyUI installation on sys.path")
    from music2prompts import node

    return node


def test_a_cancelled_run_hands_the_models_back(monkeypatch):
    """This is the out-of-memory the user actually hit: cancel, then the next node loads."""
    node = node_module()
    done: list[str] = []
    monkeypatch.setattr(asr_module, "release_cache", lambda: done.append("allocator"))

    @node._releases_the_card
    def run():
        node._on_cancel(lambda: done.append("whisper"))
        node._on_cancel(lambda: done.append("lm studio"))
        raise Cancelled()

    with pytest.raises(Cancelled):
        run()
    assert done == ["whisper", "lm studio", "allocator"], "the allocator is emptied last"


def test_a_run_that_finishes_frees_nothing_twice(monkeypatch):
    """The end of a run already unloads what it asked for; this path is for the other exits."""
    node = node_module()
    done: list[str] = []
    monkeypatch.setattr(asr_module, "release_cache", lambda: done.append("allocator"))

    @node._releases_the_card
    def run():
        node._on_cancel(lambda: done.append("whisper"))
        return "finished"

    assert run() == "finished"
    assert done == []


def test_a_cleanup_that_fails_does_not_replace_the_cancel(monkeypatch):
    """LM Studio may well be the thing that died; the user still asked to stop."""
    node = node_module()
    done: list[str] = []
    monkeypatch.setattr(asr_module, "release_cache", lambda: done.append("allocator"))

    def explode():
        raise RuntimeError("LM Studio is gone")

    @node._releases_the_card
    def run():
        node._on_cancel(explode)
        node._on_cancel(lambda: done.append("whisper"))
        raise Cancelled()

    with pytest.raises(Cancelled):
        run()
    assert done == ["whisper", "allocator"], "one failure must not skip the rest"


def test_an_ordinary_failure_hands_the_card_back_too(monkeypatch):
    node = node_module()
    done: list[str] = []
    monkeypatch.setattr(asr_module, "release_cache", lambda: done.append("allocator"))

    @node._releases_the_card
    def run():
        node._on_cancel(lambda: done.append("whisper"))
        raise RuntimeError("fal refused the key")

    with pytest.raises(RuntimeError):
        run()
    assert done == ["whisper", "allocator"]


def test_registering_outside_a_run_is_harmless():
    """Nothing would ever call it, and a raise here would take down an unrelated import."""
    node_module()._on_cancel(lambda: None)


def test_the_node_still_looks_like_a_node_through_the_decorator():
    """ComfyUI passes every widget by keyword; the wrapper must not eat the signature."""
    import inspect

    node = node_module()
    parameters = inspect.signature(node.Music2PromptsLM.execute).parameters
    assert "project_name" in parameters and "lm_unload_after" in parameters
