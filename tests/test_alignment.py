"""Shot N's image and shot N's video prompt have to be about shot N.

Both come from batched LLM calls that ask the model to echo each shot's number back. When
it echoes the wrong one - renumbering every batch from 1, or slipping by one - a shot is
given its neighbour's description. And because the shot text and the image prompt are two
separate calls, they slip differently: the start frame shows one thing, the video prompt
describes another, and the video model dissolves from the frame it was handed into the
scene it was told about.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.llm_stages import pair_with_shots  # noqa: E402
from music2prompts.shots import ShotSlot  # noqa: E402


def shots(*indices: int) -> list[ShotSlot]:
    return [ShotSlot(index=index, start=index * 6.0, end=(index + 1) * 6.0) for index in indices]


def wrote(*numbers: int) -> list[dict]:
    """What the model sent back: a shot number, and text that names the shot it is really for."""
    return [{"shot": number, "text": f"content {position}"} for position, number in enumerate(numbers, start=1)]


def paired(batch, items):
    return [(slot.index, (item or {}).get("text")) for slot, item in pair_with_shots(batch, items, "stage")]


# --------------------------------------------------------------------------- the good case


def test_the_numbers_are_used_when_they_are_the_ones_that_were_sent():
    batch = shots(5, 6, 7, 8)
    assert paired(batch, wrote(5, 6, 7, 8)) == [
        (5, "content 1"), (6, "content 2"), (7, "content 3"), (8, "content 4")
    ]


def test_an_answer_that_came_back_out_of_order_is_put_back_in_order():
    """The numbers are all there, so they are worth more than the order."""
    batch = shots(5, 6, 7, 8)
    items = [
        {"shot": 7, "text": "seven"},
        {"shot": 5, "text": "five"},
        {"shot": 8, "text": "eight"},
        {"shot": 6, "text": "six"},
    ]
    assert paired(batch, items) == [(5, "five"), (6, "six"), (7, "seven"), (8, "eight")]


# --------------------------------------------------------------------------- the reported one


def test_a_model_that_renumbers_every_batch_from_one_still_lands_on_the_right_shot():
    """The commonest slip with a small local model: batch 2 comes back as shots 1-4."""
    batch = shots(5, 6, 7, 8)
    assert paired(batch, wrote(1, 2, 3, 4)) == [
        (5, "content 1"), (6, "content 2"), (7, "content 3"), (8, "content 4")
    ]


def test_a_model_that_slips_by_one_no_longer_shifts_every_shot(caplog):
    """This is the one that silently misaligned everything: 4,5,6,7 for shots 5,6,7,8.

    Three of the four numbers exist in the batch, so the old lookup matched them - and
    handed each shot the text written for the shot after it.
    """
    import logging

    batch = shots(5, 6, 7, 8)
    with caplog.at_level(logging.WARNING):
        result = paired(batch, wrote(4, 5, 6, 7))
    assert result == [(5, "content 1"), (6, "content 2"), (7, "content 3"), (8, "content 4")]
    assert "going by the order" in caplog.text, "and it says so, because the model is misbehaving"


def test_a_number_repeated_twice_does_not_give_two_shots_the_same_text():
    batch = shots(1, 2, 3, 4)
    assert paired(batch, wrote(1, 1, 2, 3)) == [
        (1, "content 1"), (2, "content 2"), (3, "content 3"), (4, "content 4")
    ]


# --------------------------------------------------------------------------- short answers


def test_a_short_answer_falls_back_to_the_numbers_rather_than_guessing():
    """With a count that does not match, position means nothing - some shot was skipped."""
    batch = shots(1, 2, 3, 4)
    assert paired(batch, wrote(1, 3)) == [(1, "content 1"), (2, None), (3, "content 2"), (4, None)]


def test_a_stage_that_failed_outright_leaves_every_shot_empty():
    batch = shots(1, 2, 3)
    assert paired(batch, []) == [(1, None), (2, None), (3, None)]


def test_a_short_answer_is_reported(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        pair_with_shots(shots(1, 2, 3, 4), wrote(1, 2), "image prompts 2")
    assert "2 of the 4 shot(s)" in caplog.text


def test_an_answer_with_no_numbers_at_all_still_lands_in_order():
    """Nothing in the schema forces the field to be there, and small models drop it."""
    batch = shots(3, 4)
    items = [{"text": "first"}, {"text": "second"}]
    assert paired(batch, items) == [(3, "first"), (4, "second")]


# --------------------------------------------------------------------------- end to end


def test_the_two_stages_cannot_slip_against_each_other():
    """The whole point: shot N's image prompt and shot N's video prompt describe shot N.

    Here the content stage numbers correctly and the image stage slips by one - the exact
    combination that produced a start frame and a video prompt about different scenes.
    """
    batch = shots(5, 6, 7, 8)
    content = dict(paired(batch, wrote(5, 6, 7, 8)))
    images = dict(paired(batch, wrote(4, 5, 6, 7)))
    assert content == images
