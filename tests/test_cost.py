"""What one run cost: the arithmetic, the provenance, and what reaches the browser."""

from __future__ import annotations

import json
import os
import sys
import types
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts import cost as cost_module  # noqa: E402
from music2prompts import render as render_module  # noqa: E402
from music2prompts.cost import COST_EVENT, Charge, CostLedger, Price, dec, usd  # noqa: E402


@pytest.fixture(autouse=True)
def prices(monkeypatch):
    """A stand-in price catalogue, so no test ever reaches fal.ai."""
    catalogue = {
        "fal-ai/nano-banana-pro/edit": Price(Decimal("0.15"), "images"),
        "minimax/h3/reference-to-video": Price(Decimal("0.05"), "seconds"),
        "fal-ai/odd-endpoint": Price(Decimal("0.01"), "1000 characters"),
    }
    monkeypatch.setattr(cost_module, "_FAL_PRICES", dict(catalogue))
    return catalogue


@pytest.fixture
def sent(monkeypatch):
    """Capture what the node would push over ComfyUI's websocket."""
    events: list[tuple[str, dict]] = []

    class FakeServer:
        instance = None

        def send_sync(self, event, data, sid=None):
            events.append((event, data))

    server = types.ModuleType("server")
    server.PromptServer = FakeServer
    FakeServer.instance = FakeServer()
    monkeypatch.setitem(sys.modules, "server", server)
    return events


class FakeResponse:
    def __init__(self, status_code: int, body, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body) if isinstance(body, dict) else str(body)

    def json(self):
        return self._body


# --------------------------------------------------------------------------- arithmetic


def test_money_is_added_up_in_decimal_so_a_thousand_small_charges_still_land_on_the_exact_total():
    ledger = CostLedger("1")
    for _ in range(1000):
        ledger.add(Charge("fal", "m", "video", provenance="computed", usd=dec("0.00017")))
    assert ledger.total()["usd"] == "0.17000"
    assert dec(ledger.total()["usd"]) == Decimal("0.17"), "float would give 0.17000000000000284"


def test_a_price_string_from_the_provider_is_parsed_as_decimal_and_never_through_float():
    assert dec("0.000000175") == Decimal("0.000000175")
    assert dec(0.00017) == Decimal("0.00017"), "str() first, so the binary error never enters"


def test_a_charge_smaller_than_a_cent_keeps_three_significant_digits_instead_of_showing_as_zero():
    assert usd(Decimal("0.00017")) == "$0.00017"
    assert usd(Decimal("0.0034")) == "$0.0034"
    assert usd(Decimal("0.0403125")) == "$0.0403"


def test_a_run_over_a_dollar_is_shown_to_the_cent_with_a_thousands_separator():
    assert usd(Decimal("3.14159")) == "$3.14"
    assert usd(Decimal("47.2")) == "$47.20"
    assert usd(Decimal("1234.567")) == "$1,234.57"


def test_a_charge_too_small_to_print_says_so_instead_of_printing_zero():
    assert usd(Decimal("0.0000004")) == "<$0.000001"
    assert usd(Decimal(0)) == "$0.00"
    assert usd(None) == "—"


def test_half_way_values_round_up_rather_than_to_the_nearest_even():
    assert usd(Decimal("1.005")) == "$1.01", "bankers' rounding would give $1.00"


def test_the_total_is_computed_from_the_unrounded_values_not_from_the_rounded_rows():
    ledger = CostLedger("1")
    ledger.add(Charge("openrouter", "a", "llm", provenance="billed", usd=dec("0.0063")))
    ledger.add(Charge("fal", "b", "video", provenance="computed", usd=dec("3.60")))
    ledger.add(Charge("fal", "c", "image", provenance="computed", usd=dec("0.00017")))
    assert ledger.total()["usd"] == "3.60647", "the exact sum, not the sum of the displayed rows"
    assert ledger.total()["display"] == "$3.61"


def test_every_money_field_is_serialised_as_a_string_and_the_payload_is_json_safe():
    ledger = CostLedger("1")
    ledger.record_fal("minimax/h3/reference-to-video", "video", "6", label="shot 1")
    payload = ledger.payload()
    text = json.dumps(payload)  # a Decimal in here would raise
    assert '"usd": "0.30"' in text
    assert isinstance(payload["total"]["nano_usd"], int)


# --------------------------------------------------------------------------- fal


def test_the_billable_units_header_is_read_off_the_result_and_multiplied_by_the_catalogue_price(monkeypatch):
    seen: list[str] = []

    def post(url, headers=None, data=None, timeout=None):
        return FakeResponse(200, {"status_url": "s", "response_url": "r"})

    def get(url, headers=None, timeout=None):
        seen.append(url)
        if url == "s":
            return FakeResponse(200, {"status": "COMPLETED", "metrics": {"inference_time": 447.171}})
        return FakeResponse(200, {"video": {"url": "u"}}, headers={"x-fal-billable-units": "6"})

    module = types.ModuleType("requests")
    module.post, module.get = post, get
    monkeypatch.setitem(sys.modules, "requests", module)
    monkeypatch.setattr(render_module, "_download", lambda url, headers=None, timeout=300: b"mp4")

    ledger = CostLedger("1")
    client = render_module.FalClient("key", ledger=ledger)
    client.run("minimax/h3/reference-to-video", {}, label="shot 1", kind="video")
    charge = ledger.charges[0]
    assert charge.usd == Decimal("0.30"), "6 s at $0.05/s"
    assert charge.provenance == "computed"
    assert charge.unit == "seconds"


def test_the_inference_time_from_the_status_body_is_never_used_as_a_cost(monkeypatch):
    """447 s of runner wall clock against 6 billed units - they are different numbers."""
    def post(url, headers=None, data=None, timeout=None):
        return FakeResponse(200, {"status_url": "s", "response_url": "r"})

    def get(url, headers=None, timeout=None):
        if url == "s":
            return FakeResponse(200, {"status": "COMPLETED", "metrics": {"inference_time": 447.171}})
        return FakeResponse(200, {"video": {"url": "u"}})  # no header at all

    module = types.ModuleType("requests")
    module.post, module.get = post, get
    monkeypatch.setitem(sys.modules, "requests", module)
    monkeypatch.setattr(render_module, "_download", lambda url, headers=None, timeout=300: b"mp4")

    ledger = CostLedger("1")
    render_module.FalClient("key", ledger=ledger).run("minimax/h3/reference-to-video", {}, kind="video")
    assert ledger.charges[0].usd is None
    assert ledger.charges[0].provenance == "unknown"


def test_an_endpoint_missing_from_the_price_catalogue_is_unknown_rather_than_free():
    ledger = CostLedger("1")
    charge = ledger.record_fal("fal-ai/something-new", "image", "1")
    assert charge.usd is None and charge.provenance == "unknown"
    assert "price list" in charge.note
    assert ledger.total()["usd"] == "0", "an unpriced call adds nothing"
    assert ledger.total()["unpriced_calls"] == 1


def test_a_missing_billable_units_header_is_unknown_rather_than_zero():
    ledger = CostLedger("1")
    charge = ledger.record_fal("fal-ai/nano-banana-pro/edit", "image", None)
    assert charge.usd is None
    assert "x-fal-billable-units" in charge.note


def test_a_unit_whose_folding_was_never_observed_is_estimated_rather_than_computed():
    ledger = CostLedger("1")
    assert ledger.record_fal("fal-ai/odd-endpoint", "image", "3").provenance == "estimated"
    assert ledger.record_fal("fal-ai/nano-banana-pro/edit", "image", "3").provenance == "computed"


def test_a_price_lookup_that_fails_costs_the_number_and_not_the_render(monkeypatch):
    monkeypatch.setattr(cost_module, "_FAL_PRICES", None)
    module = types.ModuleType("requests")
    module.get = lambda url, timeout=None: (_ for _ in ()).throw(OSError("no network"))
    monkeypatch.setitem(sys.modules, "requests", module)
    assert cost_module.fal_prices() == {}
    assert cost_module._FAL_PRICES == {}, "cached, so it is not retried once per shot"


def test_a_submit_stage_refusal_is_recorded_as_an_attempt_that_cost_nothing(monkeypatch):
    module = types.ModuleType("requests")
    module.post = lambda url, headers=None, data=None, timeout=None: FakeResponse(422, {"detail": []})
    module.get = lambda *a, **k: FakeResponse(200, {})
    monkeypatch.setitem(sys.modules, "requests", module)

    ledger = CostLedger("1")
    client = render_module.FalClient("key", ledger=ledger)
    with pytest.raises(render_module.RenderError):
        client.run("fal-ai/nano-banana-pro/edit", {}, kind="image")
    assert ledger.charges, "a refused submit is still an attempt worth showing"
    assert all(charge.outcome == "rejected" for charge in ledger.charges)
    assert not any(charge.possibly_billed for charge in ledger.charges)


def test_a_result_stage_refusal_is_recorded_as_possibly_billed_because_the_job_already_ran(monkeypatch):
    def get(url, headers=None, timeout=None):
        if url == "s":
            return FakeResponse(200, {"status": "COMPLETED"})
        return FakeResponse(422, {"detail": []})

    module = types.ModuleType("requests")
    module.post = lambda url, headers=None, data=None, timeout=None: FakeResponse(
        200, {"status_url": "s", "response_url": "r"}
    )
    module.get = get
    monkeypatch.setitem(sys.modules, "requests", module)

    ledger = CostLedger("1")
    with pytest.raises(render_module.RenderError):
        render_module.FalClient("key", ledger=ledger).run("fal-ai/nano-banana-pro/edit", {}, kind="image")
    assert any(charge.possibly_billed for charge in ledger.charges)
    assert ledger.total()["possibly_billed_calls"] >= 1


def test_the_price_table_is_primed_on_the_calling_thread_before_any_worker_starts(monkeypatch):
    primed: list[bool] = []
    monkeypatch.setattr(cost_module, "fal_prices", lambda *a, **k: primed.append(True) or {})
    monkeypatch.setattr(render_module, "resolve_key", lambda provider, key="": "k")
    render_module.make_media_client("fal", "k")
    assert primed, "a worker thread must never be the one making that HTTP call"


# --------------------------------------------------------------------------- OpenRouter


def test_openrouter_reports_its_own_cost_and_the_node_records_it_verbatim_as_billed():
    ledger = CostLedger("1")
    charge = ledger.record_llm("openrouter", "google/gemini-3.7-flash", {"cost": 0.0063})
    assert charge.provenance == "billed" and charge.usd == Decimal("0.0063")


def test_a_null_cost_from_openrouter_is_unknown_rather_than_zero():
    ledger = CostLedger("1")
    assert ledger.record_llm("openrouter", "m", {"cost": None}).usd is None
    assert ledger.record_openrouter_media("m", "image", {}).usd is None


def test_a_call_billed_on_your_own_provider_key_is_not_counted_as_the_real_spend():
    ledger = CostLedger("1")
    charge = ledger.record_llm("openrouter", "m", {"cost": 0.0001, "is_byok": True})
    assert charge.usd is None and "BYOK" in charge.note


def test_the_openrouter_image_cost_is_captured_even_though_the_response_carries_no_id(monkeypatch):
    module = types.ModuleType("requests")
    module.post = lambda url, headers=None, data=None, timeout=None: FakeResponse(
        200, {"data": [{"b64_json": "QUJD"}], "usage": {"cost": 0.04}}
    )
    monkeypatch.setitem(sys.modules, "requests", module)
    monkeypatch.setattr(render_module, "resolve_key", lambda provider, key="": "k")

    ledger = CostLedger("1")
    client = render_module.OpenRouterMediaClient("k", ledger=ledger)
    client.image("google/gemini-3.1-flash-image", render_module.ImageRequest(prompt="p"))
    assert ledger.charges[0].usd == Decimal("0.04")
    assert ledger.charges[0].provenance == "billed"


def test_the_openrouter_video_cost_is_read_only_once_the_job_reports_completed(monkeypatch):
    bodies = iter(
        [
            {"status": "processing", "usage": {"cost": 999}},
            {"status": "completed", "usage": {"cost": 0.42}, "unsigned_urls": ["u"]},
        ]
    )
    module = types.ModuleType("requests")
    module.post = lambda url, headers=None, data=None, timeout=None: FakeResponse(
        200, {"polling_url": "p", "id": "x"}
    )
    module.get = lambda url, headers=None, timeout=None: FakeResponse(200, next(bodies))
    monkeypatch.setitem(sys.modules, "requests", module)
    monkeypatch.setattr(render_module, "resolve_key", lambda provider, key="": "k")
    monkeypatch.setattr(render_module, "_download", lambda url, headers=None, timeout=300: b"mp4")

    ledger = CostLedger("1")
    client = render_module.OpenRouterMediaClient("k", poll_seconds=0, ledger=ledger)
    client.video("some/model", render_module.VideoRequest(prompt="p", seconds=6))
    assert [charge.usd for charge in ledger.charges] == [Decimal("0.42")], "the 999 was never final"


# --------------------------------------------------------------------------- token pricing


def test_openai_cached_tokens_are_subtracted_from_the_prompt_tokens_not_added_to_them():
    amount, tokens = cost_module.openai_usd(
        "gpt-5.1", {"prompt_tokens": 1000, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 400}}
    )
    # 600 fresh at $1.25/M + 400 cached at $0.125/M
    assert amount == Decimal("0.00080")
    assert tokens["input"] == 1000 and tokens["cache_read"] == 400


def test_anthropic_cache_tokens_are_added_to_the_input_tokens_not_subtracted():
    amount, _ = cost_module.anthropic_usd(
        "claude-sonnet-5",
        {"input_tokens": 1000, "output_tokens": 0, "cache_read_input_tokens": 1000},
    )
    # 1000 input at $2/M plus 1000 cache hits at $0.20/M - not 0 input
    assert amount == Decimal("0.0022")


def test_anthropic_five_minute_and_one_hour_cache_writes_are_priced_apart():
    amount, _ = cost_module.anthropic_usd(
        "claude-sonnet-5",
        {
            "input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 2000,
            "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 1000},
        },
    )
    assert amount == Decimal("0.0065"), "1000 at $2.50/M plus 1000 at $4/M"


def test_the_batch_tier_is_half_price():
    full, _ = cost_module.anthropic_usd("claude-sonnet-5", {"input_tokens": 1000000, "output_tokens": 0})
    half, _ = cost_module.anthropic_usd(
        "claude-sonnet-5", {"input_tokens": 1000000, "output_tokens": 0, "service_tier": "batch"}
    )
    assert full == Decimal(2) and half == Decimal(1)


def test_a_model_outside_the_built_in_table_reports_its_tokens_and_says_the_price_is_unknown():
    ledger = CostLedger("1")
    charge = ledger.record_llm("openai", "gpt-9-turbo", {"prompt_tokens": 10, "completion_tokens": 5})
    assert charge.usd is None
    assert "built-in rate table" in charge.note
    assert charge.tokens == {"input": 10, "output": 5, "cache_read": 0, "cache_write": 0}


def test_lm_studio_is_reported_as_free_rather_than_unknown():
    ledger = CostLedger("1")
    charge = ledger.record_llm("lmstudio", "gemma", {"prompt_tokens": 900})
    assert charge.provenance == "free" and charge.usd == Decimal(0)
    assert ledger.total()["unpriced_calls"] == 0


def test_every_attempt_is_recorded_so_a_retry_after_a_billed_reply_is_not_lost():
    ledger = CostLedger("1")
    for attempt in (1, 2, 3):
        ledger.record_llm("openrouter", "m", {"cost": 0.01}, "shot content", attempt)
    assert ledger.total()["usd"] == "0.03"
    assert ledger.groups()[0]["calls"] == 3


def test_a_reply_that_parses_badly_still_leaves_its_tokens_in_the_ledger(monkeypatch):
    """_post records before anything downstream can reject the body."""
    from music2prompts import providers as providers_module

    module = types.ModuleType("requests")
    module.post = lambda url, headers=None, data=None, timeout=None: FakeResponse(
        200, {"choices": [], "usage": {"cost": 0.02}}
    )
    module.exceptions = types.SimpleNamespace(
        Timeout=TimeoutError, RequestException=OSError, ConnectionError=ConnectionError
    )
    monkeypatch.setitem(sys.modules, "requests", module)

    ledger = CostLedger("1")
    client = providers_module.OpenAICompatClient(
        "https://x", "key", 30, 0, False, ledger=ledger, name="openrouter"
    )
    with pytest.raises(providers_module.ProviderError):
        client.chat_json("m", "sys", "user")
    assert ledger.total()["usd"] == "0.02", "the tokens were generated, so they were charged"


# --------------------------------------------------------------------------- ledger


def test_the_weakest_provenance_in_a_group_is_the_one_the_group_reports():
    ledger = CostLedger("1")
    ledger.add(Charge("openai", "m", "llm", provenance="billed", usd=dec("1")))
    ledger.add(Charge("openai", "m", "llm", provenance="estimated", usd=dec("1")))
    assert ledger.groups()[0]["provenance"] == "estimated"


def test_free_calls_never_weaken_a_group():
    ledger = CostLedger("1")
    ledger.add(Charge("lmstudio", "m", "llm", provenance="free", usd=dec(0)))
    ledger.add(Charge("lmstudio", "m", "llm", provenance="billed", usd=dec("1")))
    assert ledger.groups()[0]["provenance"] == "billed"


def test_one_unpriced_call_does_not_relabel_an_otherwise_billed_total():
    ledger = CostLedger("1")
    ledger.add(Charge("openrouter", "a", "llm", provenance="billed", usd=dec("1")))
    ledger.add(Charge("fal", "b", "image", provenance="unknown"))
    total = ledger.total()
    assert total["provenance"] == "billed"
    assert total["unpriced_calls"] == 1, "reported next to the number, not folded into it"


def test_money_spent_on_calls_that_failed_is_reported_on_its_own_line():
    ledger = CostLedger("1")
    ledger.add(Charge("fal", "a", "video", provenance="computed", usd=dec("2"), outcome="failed"))
    ledger.add(Charge("fal", "a", "video", provenance="computed", usd=dec("3")))
    total = ledger.total()
    assert total["usd"] == "3" and total["failed_usd"] == "2"


def test_every_worker_thread_lands_in_the_ledger():
    from concurrent.futures import ThreadPoolExecutor

    ledger = CostLedger("1")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: ledger.record_fal("fal-ai/nano-banana-pro/edit", "image", "1"), range(64)))
    assert len(ledger.charges) == 64
    assert ledger.total()["usd"] == "9.60"


# --------------------------------------------------------------------------- transport


def test_the_cost_event_carries_the_node_id_the_gallery_is_addressed_by(sent):
    ledger = CostLedger(42)
    ledger.record_llm("openrouter", "m", {"cost": 0.01})
    assert ledger.publish() is True
    event, payload = sent[-1]
    assert event == COST_EVENT, "its own type: the preview listener drops anything without a filename"
    assert payload["node"] == "42"
    assert payload["final"] is False


def test_the_final_event_says_so(sent):
    ledger = CostLedger(7)
    ledger.record_llm("openrouter", "m", {"cost": 0.01})
    ledger.publish(final=True)
    assert sent[-1][1]["final"] is True


def test_the_replayed_ui_payload_holds_the_same_total_as_the_live_events(sent):
    ledger = CostLedger(7)
    ledger.record_fal("minimax/h3/reference-to-video", "video", "6")
    ledger.publish()
    live = sent[-1][1]["total"]["usd"]
    assert ledger.ui()["m2p_cost"][0]["total"]["usd"] == live


def test_the_cost_survives_a_run_that_wrote_no_previews_at_all():
    """`ui=feed.ui() or None` used to drop everything when nothing was previewed."""
    from music2prompts.preview import PreviewFeed

    feed = PreviewFeed(None)
    ledger = CostLedger(7)
    ledger.record_llm("openrouter", "m", {"cost": 0.01})
    replay = feed.ui()
    replay.update(ledger.ui())
    assert feed.ui() == {}
    assert replay and "m2p_cost" in replay


def test_a_run_that_spent_nothing_publishes_nothing_to_replay():
    assert CostLedger(7).ui() == {}


def test_a_closed_socket_costs_the_cost_panel_and_not_the_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "server", types.ModuleType("server"))
    ledger = CostLedger(7)
    ledger.record_llm("openrouter", "m", {"cost": 0.01})
    assert ledger.publish() is False


def test_a_node_outside_a_real_queue_run_still_accumulates(sent):
    ledger = CostLedger(None)
    ledger.record_llm("openrouter", "m", {"cost": 0.01})
    assert ledger.publish() is False and not sent
    assert ledger.total()["usd"] == "0.01"


# --------------------------------------------------------------------------- on disk


def test_the_written_report_names_every_call_and_the_assumptions_behind_the_number():
    ledger = CostLedger("1")
    ledger.record_fal("minimax/h3/reference-to-video", "video", "6", label="shot 1")
    ledger.record_llm("openai", "gpt-5.1", {"prompt_tokens": 1000, "completion_tokens": 100})
    document = json.loads(ledger.report_json())
    assert len(document["charges"]) == 2
    assert document["charges"][0]["label"] == "shot 1"
    assert any("no pricing API" in line for line in document["assumptions"])
    text = ledger.report_text()
    assert "minimax/h3/reference-to-video" in text and "TOTAL" in text
