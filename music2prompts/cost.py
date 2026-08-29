"""What one run costs, in USD - per model and in total.

A single run can spend money in four different places (the LLM that writes the prompts,
the image model, the video model, and every retry of any of them), and each provider
reports its charges differently. The point of this module is that the number the node
shows is never quietly wrong, so each charge is carried with a word saying how well we
know it:

``billed``
    The provider stated the charge for this specific call. OpenRouter puts ``usage.cost``
    on every reply - chat, image and video - and that is the amount charged to the account.
``computed``
    A live unit price times units we actually measured. fal.ai answers the result GET with
    an ``x-fal-billable-units`` header, and publishes its list prices at a keyless endpoint.
``estimated``
    Real token counts times a rate table that ships inside this file. OpenAI and Anthropic
    publish no pricing API, so this is the best anyone can do without scraping.
``unknown``
    No price could be resolved. Such a call adds **nothing** to the total and is counted
    separately instead, because inventing a number is the one outcome worse than admitting
    the gap.
``free``
    A known exact zero - LM Studio runs on your own machine.

Two rules follow from that and are enforced throughout:

* money is :class:`decimal.Decimal` from end to end. ``1000 * 0.00017`` is
  ``0.17000000000000284`` in binary floating point, and a run makes hundreds of charges
  that size. Rounding happens once, for display.
* nothing in here may cost a render. Every entry point is wrapped by the caller, the price
  lookup degrades to "unknown" on any failure and caches that failure so it is not retried
  once per shot, and a closed websocket is not an error.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, localcontext

from .util import PREFIX, warn

COST_EVENT = "music2prompts/cost"

#: Weakest last. A group reports the weakest word present in it, except that ``free`` is
#: exact so it never weakens anything, and ``unknown`` is reported by a separate counter.
_RANK = {"billed": 0, "computed": 1, "estimated": 2}

#: When the built-in rate tables below were read off the providers' own pricing pages.
PRICES_READ_ON = "2026-08-29"

#: Keyless, unauthenticated, one flat list of every endpoint fal bills for.
FAL_PRICES_URL = "https://rest.alpha.fal.ai/billing/prices"

#: Units whose meaning has been observed end to end: the multipliers (resolution tier,
#: megapixels, the per-model minimum duration) are folded into the unit count server-side,
#: so cost really is units x price. Endpoints billed in anything else are still priced the
#: same way, but reported as `estimated` rather than `computed` because the folding was
#: never confirmed for them.
VERIFIED_UNITS = frozenset(
    {"images", "megapixels", "processed megapixels", "seconds", "units", "compute seconds"}
)

# --------------------------------------------------------------------------- rate tables

#: USD per 1M tokens. OpenAI publishes Input / Cached input / Output for these four and no
#: cache-write column at all, so a cache write is priced at the ordinary input rate.
OPENAI_PRICES = {
    "gpt-5.2": {"in": "1.75", "cached_in": "0.175", "out": "14.00"},
    "gpt-5.1": {"in": "1.25", "cached_in": "0.125", "out": "10.00"},
    "gpt-5-mini": {"in": "0.25", "cached_in": "0.025", "out": "2.00"},
    "gpt-4.1": {"in": "2.00", "cached_in": "0.50", "out": "8.00"},
}

#: USD per 1M tokens: base input, 5-minute cache write, 1-hour cache write, cache hit, output.
ANTHROPIC_PRICES = {
    "claude-opus-5": {"in": "5", "w5m": "6.25", "w1h": "10", "read": "0.50", "out": "25"},
    "claude-opus-4-8": {"in": "5", "w5m": "6.25", "w1h": "10", "read": "0.50", "out": "25"},
    "claude-sonnet-5": {"in": "2", "w5m": "2.50", "w1h": "4", "read": "0.20", "out": "10"},
    "claude-haiku-4-5": {"in": "1", "w5m": "1.25", "w1h": "2", "read": "0.10", "out": "5"},
}

MILLION = Decimal(1000000)


# --------------------------------------------------------------------------- money

def dec(value) -> Decimal:
    """A Decimal that never imports binary error.

    ``Decimal(0.00017)`` is ``0.000170000000000000012212453270876721944...``; going through
    ``str`` first keeps the number the provider actually published.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    return Decimal(str(value))


def usd(value: Decimal | None) -> str:
    """One display rule that serves a $0.0034 run and a $47.20 run without lying.

    Two decimals would print $0.00017 as "$0.00"; six would print $47.20 as "$47.200000"
    and $0.15 as "$0.150000", claiming precision nobody has.
    """
    if value is None:
        return "—"
    amount = dec(value)
    if amount == 0:
        return "$0.00"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1:
        return f"{sign}${amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"
    # three significant digits, but never more than six decimal places
    exponent = amount.adjusted()  # -1 for 0.5, -4 for 0.00017
    places = min(6, 2 - exponent)
    shown = amount.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if shown == 0:
        return f"{sign}<$0.000001"
    text = format(shown, "f")
    if "." in text:
        text = text.rstrip("0")
        whole, _, fraction = text.partition(".")
        text = f"{whole}.{fraction.ljust(2, '0')}"
    return f"{sign}${text}"


def _weakest(words) -> str:
    """The provenance a group reports: the least certain of the ones that carry money."""
    ranked = [word for word in words if word in _RANK]
    if not ranked:
        return "free" if any(word == "free" for word in words) else "unknown"
    return max(ranked, key=lambda word: _RANK[word])


# --------------------------------------------------------------------------- fal prices

@dataclass(frozen=True)
class Price:
    """One endpoint's published list rate. ``usd`` is the price of a single ``unit``."""

    usd: Decimal
    unit: str
    source: str = "fal-catalogue"


_FAL_PRICES: dict[str, Price] | None = None
_FAL_LOCK = threading.Lock()


def fal_prices(timeout: float = 6.0) -> dict[str, Price]:
    """fal's public price list, fetched once per process.

    Deliberately shaped like :func:`render.fal_schema`: one best-effort GET, a failure is
    warned about exactly once and then cached as "no prices", and the render never sees an
    exception. An empty dict means every fal row will read ``unknown`` - which is the
    honest answer, and far better than pricing a render off a stale guess.
    """
    global _FAL_PRICES
    if _FAL_PRICES is not None:
        return _FAL_PRICES
    with _FAL_LOCK:
        if _FAL_PRICES is not None:
            return _FAL_PRICES
        prices: dict[str, Price] = {}
        try:
            import requests  # type: ignore

            response = requests.get(FAL_PRICES_URL, timeout=timeout)
            if response.status_code < 400:
                for row in response.json() or []:
                    endpoint = str(row.get("endpoint") or "").strip()
                    if not endpoint or row.get("price") is None:
                        continue
                    prices[endpoint] = Price(
                        usd=dec(row.get("price")),
                        unit=str(row.get("billable_unit") or "").strip() or "units",
                    )
            else:
                warn(f"fal price list returned HTTP {response.status_code}; fal costs will read as unknown")
        except Exception as exc:  # pragma: no cover - network shapes vary
            warn(f"could not read the fal price list ({exc}); fal costs will read as unknown")
        _FAL_PRICES = prices
        return _FAL_PRICES


def fal_usd(model: str, units) -> tuple[Decimal | None, Price | None, str]:
    """USD for one fal render: the units it billed times its published unit price."""
    price = fal_prices().get(model)
    if price is None or units is None:
        return None, price, "unknown"
    try:
        counted = dec(units)
    except Exception:
        return None, price, "unknown"
    with localcontext() as context:
        context.prec = 34
        amount = counted * price.usd
    known = price.unit.lower() in VERIFIED_UNITS
    return amount, price, "computed" if known else "estimated"


# --------------------------------------------------------------------------- LLM prices

def llm_price(provider: str, model: str) -> dict | None:
    table = OPENAI_PRICES if provider == "openai" else ANTHROPIC_PRICES if provider == "anthropic" else {}
    return table.get(model)


def openai_usd(model: str, usage: dict) -> tuple[Decimal | None, dict]:
    """OpenAI charges cached prompt tokens at a lower rate, and they are already inside
    ``prompt_tokens`` - so they are subtracted out, never added on top."""
    rates = llm_price("openai", model)
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    counted = {"input": prompt, "output": completion, "cache_read": cached, "cache_write": 0}
    if not rates:
        return None, counted
    fresh = max(prompt - cached, 0)
    with localcontext() as context:
        context.prec = 34
        amount = (
            dec(fresh) * dec(rates["in"])
            + dec(cached) * dec(rates["cached_in"])
            + dec(completion) * dec(rates["out"])
        ) / MILLION
    return amount, counted


def anthropic_usd(model: str, usage: dict) -> tuple[Decimal | None, dict]:
    """Anthropic reports cache tokens *outside* ``input_tokens``, so they are added on."""
    rates = llm_price("anthropic", model)
    usage = usage or {}
    inputs = int(usage.get("input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    split = usage.get("cache_creation") or {}
    write_5m = int(split.get("ephemeral_5m_input_tokens") or 0)
    write_1h = int(split.get("ephemeral_1h_input_tokens") or 0)
    if not (write_5m or write_1h):
        # no split reported: the node sends no cache_control, and 5m is the default TTL
        write_5m = written
    counted = {"input": inputs, "output": output, "cache_read": read, "cache_write": write_5m + write_1h}
    if not rates:
        return None, counted
    with localcontext() as context:
        context.prec = 34
        amount = (
            dec(inputs) * dec(rates["in"])
            + dec(read) * dec(rates["read"])
            + dec(write_5m) * dec(rates["w5m"])
            + dec(write_1h) * dec(rates["w1h"])
            + dec(output) * dec(rates["out"])
        ) / MILLION
        if str(usage.get("service_tier") or "").lower() == "batch":
            amount = amount / 2  # the batch tier is half price
    return amount, counted


# --------------------------------------------------------------------------- the ledger

@dataclass
class Charge:
    """One billable attempt. Append-only: a retry is its own Charge, never a mutation."""

    provider: str
    model: str
    kind: str  # "llm" | "image" | "video"
    label: str = ""
    attempt: int = 1
    outcome: str = "ok"  # "ok" | "failed" | "rejected"
    provenance: str = "unknown"
    usd: Decimal | None = None
    units: Decimal | None = None
    unit: str = ""
    unit_usd: Decimal | None = None
    tokens: dict | None = None
    note: str = ""
    #: A fal result-stage refusal means a runner already spent GPU time on it.
    possibly_billed: bool = False


@dataclass
class CostLedger:
    """Every charge one run made, and what it adds up to.

    Written from worker threads (fal and OpenRouter renders run concurrently) and read from
    the main thread, so the list is guarded. Publishing is best-effort: a closed socket must
    not be able to reach the render.
    """

    node_id: str = ""
    charges: list[Charge] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.node_id = str(self.node_id) if self.node_id is not None else ""

    # -------------------------------------------------------------- recording

    def add(self, charge: Charge) -> Charge:
        with self._lock:
            self.charges.append(charge)
        return charge

    def record_llm(self, provider: str, model: str, usage: dict, stage: str = "", attempt: int = 1) -> Charge:
        """One LLM reply. Called for every attempt, including ones whose JSON failed to
        parse afterwards - those tokens were generated, so they were charged."""
        usage = usage or {}
        if provider == "lmstudio":
            return self.add(
                Charge(provider, model, "llm", stage, attempt, provenance="free", usd=Decimal(0),
                       tokens={"input": int(usage.get("prompt_tokens") or 0),
                               "output": int(usage.get("completion_tokens") or 0),
                               "cache_read": 0, "cache_write": 0})
            )
        if provider == "openrouter":
            cost = usage.get("cost")
            byok = bool(usage.get("is_byok"))
            tokens = {
                "input": int(usage.get("prompt_tokens") or 0),
                "output": int(usage.get("completion_tokens") or 0),
                "cache_read": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "cache_write": 0,
            }
            if cost is None or byok:
                note = "billed on your own provider key (BYOK)" if byok else "no cost reported"
                return self.add(Charge(provider, model, "llm", stage, attempt, tokens=tokens, note=note))
            return self.add(
                Charge(provider, model, "llm", stage, attempt, provenance="billed", usd=dec(cost), tokens=tokens)
            )
        if provider == "openai":
            amount, tokens = openai_usd(model, usage)
        elif provider == "anthropic":
            amount, tokens = anthropic_usd(model, usage)
        else:
            amount, tokens = None, None
        if amount is None:
            note = f"'{model}' is not in the built-in rate table (read {PRICES_READ_ON})"
            return self.add(Charge(provider, model, "llm", stage, attempt, tokens=tokens, note=note))
        return self.add(
            Charge(provider, model, "llm", stage, attempt, provenance="estimated", usd=amount, tokens=tokens)
        )

    def record_fal(
        self, model: str, kind: str, units, label: str = "", attempt: int = 1,
        outcome: str = "ok", possibly_billed: bool = False,
    ) -> Charge:
        amount, price, provenance = fal_usd(model, units)
        note = ""
        if price is None:
            note = "endpoint is not in fal's price list"
        elif units is None:
            note = "fal sent no x-fal-billable-units header"
        return self.add(
            Charge(
                "fal", model, kind, label, attempt, outcome,
                provenance=provenance if amount is not None else "unknown",
                usd=amount,
                units=dec(units) if units is not None else None,
                unit=price.unit if price else "",
                unit_usd=price.usd if price else None,
                note=note,
                possibly_billed=possibly_billed,
            )
        )

    def record_openrouter_media(
        self, model: str, kind: str, usage: dict, label: str = "", attempt: int = 1, outcome: str = "ok",
    ) -> Charge:
        usage = usage or {}
        cost = usage.get("cost")
        if cost is None:
            return self.add(
                Charge("openrouter", model, kind, label, attempt, outcome, note="no cost reported")
            )
        return self.add(
            Charge("openrouter", model, kind, label, attempt, outcome, provenance="billed", usd=dec(cost))
        )

    # -------------------------------------------------------------- reporting

    def groups(self) -> list[dict]:
        """One row per (provider, model, kind), in the order each first appeared."""
        order: list[tuple] = []
        buckets: dict[tuple, dict] = {}
        with self._lock:
            charges = list(self.charges)
        for charge in charges:
            key = (charge.provider, charge.model, charge.kind)
            if key not in buckets:
                order.append(key)
                buckets[key] = {
                    "provider": charge.provider, "model": charge.model, "kind": charge.kind,
                    "calls": 0, "unpriced": 0, "usd": Decimal(0), "units": Decimal(0),
                    "unit": "", "unit_usd": None, "words": [], "failed_usd": Decimal(0),
                    "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                    "notes": [],
                }
            bucket = buckets[key]
            bucket["calls"] += 1
            bucket["words"].append(charge.provenance)
            if charge.unit and not bucket["unit"]:
                bucket["unit"], bucket["unit_usd"] = charge.unit, charge.unit_usd
            if charge.units is not None:
                bucket["units"] += charge.units
            if charge.tokens:
                for name, count in charge.tokens.items():
                    bucket["tokens"][name] = bucket["tokens"].get(name, 0) + int(count)
            if charge.note and charge.note not in bucket["notes"]:
                bucket["notes"].append(charge.note)
            if charge.usd is None:
                bucket["unpriced"] += 1
            elif charge.outcome == "ok":
                bucket["usd"] += charge.usd
            else:
                bucket["failed_usd"] += charge.usd
        rows = []
        for key in order:
            bucket = buckets[key]
            provenance = _weakest(bucket["words"])
            rows.append(
                {
                    "provider": bucket["provider"], "model": bucket["model"], "kind": bucket["kind"],
                    "calls": bucket["calls"], "unpriced": bucket["unpriced"],
                    "units": format(bucket["units"], "f") if bucket["units"] else None,
                    "unit": bucket["unit"],
                    "unit_usd": format(bucket["unit_usd"], "f") if bucket["unit_usd"] is not None else None,
                    "usd": format(bucket["usd"], "f"),
                    "display": usd(bucket["usd"]) if bucket["calls"] != bucket["unpriced"] else "—",
                    "provenance": provenance,
                    "tokens": bucket["tokens"] if any(bucket["tokens"].values()) else None,
                    "failed_usd": format(bucket["failed_usd"], "f") if bucket["failed_usd"] else None,
                    "note": "; ".join(bucket["notes"]),
                }
            )
        return rows

    def total(self) -> dict:
        """The authoritative number: summed from unrounded charges, never from the rows."""
        with self._lock:
            charges = list(self.charges)
        with localcontext() as context:
            context.prec = 34
            spent = Decimal(0)
            failed = Decimal(0)
            for charge in charges:
                if charge.usd is None:
                    continue
                if charge.outcome == "ok":
                    spent += charge.usd
                else:
                    failed += charge.usd
        priced = sum(1 for charge in charges if charge.usd is not None)
        unpriced = sum(1 for charge in charges if charge.usd is None)
        at_risk = sum(1 for charge in charges if charge.possibly_billed)
        return {
            "usd": format(spent, "f"),
            "nano_usd": int(spent * 1000000000),
            "display": usd(spent),
            "provenance": _weakest([charge.provenance for charge in charges]),
            "priced_calls": priced,
            "unpriced_calls": unpriced,
            "possibly_billed_calls": at_risk,
            "failed_usd": format(failed, "f"),
            "failed_display": usd(failed),
            "note": f"list prices, read {PRICES_READ_ON}",
        }

    def payload(self, final: bool = False) -> dict:
        return {
            "node": self.node_id,
            "final": bool(final),
            "currency": "USD",
            "prices_read_on": PRICES_READ_ON,
            "models": self.groups(),
            "total": self.total(),
        }

    def ui(self) -> dict:
        """Replayed into the panel when a finished job is reopened from the queue."""
        return {"m2p_cost": [self.payload(final=True)]} if self.charges else {}

    def publish(self, final: bool = False) -> bool:
        """Push the running total to the node. Never raises: no panel is worth a render."""
        if not self.node_id:
            return False
        try:
            from .preview import _send

            return _send(self.payload(final=final), COST_EVENT)
        except Exception as exc:  # pragma: no cover - defensive
            warn(f"could not send the cost event: {exc}")
            return False

    # -------------------------------------------------------------- on disk

    def report_json(self) -> str:
        import json

        with self._lock:
            charges = list(self.charges)
        document = self.payload(final=True)
        document["charges"] = [
            {
                "provider": charge.provider, "model": charge.model, "kind": charge.kind,
                "label": charge.label, "attempt": charge.attempt, "outcome": charge.outcome,
                "provenance": charge.provenance,
                "usd": format(charge.usd, "f") if charge.usd is not None else None,
                "units": format(charge.units, "f") if charge.units is not None else None,
                "unit": charge.unit,
                "unit_usd": format(charge.unit_usd, "f") if charge.unit_usd is not None else None,
                "tokens": charge.tokens, "note": charge.note,
                "possibly_billed": charge.possibly_billed,
            }
            for charge in charges
        ]
        document["assumptions"] = [
            "A fal submit refused with 400/422 is counted as costing nothing: the gateway "
            "rejects the schema before a runner starts. fal only guarantees $0 for 5xx.",
            "OpenAI and Anthropic publish no pricing API, so their rows are this file's "
            f"table (read {PRICES_READ_ON}) times the token counts they reported.",
            "Account-level discounts, batch and priority tiers and region multipliers are "
            "not visible to the node and are not applied.",
        ]
        return json.dumps(document, indent=2, ensure_ascii=False)

    def report_text(self) -> str:
        rows = self.groups()
        total = self.total()
        width = max([len(row["model"]) for row in rows] + [20])
        lines = [f"{PREFIX} cost report ({total['note']})", ""]
        for row in rows:
            line = f"  {row['model']:<{width}}  {row['calls']:>3} calls  {row['display']:>12}"
            if row["units"] and row["unit_usd"]:
                line += f"   [{row['units']} {row['unit']} x ${row['unit_usd']}]"
            if row["tokens"]:
                tokens = row["tokens"]
                line += f"   [in {tokens['input']} / out {tokens['output']} tokens]"
            if row["provenance"] != "billed":
                line += f"   ({row['provenance']})"
            if row["note"]:
                line += f"   - {row['note']}"
            lines.append(line)
        lines.append("")
        lines.append(f"  {'TOTAL':<{width}}  {total['priced_calls']:>3} calls  {total['display']:>12}")
        if total["unpriced_calls"]:
            lines.append(f"  {total['unpriced_calls']} call(s) could not be priced and are not in that number.")
        if total["possibly_billed_calls"]:
            lines.append(
                f"  {total['possibly_billed_calls']} call(s) were refused after the job had already run "
                "and may still have been charged."
            )
        if total["failed_usd"] and dec(total["failed_usd"]) > 0:
            lines.append(f"  {total['failed_display']} was spent on calls that did not return a result.")
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CostLedger {PREFIX} node={self.node_id} charges={len(self.charges)}>"
