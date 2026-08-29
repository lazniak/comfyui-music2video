"""Provider payload shapes - no network, no torch."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music2prompts.providers import (  # noqa: E402
    KEY_ENV,
    AnthropicClient,
    OpenAICompatClient,
    ProviderError,
    _split_data_uri,
    make_llm_client,
    resolve_key,
    strictify,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"action": {"type": "string"}, "camera": {"type": "string"}},
                "required": ["action"],
            },
        },
    },
    "required": ["title"],
}

PIXEL = "data:image/png;base64,iVBORw0KGgo="


def capture(client, reply):
    """Replace the HTTP layer and remember the payload the client would send."""
    seen: dict = {}

    def fake_post(path, payload):
        seen["path"] = path
        seen["payload"] = payload
        return reply

    client._post = fake_post  # type: ignore[assignment]
    return seen


# --------------------------------------------------------------------------- keys


def test_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert resolve_key("openai", "from-widget") == "from-widget"
    assert resolve_key("openai", "  ") == "from-env"


@pytest.mark.parametrize(
    "provider,variable",
    [
        ("openrouter", "OPENROUTER_API_KEY"),
        ("openrouter", "OPEN_ROUTER_API_KEY"),
        ("openrouter", "OPENROUTER_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("openai", "OPEN_AI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("anthropic", "ANTHROPIC_AUTH_TOKEN"),
        ("fal", "FAL_KEY"),
        ("fal", "FAL_API_KEY"),
        ("fal", "FAL_ADMIN_API_KEY"),
    ],
)
def test_every_accepted_environment_variable_is_read(monkeypatch, provider, variable):
    for name in KEY_ENV[provider]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "from-env")
    assert resolve_key(provider, "") == "from-env"


def test_the_first_variable_set_wins(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "primary")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "secondary")
    assert resolve_key("openrouter", "") == "primary"


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ProviderError, match="no API key"):
        make_llm_client("anthropic")


def test_unknown_provider_is_rejected():
    with pytest.raises(ProviderError, match="unknown LLM provider"):
        make_llm_client("gemini-in-my-basement")


# --------------------------------------------------------------------------- schema


def test_strictify_makes_every_object_strict():
    strict = strictify(SCHEMA)
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["title", "shots"]
    item = strict["properties"]["shots"]["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["action", "camera"]
    # the original is untouched
    assert SCHEMA["required"] == ["title"]


def test_split_data_uri():
    assert _split_data_uri("data:image/jpeg;base64,AAAA") == ("image/jpeg", "AAAA")
    with pytest.raises(ProviderError):
        _split_data_uri("https://example.com/a.png")


# --------------------------------------------------------------------------- OpenAI / OpenRouter


def openai_reply(text: str = '{"title": "ok"}'):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def test_openai_payload_uses_strict_json_schema():
    client = OpenAICompatClient("https://api.openai.com/v1", "k", 60, 0, False, name="openai")
    seen = capture(client, openai_reply())
    assert client.chat_json("gpt-5.2", "sys", "user", schema=SCHEMA, seed=7) == {"title": "ok"}
    payload = seen["payload"]
    assert seen["path"] == "/chat/completions"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert payload["seed"] == 7
    assert "reasoning_effort" not in payload, "'none' is not a valid OpenAI effort"


def test_openai_images_become_content_parts():
    client = OpenAICompatClient("https://api.openai.com/v1", "k", 60, 0, False, name="openai")
    seen = capture(client, openai_reply())
    client.chat_json("gpt-5.2", "sys", "describe", images=[PIXEL])
    parts = seen["payload"]["messages"][1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"] == PIXEL


def test_openrouter_sends_effort_when_asked():
    client = OpenAICompatClient("https://openrouter.ai/api/v1", "k", 60, 0, False, name="openrouter")
    seen = capture(client, openai_reply())
    client.chat_json("anthropic/claude-sonnet-5", "sys", "user", reasoning_effort="medium")
    assert seen["payload"]["reasoning_effort"] == "medium"
    assert client._headers()["X-Title"] == "ComfyUI Music2Prompts"


def test_empty_reply_raises():
    client = OpenAICompatClient("https://api.openai.com/v1", "k", 60, 0, False, name="openai")
    capture(client, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    with pytest.raises(ProviderError, match="token limit"):
        client.chat_json("gpt-5.2", "sys", "user")


def test_retry_drops_the_schema_on_the_last_attempt():
    client = OpenAICompatClient("https://api.openai.com/v1", "k", 60, 1, False, name="openai")
    calls: list[dict] = []

    def flaky(path, payload):
        calls.append(payload)
        if len(calls) == 1:
            raise ProviderError("boom")
        return openai_reply()

    client._post = flaky  # type: ignore[assignment]
    assert client.chat_json("gpt-5.2", "sys", "user", schema=SCHEMA) == {"title": "ok"}
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert "raw JSON only" in calls[1]["messages"][0]["content"]


# --------------------------------------------------------------------------- Anthropic


def test_anthropic_structured_stage_uses_forced_tool_use():
    client = AnthropicClient("https://api.anthropic.com/v1", "k", 60, 0, False)
    seen = capture(
        client,
        {"content": [{"type": "tool_use", "name": "result", "input": {"title": "ok"}}]},
    )
    assert client.chat_json("claude-opus-5", "sys", "user", schema=SCHEMA) == {"title": "ok"}
    payload = seen["payload"]
    assert seen["path"] == "/messages"
    assert payload["tool_choice"] == {"type": "tool", "name": "result"}
    assert payload["tools"][0]["input_schema"] == SCHEMA
    assert "temperature" not in payload, "current Claude models reject temperature"
    assert client._headers()["anthropic-version"] == "2023-06-01"


def test_anthropic_images_become_base64_blocks():
    client = AnthropicClient("https://api.anthropic.com/v1", "k", 60, 0, False)
    seen = capture(client, {"content": [{"type": "text", "text": '{"title": "ok"}'}]})
    assert client.chat_json("claude-opus-5", "sys", "look", images=[PIXEL]) == {"title": "ok"}
    blocks = seen["payload"]["messages"][0]["content"]
    assert blocks[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgo=",
    }
    assert blocks[-1]["type"] == "text"


def test_anthropic_refusal_is_reported():
    client = AnthropicClient("https://api.anthropic.com/v1", "k", 60, 0, False)
    capture(client, {"content": [], "stop_reason": "refusal", "stop_details": {"category": "cyber"}})
    with pytest.raises(ProviderError, match="declined"):
        client.chat_json("claude-opus-5", "sys", "user")


def test_cloud_clients_have_no_op_model_lifecycle():
    client = AnthropicClient("https://api.anthropic.com/v1", "k", 60, 0, False)
    assert client.ensure_model("claude-opus-5", auto_load=True) is None
    assert client.unload("claude-opus-5") is None
