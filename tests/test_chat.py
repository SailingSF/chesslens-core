"""
Unit tests for the agentic position chat (explanation/chat.py).

No Stockfish binary or API key needed: the LLM clients are faked and the
Stockfish tool executor is stubbed. These tests exercise the agentic loop
wiring — tool_use -> tool_result -> final answer — for both providers.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from explanation.chat import ChatAgent
from explanation.providers import LLMConfig


def _make_agent(fake_client, *, openai=False):
    if openai:
        explainer = SimpleNamespace(get_openai_client=lambda key: fake_client)
    else:
        explainer = SimpleNamespace(get_anthropic_client=lambda key: fake_client)
    agent = ChatAgent(engine=None, assembler=None, explainer=explainer)
    agent._execute_tool = AsyncMock(return_value="ENGINE OUTPUT")
    return agent


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

class _FakeAnthropicMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeAnthropic:
    def __init__(self, responses):
        self.messages = _FakeAnthropicMessages(responses)


def test_anthropic_loop_executes_tool_then_answers():
    tool_block = SimpleNamespace(
        type="tool_use", name="analyze_position",
        input={"fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}, id="tu1",
    )
    text_block = SimpleNamespace(type="text", text="The position is winning.")
    client = _FakeAnthropic([
        SimpleNamespace(stop_reason="tool_use", content=[tool_block]),
        SimpleNamespace(stop_reason="end_turn", content=[text_block]),
    ])
    agent = _make_agent(client)
    cfg = LLMConfig(model="claude-sonnet-4-6", api_key=None)

    result = asyncio.run(agent.run(
        [{"role": "user", "content": "Is this winning?"}],
        system="sys", llm_config=cfg, engine_defaults={},
    ))

    assert result.reply == "The position is winning."
    assert result.tool_calls == [
        {"tool": "analyze_position", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}
    ]
    agent._execute_tool.assert_awaited_once()
    # Two API calls: the tool-use turn and the final answer turn.
    assert len(client.messages.calls) == 2
    # Tools are advertised on the calls.
    assert client.messages.calls[0]["tools"][0]["name"] == "analyze_position"


def test_anthropic_turn_truncated_during_thinking_is_not_blank():
    """A reasoning model can spend the whole token budget thinking and emit no
    text block. The reply must never come back blank — the client stores it in
    the thread and posts it back, where a blank message is rejected."""
    thinking_block = SimpleNamespace(type="thinking", thinking="Let me work this out...")
    client = _FakeAnthropic([
        SimpleNamespace(stop_reason="max_tokens", content=[thinking_block]),
    ])
    agent = _make_agent(client)
    cfg = LLMConfig(model="claude-sonnet-4-6", api_key=None)

    result = asyncio.run(agent.run(
        [{"role": "user", "content": "Is this winning?"}],
        system="sys", llm_config=cfg, engine_defaults={},
    ))

    assert result.reply.strip()


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

class _FakeOpenAIResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeOpenAI:
    def __init__(self, responses):
        self.responses = _FakeOpenAIResponses(responses)


def test_openai_loop_executes_tool_then_answers():
    fn_call = SimpleNamespace(
        type="function_call", name="analyze_position",
        arguments='{"fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}', call_id="c1",
        model_dump=lambda: {"type": "function_call", "call_id": "c1"},
    )
    client = _FakeOpenAI([
        SimpleNamespace(output=[fn_call], output_text=""),
        SimpleNamespace(output=[], output_text="The position is winning."),
    ])
    agent = _make_agent(client, openai=True)
    cfg = LLMConfig(model="gpt-4.1", api_key=None)

    result = asyncio.run(agent.run(
        [{"role": "user", "content": "Is this winning?"}],
        system="sys", llm_config=cfg, engine_defaults={},
    ))

    assert result.reply == "The position is winning."
    assert result.tool_calls == [
        {"tool": "analyze_position", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}
    ]
    agent._execute_tool.assert_awaited_once()
    # The tool result is fed back as a function_call_output item.
    second_input = client.responses.calls[1]["input"]
    assert any(i.get("type") == "function_call_output" for i in second_input)


# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------

def test_unknown_tool_returns_message():
    agent = ChatAgent(engine=None, assembler=None, explainer=None)
    out = asyncio.run(agent._execute_tool("nope", {}, {}))
    assert "Unknown tool" in out


def test_analyze_position_rejects_invalid_fen():
    agent = ChatAgent(engine=None, assembler=None, explainer=None)
    out = asyncio.run(agent._execute_analyze_position("not-a-fen", {}))
    assert "Invalid FEN" in out
