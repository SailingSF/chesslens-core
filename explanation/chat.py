"""
ChatAgent — an agentic follow-up chat about a single chess position.

The position analysis + engine output are injected into the system prompt (see
explanation/prompts), so the model starts the conversation already knowing
everything the one-shot analysis produced. During a turn the model may call the
`analyze_position` tool to run Stockfish on any variation before answering.

The agentic loop is implemented once per provider (Anthropic Messages tool-use
and OpenAI Responses function-calling); both share one tool executor. Tool-call
round-trips live entirely inside a single request — the caller only ever sees
plain text turns, keeping the API stateless.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import chess

from analysis.context import ContextAssembler, format_candidates
from explanation.generator import ExplanationGenerator, get_explainer
from explanation.prompts import analyze_position_tool, format_position_context
from explanation.providers import LLMConfig
from explanation.retry import retry_overloaded

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5
MAX_TOKENS = 1024


@dataclass
class ChatResult:
    reply: str
    tool_calls: list[dict] = field(default_factory=list)


class ChatAgent:
    def __init__(
        self,
        engine,
        assembler: ContextAssembler,
        explainer: ExplanationGenerator,
    ):
        self._engine = engine
        self._assembler = assembler
        self._explainer = explainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        conversation: list[dict],
        *,
        system: str,
        llm_config: LLMConfig,
        engine_defaults: dict,
    ) -> ChatResult:
        """Run one chat turn. `conversation` is the visible thread of plain
        {role, content} messages ending with the user's new question."""
        if llm_config.is_openai:
            client = self._explainer.get_openai_client(llm_config.api_key)
            return await self._run_openai_loop(
                client, llm_config.model, system, conversation,
                llm_config.reasoning_effort, engine_defaults,
            )
        client = self._explainer.get_anthropic_client(llm_config.api_key)
        return await self._run_anthropic_loop(
            client, llm_config.model, system, conversation, engine_defaults,
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, args: dict, engine_defaults: dict) -> str:
        if name == "analyze_position":
            return await self._execute_analyze_position(args.get("fen", ""), engine_defaults)
        return f"Unknown tool: {name}"

    async def _execute_analyze_position(self, fen: str, engine_defaults: dict) -> str:
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            return f"Invalid FEN ({exc}). Provide a full, legal FEN string."

        result = await self._engine.analyze(fen, depth=20, **engine_defaults)
        context = self._assembler.assemble(board, result)
        candidates = format_candidates(board, result)
        return format_position_context(context, candidates, board)

    # ------------------------------------------------------------------
    # Anthropic loop (Messages tool-use)
    # ------------------------------------------------------------------

    async def _run_anthropic_loop(
        self, client, model, system, messages, engine_defaults,
    ) -> ChatResult:
        tool = analyze_position_tool()
        tools = [{
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }]
        convo = list(messages)
        tool_calls: list[dict] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            message = await retry_overloaded(
                lambda: client.messages.create(
                    model=model, max_tokens=MAX_TOKENS, system=system,
                    messages=convo, tools=tools,
                )
            )
            if message.stop_reason != "tool_use":
                return ChatResult(reply=_anthropic_text(message), tool_calls=tool_calls)

            convo.append({"role": "assistant", "content": message.content})
            results = []
            for block in message.content:
                if block.type == "tool_use":
                    out = await self._execute_tool(block.name, block.input, engine_defaults)
                    tool_calls.append({"tool": block.name, "fen": block.input.get("fen")})
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": out,
                    })
            convo.append({"role": "user", "content": results})

        # Tool budget exhausted — force a final text answer without tools.
        message = await retry_overloaded(
            lambda: client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=system, messages=convo,
            )
        )
        return ChatResult(reply=_anthropic_text(message), tool_calls=tool_calls)

    # ------------------------------------------------------------------
    # OpenAI loop (Responses function-calling)
    # ------------------------------------------------------------------

    async def _run_openai_loop(
        self, client, model, system, messages, reasoning, engine_defaults,
    ) -> ChatResult:
        tool = analyze_position_tool()
        tools = [{
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }]
        input_items: list[dict] = [{"role": "system", "content": system}]
        input_items += [{"role": m["role"], "content": m["content"]} for m in messages]
        tool_calls: list[dict] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            kwargs = dict(
                model=model, input=input_items, tools=tools,
                tool_choice="auto", max_output_tokens=MAX_TOKENS,
            )
            if reasoning:
                kwargs["reasoning"] = {"effort": reasoning}
            response = await client.responses.create(**kwargs)

            fn_calls = [i for i in response.output if getattr(i, "type", None) == "function_call"]
            if not fn_calls:
                return ChatResult(reply=response.output_text.strip(), tool_calls=tool_calls)

            # Echo the model's output items back, then append each tool result.
            for item in response.output:
                input_items.append(item.model_dump())
            for call in fn_calls:
                args = json.loads(call.arguments or "{}")
                out = await self._execute_tool(call.name, args, engine_defaults)
                tool_calls.append({"tool": call.name, "fen": args.get("fen")})
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": out,
                })

        # Tool budget exhausted — force a final text answer without tools.
        response = await client.responses.create(
            model=model, input=input_items, max_output_tokens=MAX_TOKENS,
        )
        return ChatResult(reply=response.output_text.strip(), tool_calls=tool_calls)


def _anthropic_text(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text").strip()


# Singleton factory
_chat_agent: Optional[ChatAgent] = None


def get_chat_agent() -> ChatAgent:
    global _chat_agent
    if _chat_agent is None:
        from chess_engine.service import get_engine_service
        _chat_agent = ChatAgent(
            get_engine_service(),
            ContextAssembler(),
            get_explainer(),
        )
    return _chat_agent
