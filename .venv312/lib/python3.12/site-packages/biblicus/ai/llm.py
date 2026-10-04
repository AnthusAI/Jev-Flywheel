"""
Provider-backed chat completions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .models import LlmClientConfig


@dataclass
class ChatCompletionResult:
    """
    Normalized response from a chat completion call.

    :param text: Generated assistant text.
    :type text: str
    :param tool_calls: Structured tool calls from the provider.
    :type tool_calls: list[dict[str, Any]]
    """

    text: str
    tool_calls: list[dict[str, Any]]


def _require_dspy():
    try:
        import dspy
    except ImportError as import_error:
        raise ValueError(
            "DSPy backend requires an optional dependency. "
            'Install it with pip install "biblicus[dspy]".'
        ) from import_error
    if not hasattr(dspy, "LM"):
        raise ValueError(
            "DSPy backend requires an optional dependency with LM support. "
            'Install it with pip install "biblicus[dspy]".'
        )
    return dspy


def _normalize_tool_calls(tool_calls: Sequence[object]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            function = tool_call.get("function") or {}
            normalized.append(
                {
                    "id": str(tool_call.get("id") or ""),
                    "type": str(tool_call.get("type") or "function"),
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": str(function.get("arguments") or ""),
                    },
                }
            )
            continue
        function = getattr(tool_call, "function", None)
        normalized.append(
            {
                "id": str(getattr(tool_call, "id", "") or ""),
                "type": str(getattr(tool_call, "type", "function") or "function"),
                "function": {
                    "name": str(getattr(function, "name", "") or ""),
                    "arguments": str(getattr(function, "arguments", "") or ""),
                },
            }
        )
    return normalized


def chat_completion(
    *,
    client: LlmClientConfig,
    messages: Sequence[dict[str, Any]],
    tools: Optional[Sequence[dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> ChatCompletionResult:
    """
    Execute a chat completion using DSPy (LiteLLM-backed).

    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param messages: Chat messages payload.
    :type messages: Sequence[dict[str, Any]]
    :param tools: Optional tool definitions to pass through.
    :type tools: Sequence[dict[str, Any]] or None
    :param tool_choice: Optional tool choice directive.
    :type tool_choice: str or None
    :return: Normalized completion result.
    :rtype: ChatCompletionResult
    :raises ValueError: If required dependencies or credentials are missing.
    """
    dspy = _require_dspy()
    lm = dspy.LM(client.litellm_model(), **client.build_litellm_kwargs())
    request_kwargs: dict[str, Any] = {}
    if tools:
        request_kwargs["tools"] = list(tools)
    if tool_choice:
        request_kwargs["tool_choice"] = tool_choice
    if client.response_format and not tools:
        request_kwargs["response_format"] = {"type": client.response_format}

    response = lm(messages=list(messages), **request_kwargs)
    item = response[0] if isinstance(response, list) and response else response
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("content") or "")
        tool_calls = _normalize_tool_calls(item.get("tool_calls") or [])
        return ChatCompletionResult(text=text, tool_calls=tool_calls)
    return ChatCompletionResult(text=str(item or ""), tool_calls=[])


def generate_completion(
    *,
    client: LlmClientConfig,
    system_prompt: Optional[str],
    user_prompt: str,
) -> str:
    """
    Generate a completion using the configured provider.

    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param system_prompt: Optional system prompt content.
    :type system_prompt: str or None
    :param user_prompt: User prompt content.
    :type user_prompt: str
    :return: Generated completion text.
    :rtype: str
    :raises ValueError: If required dependencies or credentials are missing.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return chat_completion(client=client, messages=messages).text
