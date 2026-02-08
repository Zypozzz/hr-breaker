"""Monkey-patch pydantic-ai-litellm to support vision (BinaryContent).

pydantic-ai-litellm v0.2.3 stringifies BinaryContent instead of encoding
images as base64 data URIs. This patch fixes _map_messages to produce
OpenAI-compatible image_url parts that litellm forwards to any provider.
"""

import base64
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai._utils import guard_tool_call_id as _guard_tool_call_id
from pydantic_ai_litellm import LiteLLMModel

_ORIGINAL = LiteLLMModel._map_messages


def _convert_user_content(content) -> str | list[dict[str, Any]]:
    """Convert UserPromptPart.content to litellm format, handling vision."""
    if isinstance(content, str):
        return content

    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
        elif isinstance(item, BinaryContent) and item.is_image:
            b64 = base64.b64encode(item.data).decode("utf-8")
            data_uri = f"data:{item.media_type};base64,{b64}"
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        elif isinstance(item, ImageUrl):
            parts.append({"type": "image_url", "image_url": {"url": item.url}})
        elif isinstance(item, BinaryContent):
            # Non-image binary: fall back to text description
            parts.append({"type": "text", "text": f"[{item.media_type} binary content]"})
        else:
            parts.append({"type": "text", "text": str(item)})
    return parts


async def _patched_map_messages(
    self, messages: list[ModelMessage]
) -> list[dict[str, Any]]:
    """Patched _map_messages with proper BinaryContent/vision support."""
    litellm_messages: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    litellm_messages.append(
                        {"role": "system", "content": part.content}
                    )
                elif isinstance(part, UserPromptPart):
                    litellm_messages.append(
                        {"role": "user", "content": _convert_user_content(part.content)}
                    )
                elif isinstance(part, ToolReturnPart):
                    litellm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _guard_tool_call_id(t=part),
                            "content": part.model_response_str(),
                        }
                    )
                elif isinstance(part, RetryPromptPart):
                    if part.tool_name is None:
                        litellm_messages.append(
                            {"role": "user", "content": part.model_response()}
                        )
                    else:
                        litellm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": _guard_tool_call_id(t=part),
                                "content": part.model_response(),
                            }
                        )

        elif isinstance(message, ModelResponse):
            message_content = None
            tool_calls = []

            for part in message.parts:
                if isinstance(part, TextPart):
                    message_content = part.content
                elif isinstance(part, ToolCallPart):
                    tool_calls.append(
                        {
                            "id": _guard_tool_call_id(t=part),
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": part.args_as_json_str(),
                            },
                        }
                    )

            assistant_message: dict[str, Any] = {"role": "assistant"}
            if message_content:
                assistant_message["content"] = message_content
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            litellm_messages.append(assistant_message)

    return litellm_messages


def apply():
    """Apply the vision patch to LiteLLMModel."""
    LiteLLMModel._map_messages = _patched_map_messages
