"""Tests for litellm vision patch."""

import base64

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from hr_breaker.litellm_patch import _convert_user_content, _patched_map_messages


class TestConvertUserContent:
    def test_plain_string(self):
        assert _convert_user_content("hello") == "hello"

    def test_string_items_become_text_parts(self):
        result = _convert_user_content(["hello", "world"])
        assert result == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]

    def test_binary_image_becomes_base64_image_url(self):
        png_bytes = b"\x89PNG fake image data"
        content = BinaryContent(data=png_bytes, media_type="image/png")
        result = _convert_user_content([content])

        assert len(result) == 1
        part = result[0]
        assert part["type"] == "image_url"
        url = part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        assert decoded == png_bytes

    def test_mixed_text_and_image(self):
        png_bytes = b"img"
        result = _convert_user_content([
            "Describe this image:",
            BinaryContent(data=png_bytes, media_type="image/png"),
        ])
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Describe this image:"}
        assert result[1]["type"] == "image_url"

    def test_image_url_object(self):
        result = _convert_user_content([
            ImageUrl(url="https://example.com/img.png"),
        ])
        assert result == [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]

    def test_non_image_binary_falls_back_to_text(self):
        result = _convert_user_content([
            BinaryContent(data=b"audio data", media_type="audio/mp3"),
        ])
        assert result == [
            {"type": "text", "text": "[audio/mp3 binary content]"},
        ]


class TestPatchedMapMessages:
    @pytest.mark.asyncio
    async def test_system_and_user_text(self):
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="You are helpful"),
                UserPromptPart(content="Hello"),
            ]),
        ]
        # Call as unbound method (self=None since we don't need instance state)
        result = await _patched_map_messages(None, messages)
        assert result == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

    @pytest.mark.asyncio
    async def test_user_with_image(self):
        png_bytes = b"fake png"
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content=[
                    "Check this resume:",
                    BinaryContent(data=png_bytes, media_type="image/png"),
                ]),
            ]),
        ]
        result = await _patched_map_messages(None, messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "Check this resume:"}
        assert msg["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_model_response_with_text(self):
        messages = [
            ModelResponse(parts=[TextPart(content="Sure, here's the answer")]),
        ]
        result = await _patched_map_messages(None, messages)
        assert result == [
            {"role": "assistant", "content": "Sure, here's the answer"},
        ]

    @pytest.mark.asyncio
    async def test_model_response_with_tool_call(self):
        messages = [
            ModelResponse(parts=[
                ToolCallPart(
                    tool_name="check_length",
                    args='{"html": "<p>test</p>"}',
                    tool_call_id="call_1",
                ),
            ]),
        ]
        result = await _patched_map_messages(None, messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "check_length"
