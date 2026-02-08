"""Reproduce pydantic-ai-litellm vision bug end-to-end.

Sends an image to a model via pydantic-ai + LiteLLMModel and shows
the model never receives the actual image.

Usage:
  uv run python scripts/repro_vision_bug.py           # without patch (broken)
  uv run python scripts/repro_vision_bug.py --patch    # with patch (fixed)
"""

import asyncio
import os
import struct
import sys
import zlib

import litellm
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai_litellm import LiteLLMModel

load_dotenv()
litellm.suppress_debug_info = True

if "GEMINI_API_KEY" not in os.environ and os.environ.get("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

use_patch = "--patch" in sys.argv
if use_patch:
    from hr_breaker.litellm_patch import apply
    apply()


def make_png_with_text() -> bytes:
    """Generate a 200x50 white PNG with 'HELLO 42' written in black pixels.

    Hand-drawn blocky pixel letters so the model must actually see the image
    to read the text. No font rendering libraries needed.
    """
    w, h = 200, 50
    # White background
    pixels = [[255] * (w * 3) for _ in range(h)]

    # Draw "42" in big blocky letters starting at x=80, y=10, each char 20px wide
    # "4":
    for y in range(10, 35):
        # vertical left stroke (top half)
        if y < 25:
            for x in range(80, 84):
                pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]
        # horizontal middle bar
        if 24 <= y <= 27:
            for x in range(80, 100):
                pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]
        # vertical right stroke (full height)
        for x in range(96, 100):
            pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]

    # "2":
    for y in range(10, 35):
        # top horizontal bar
        if 10 <= y <= 13:
            for x in range(105, 125):
                pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]
        # right stroke top half
        if 10 <= y <= 22:
            for x in range(121, 125):
                pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]
        # middle horizontal bar
        if 21 <= y <= 24:
            for x in range(105, 125):
                pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]
        # left stroke bottom half
        if 22 <= y <= 34:
            for x in range(105, 109):
                pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]
        # bottom horizontal bar
        if 31 <= y <= 34:
            for x in range(105, 125):
                pixels[y][x * 3 : x * 3 + 3] = [0, 0, 0]

    # Build PNG
    raw = b""
    for row in pixels:
        raw += b"\x00" + bytes(row)

    def chunk(tag, data):
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


async def main():
    model_name = os.getenv("FLASH_MODEL", "openrouter/google/gemini-2.5-flash")
    model = LiteLLMModel(model_name=model_name)
    agent = Agent(
        model,
        output_type=str,
        system_prompt="Answer with ONLY the number you see. Nothing else.",
    )

    png = make_png_with_text()
    print(f"Image: {len(png)} bytes, white PNG with '42' drawn in black pixels")
    print(f"Model: {model_name}")
    print(f"Patch applied: {'YES' if use_patch else 'NO'}\n")

    result = await agent.run(
        [
            "What number is written in this image?",
            BinaryContent(data=png, media_type="image/png"),
        ]
    )
    print(f"Model response: {result.output}")
    print()
    if "42" in result.output:
        print("PASS — model saw the image (or got lucky)")
    else:
        print("FAIL — model did NOT see the image (BinaryContent was stringified)")


asyncio.run(main())
