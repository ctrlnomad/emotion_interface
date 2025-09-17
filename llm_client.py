"""Utilities for streaming thoughts from an OpenAI model."""

from __future__ import annotations

import os
import time
from typing import Callable, Iterable, Optional

from config import LLMConfig


class LLMClient:
    """Thin wrapper around OpenAI's streaming chat completions."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._api_key = os.getenv(config.api_key_env)
        self._client = None
        self._enabled = False
        self._init_error: Optional[str] = None

        if self._api_key:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency guard
                self._init_error = (
                    "openai package is not installed. Install it to enable live thoughts."
                )
                return

            self._client = OpenAI(api_key=self._api_key)
            self._enabled = True
        else:
            self._init_error = (
                f"{config.api_key_env} not set. Thought generation will use local stubs."
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def _fallback_stream(self, emotion: str, mode: str) -> Iterable[str]:
        if mode == "ascii":
            ascii_templates = {
                "happy": (
                    "  _  _\n"
                    " ( \\o/ )\n"
                    "  / _ \\\n"
                    "  ^   ^"
                ),
                "neutral": (
                    " .----.\n"
                    "( 0  0 )\n"
                    "  -- --\n"
                    "  \\__/"
                ),
                "distressed": (
                    "! ! !\n"
                    "(>_<)\n"
                    " /|\\\n"
                    " / \\"
                ),
            }
            text = ascii_templates.get(
                emotion.lower(),
                "[ * ]\n  |  \n / \\",
            )
            yield text
            return

        templates = {
            "happy": "Can't help grinning; the world feels electric right now.",
            "distressed": "Mind racing, chest tight—just want the ground to steady.",
            "confused": "Thoughts loop like static, trying to make sense of all this.",
            "neutral": "Breathing in, breathing out, suspended in the in-between.",
        }
        text = templates.get(
            emotion,
            f"Emotion {emotion} flickers through me like a passing signal.",
        )
        for chunk in text.split():
            yield chunk + " "

    def stream_thought(
        self,
        emotion: str,
        on_chunk: Callable[[str], None],
        mode: str = "thought",
    ) -> None:
        """Stream a thought or ASCII art based on ``mode`` for ``emotion``."""

        if not self._enabled:
            for chunk in self._fallback_stream(emotion, mode):
                on_chunk(chunk)
                time.sleep(0.02)
            return

        assert self._client is not None  # for type checkers

        if mode == "ascii":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You respond with compact ASCII art that conveys the given emotion. "
                        "Keep width under 22 characters, height under 10 lines, and avoid explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Emotion: {emotion}. Render a quick ASCII art impression; respond with art only."
                    ).format(emotion=emotion),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": self._config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Emotion: {emotion}. Craft a matching fleeting thought bubble, "
                        "no introductions, just the inner voice."
                    ).format(emotion=emotion),
                },
            ]

        response_stream = self._client.chat.completions.create(  # type: ignore
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            messages=messages,
            stream=True,
        )

        for part in response_stream:
            choices = getattr(part, "choices", [])
            if not choices:
                continue
            delta = choices[0].delta
            if not delta:
                continue
            text = getattr(delta, "content", None)
            if text:
                on_chunk(text)


__all__ = ["LLMClient"]
