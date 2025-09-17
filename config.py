"""Project-wide configuration structures and defaults."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import cv2


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 120
    system_prompt: str = (
        "You craft short, vivid internal monologues for an interactive art installation. "
        "Stay under three sentences, keep it punchy, and reflect the supplied emotion."
    )
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout: float = 30.0


@dataclass
class ThoughtStyle:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.6
    text_thickness: int = 1
    text_color: Tuple[int, int, int] = (255, 255, 255)
    line_color: Tuple[int, int, int] = (255, 255, 255)
    line_thickness: int = 2
    box_border_color: Tuple[int, int, int] = (255, 255, 255)
    box_border_thickness: int = 2
    box_fill_color: Tuple[int, int, int] = (20, 20, 20)
    box_fill_alpha: float = 0.55  # 0..1 range for blending
    text_max_width: int = 240
    line_end_offset: int = 12  # padding between line end and box


@dataclass
class ThoughtConfig:
    hold_duration: float = 2.0
    face_padding_ratio: float = 0.12
    frame_margin: int = 24
    default_box_size: Tuple[int, int] = (260, 140)
    max_box_attempts: int = 30
    min_landmark_visibility: float = 0.3
    spawn_interval_range: Tuple[float, float] = (0.4, 0.8) # (1.8, 4.2)
    max_active_thoughts: int = 3


DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_THOUGHT_STYLE = ThoughtStyle()
DEFAULT_THOUGHT_CONFIG = ThoughtConfig()


EMOTION_STYLE_OVERRIDES: Dict[str, Dict[str, Tuple[int, int, int] | int | float]] = {
    "happy": {
        "box_fill_color": (40, 200, 255),
        "box_border_color": (60, 220, 255),
        "line_color": (80, 240, 255),
        "text_color": (20, 40, 60),
    },
    "distressed": {
        "box_fill_color": (40, 60, 200),
        "box_border_color": (40, 40, 255),
        "line_color": (60, 60, 255),
        "text_color": (245, 245, 245),
    },
    "confused": {
        "box_fill_color": (180, 120, 255),
        "box_border_color": (200, 140, 255),
        "line_color": (205, 160, 255),
        "text_color": (15, 15, 25),
    },
}


def style_for_emotion(emotion: str) -> ThoughtStyle:
    overrides = EMOTION_STYLE_OVERRIDES.get(emotion.lower())
    if not overrides:
        return replace(DEFAULT_THOUGHT_STYLE)
    return replace(DEFAULT_THOUGHT_STYLE, **overrides)


__all__ = [
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_THOUGHT_CONFIG",
    "DEFAULT_THOUGHT_STYLE",
    "EMOTION_STYLE_OVERRIDES",
    "LLMConfig",
    "ThoughtConfig",
    "ThoughtStyle",
    "style_for_emotion",
]
