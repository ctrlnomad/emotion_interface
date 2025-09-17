"""Drawing helpers for thought overlays."""

from __future__ import annotations

from typing import List, Tuple

import cv2

from config import DEFAULT_THOUGHT_STYLE, ThoughtStyle
from thought_engine import ThoughtRenderData


def draw_thought_overlay(
    frame,
    render: ThoughtRenderData,
    style: ThoughtStyle = DEFAULT_THOUGHT_STYLE,
) -> None:
    height, width, _ = frame.shape
    x, y = render.box_origin
    box_w, box_h = render.box_size

    # Ensure box stays within frame bounds.
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    if render.mode == "ascii":
        # Preserve spacing for ASCII art; trim only trailing newlines.
        text = render.text.rstrip("\n") if render.text else ""
        lines = text.splitlines() if text else ["…"]
    else:
        text = render.text.strip() if render.text else ""
        lines = _wrap_text(text or "…", style, style.text_max_width)
    text_block_size = _measure_text_block(lines, style)

    padding_x = 16
    padding_y = 18
    box_w = max(box_w, text_block_size[0] + padding_x * 2)
    box_h = max(box_h, text_block_size[1] + padding_y * 2)

    if x + box_w >= width:
        x = max(0, width - box_w - 1)
    if y + box_h >= height:
        y = max(0, height - box_h - 1)

    rect_top_left = (x, y)
    rect_bottom_right = (x + box_w, y + box_h)

    if style.box_fill_alpha > 0:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            rect_top_left,
            rect_bottom_right,
            style.box_fill_color,
            thickness=-1,
        )
        alpha = max(0.0, min(style.box_fill_alpha, 1.0))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    cv2.rectangle(
        frame,
        rect_top_left,
        rect_bottom_right,
        style.box_border_color,
        thickness=style.box_border_thickness,
        lineType=cv2.LINE_AA,
    )

    text_x = x + padding_x
    text_y = y + padding_y
    line_height = _line_height(style)

    for line in lines:
        cv2.putText(
            frame,
            line,
            (text_x, text_y),
            style.font,
            style.text_scale,
            style.text_color,
            style.text_thickness,
            lineType=cv2.LINE_AA,
        )
        text_y += line_height

    line_end = _closest_point_on_rect(
        render.line_start,
        rect_top_left,
        rect_bottom_right,
        style,
    )
    cv2.line(
        frame,
        render.line_start,
        line_end,
        style.line_color,
        thickness=style.line_thickness,
        lineType=cv2.LINE_AA,
    )


def _wrap_text(text: str, style: ThoughtStyle, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = words[0]

    for word in words[1:]:
        trial = current + " " + word
        width, _ = cv2.getTextSize(
            trial,
            style.font,
            style.text_scale,
            style.text_thickness,
        )[0]
        if width <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _measure_text_block(lines: List[str], style: ThoughtStyle) -> Tuple[int, int]:
    widths = []
    for line in lines:
        width, _ = cv2.getTextSize(
            line,
            style.font,
            style.text_scale,
            style.text_thickness,
        )[0]
        widths.append(width)
    total_height = len(lines) * _line_height(style)
    return (max(widths) if widths else 0, total_height)


def _line_height(style: ThoughtStyle) -> int:
    _, baseline = cv2.getTextSize(
        "Ag",
        style.font,
        style.text_scale,
        style.text_thickness,
    )
    return baseline + int(20 * style.text_scale)


def _closest_point_on_rect(
    point: Tuple[int, int],
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    style: ThoughtStyle,
) -> Tuple[int, int]:
    x, y = point
    x1, y1 = top_left
    x2, y2 = bottom_right

    candidates = [
        (x1, max(y1, min(y, y2))),
        (x2, max(y1, min(y, y2))),
        (max(x1, min(x, x2)), y1),
        (max(x1, min(x, x2)), y2),
    ]

    def dist_sq(p):
        dx = p[0] - x
        dy = p[1] - y
        return dx * dx + dy * dy

    best = min(candidates, key=dist_sq)

    # Slightly pull the line inward to avoid overshooting the rectangle border.
    inset = style.line_end_offset
    bx, by = best
    if best[0] == x1:
        bx += inset
    elif best[0] == x2:
        bx -= inset
    if best[1] == y1:
        by += inset
    elif best[1] == y2:
        by -= inset
    return (bx, by)


__all__ = ["draw_thought_overlay"]
