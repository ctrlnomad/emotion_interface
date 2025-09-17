# Emotion Interface

This project captures live webcam video, detects facial landmarks with MediaPipe Face Mesh, and overlays the landmarks in real time.

## Prerequisites
- Python 3.9 or newer
- [uv](https://docs.astral.sh/uv/) package manager installed
- A webcam accessible from the machine running the script

## Setup
```bash
uv sync
```
This creates the virtual environment and installs dependencies listed in `pyproject.toml`.

## Run the demo
```bash
uv run python main.py
```
Controls inside the preview window:
- Press `l` to toggle landmark rendering
- Press `q` to quit the script

If the webcam cannot be opened (for example, when running in a sandboxed environment), the script prints a helpful error message and exits.

## Emotion classification
The demo includes a lightweight rule-based classifier targeting the emotions `confused`, `happy`, and `distressed`.
It analyses eyebrow height/asymmetry, eye openness, and mouth shape to infer the most likely state, then smooths predictions over the last few frames.
The active emotion, triggering rule, and the underlying feature values are displayed in the video overlay to help with manual calibration.

## Streaming thoughts overlay
- When the classifier detects a change in emotion, the app requests a matching “thought” from OpenAI and streams the text live into a floating box.
- Set the API key before running: `export OPENAI_API_KEY=sk-...`. Without a key, the app falls back to local canned copy.
- Thought bubbles avoid the detected face, connect to a random facial landmark with a line, colour-shift to match the active emotion, and remain on screen for ~2 seconds after completion.
- New thoughts spawn at irregular intervals whenever an emotion is active, so multiple bubbles can overlap while their lifetimes intersect.
- Styling (colours, line widths, fonts) and behaviour (hold duration, placement margins) live in `config.py` for easy iteration.
