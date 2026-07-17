# Ghost Font Decoder — Codex and Claude Code plugin

"Ghost font" videos hide text in a random-dot field. A paused frame looks like
uniform noise, but dots inside the letter shapes move against the background,
making the message visible only during playback.

This repository packages a `ghost-decode` skill for Codex and Claude Code. Its
Python decoder reads the motion with dense optical flow, compensates for glyph
drift, produces a clean reveal, and optionally runs OCR.

Each tool gets a skill written for its own conventions: Claude Code loads
`claude-skills/ghost-decode/`, Codex loads `codex-skills/ghost-decode/`.
Neither tool sees the other's skill.

![Revealed message](examples/revealed.png)

## Use with Codex

The repository is a Codex plugin root:

- `.codex-plugin/plugin.json` contains the Codex plugin manifest and UI metadata,
  pointing at `codex-skills/`.
- `codex-skills/ghost-decode/SKILL.md` contains the skill workflow.
- `codex-skills/ghost-decode/agents/openai.yaml` contains the skill's Codex UI metadata.

Add this repository as a local plugin source in a Codex marketplace, install
`ghost-font-decoder`, and start a new task so Codex loads the skill. Then ask:

> Use $ghost-decode to tell me what ghost-video.mp4 says.

Codex runs the bundled decoder, checks the revealed mask, and reports the hidden
text.

## Use with Claude Code

Claude Code loads its own skill from `claude-skills/` via `.claude-plugin`:

```text
/plugin marketplace add haroontrailblazer/ghost-font-decoder
/plugin install ghost-font-decoder@ghost-font-tools
```

Then ask:

> What does ghost-video.mp4 say?

or invoke the skill directly:

```text
/ghost-decode path/to/video.mp4
```

## Requirements

Install Python 3 dependencies with:

```text
pip install -r requirements.txt
```

Tesseract is optional. On Windows it can be installed with:

```text
winget install UB-Mannheim.TesseractOCR
```

Without Tesseract, the plugin still produces images that Codex or Claude can
inspect visually.

## Standalone use

```text
python decode.py examples/ghost-message.mp4
```

The command writes:

- `revealed_heatmap.png` — raw opposition score
- `revealed.png` — cleaned binary mask

Useful options:

```text
python decode.py VIDEO -o OUT_DIR --method farneback --stride 2 --max-frames 200 --no-ocr
```

## How it works

1. Compute dense optical flow between consecutive frames.
2. Subtract median background flow and score counter-moving pixels.
3. Register the drifting glyph region with phase correlation.
4. Accumulate scores, threshold them, and clean the mask.
5. Run optional Tesseract OCR and verify the result against the reveal.

The included example decodes to `HELLO HUMAN`.
