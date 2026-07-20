# Ghost Font Decoder — skill for claude.ai

A ready-to-upload [Agent Skill](https://support.anthropic.com/en/articles/skills)
for **claude.ai** (the web/desktop app). It decodes ghost-font videos — text
hidden in moving dots that is invisible in any paused frame — by running a small
optical-flow program in claude.ai's code sandbox.

## Build the upload archive

The zip is **not** committed to the repo on purpose — a zip inside the repo is a
nested zip inside the plugin package, and claude.ai rejects that on install
(`Nested zip files are not allowed`). Build it locally in one step:

```bash
python claude-ai-skill/build-zip.py
```

This writes `claude-ai-skill/ghost-decode.zip` containing only
`ghost-decode/SKILL.md` (the decoder is embedded in it).

## Install in claude.ai (2 minutes)

1. In claude.ai, open **Settings → Capabilities → Skills** (Pro/Team/Enterprise;
   code execution must be enabled).
2. Click **Upload skill** and choose the single **`ghost-decode.zip`** you built.
   Upload only that file — do **not** upload a zip of the whole repo or of the
   `claude-ai-skill/` folder, or you'll hit the nested-zip error.
3. Start a new chat, attach a ghost-font video, and ask:

   > What does this video say?

Claude loads the `ghost-decode` skill, runs the bundled decoder, views the two
revealed images, and replies with `Text in the video: <decoded text>`.

## What's inside

- `ghost-decode/SKILL.md` — self-contained: the instructions **and** the decoder
  (dense optical flow → background subtraction → drift registration → accumulate
  → clean mask + colormapped heatmap, cropped tightly to the text). Claude writes
  the decoder to a file, runs it once, and reads the word from the two images. No
  separate files needed.

## Notes

- The sandbox installs `opencv-python-headless` and `numpy` automatically.
- The skill is deliberately minimal: run once, produce exactly two images
  (`revealed.png` + `revealed_heatmap.png`), read the word, report
  `Text in the video: <text>`. It explicitly tells Claude **not** to re-decode,
  try other methods, or explain — the decoder already crops the output to the
  text so every glyph (even a small `I`) is clearly visible.
- Same decoder ships as a `/ghost-decode` command + skill for **Claude Code** and
  a `$ghost-decode` skill for **Codex**. See the repository root README.

## Rebuilding the zip

If you edit the skill, regenerate the archive with forward-slash paths:

```bash
cd claude-ai-skill && zip -r ghost-decode.zip ghost-decode
```
