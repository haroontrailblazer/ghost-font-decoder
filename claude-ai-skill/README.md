# Ghost Font Decoder — skill for claude.ai

A ready-to-upload [Agent Skill](https://support.anthropic.com/en/articles/skills)
for **claude.ai** (the web/desktop app). It decodes ghost-font videos — text
hidden in moving dots that is invisible in any paused frame — by running a small
optical-flow program in claude.ai's code sandbox.

## Download & install (2 minutes)

1. Download **[`ghost-decode.zip`](ghost-decode.zip)** from this folder
   (on GitHub: open the file and click **Download raw**).
2. In claude.ai, open **Settings → Capabilities → Skills** (Pro/Team/Enterprise;
   code execution must be enabled).
3. Click **Upload skill** and choose `ghost-decode.zip`.
4. Start a new chat, attach a ghost-font video, and ask:

   > What does this video say?

Claude loads the `ghost-decode` skill, runs the bundled decoder, views the
revealed image, and replies with `Text in the video: <decoded text>`.

## What's inside

- `ghost-decode/SKILL.md` — the skill instructions (also embeds the full decoder
  as a fallback, so it works even if the bundled file is not found).
- `ghost-decode/decode.py` — the decoder: dense optical flow → background
  subtraction → drift registration → accumulate → clean mask → optional OCR.

## Notes

- The sandbox installs `opencv-python-headless` and `numpy` automatically.
- Tesseract OCR usually isn't present in the sandbox; that's fine — Claude reads
  the revealed image directly, which is reliable.
- Same decoder ships as a `/ghost-decode` command + skill for **Claude Code** and
  a `$ghost-decode` skill for **Codex**. See the repository root README.

## Rebuilding the zip

If you edit the skill, regenerate the archive with forward-slash paths:

```bash
cd claude-ai-skill && zip -r ghost-decode.zip ghost-decode
```
