---
name: ghost-decode
description: Decode videos that hide text in moving dots or noise using dense optical flow and optional OCR. Use for ghost-font clips, motion-defined text, random-dot kinematograms, TV-static videos with secret messages, text visible only during playback, or requests asking what a ghost-font video says.
---

# Ghost-Font Video Decoder

Recover motion-defined text with the decoder bundled at the plugin root.

## Guardrails

- Run one primary decode. Allow at most one retry with the documented options
  when the first mask is weak or empty.
- Produce only `revealed.png` and `revealed_heatmap.png`.
- Never OCR a raw frame; the message exists only in accumulated motion.
- Inspect both output images before reporting text. Treat OCR as a hint.

## Workflow

1. Resolve the video from the request, or choose the most recently modified
   `.mp4`, `.mov`, `.avi`, or `.webm` in the working directory. Ask only when
   several candidates are plausible.
2. Determine `<plugin-root>` from this installed file:
   `<plugin-root>/skills/ghost-decode/SKILL.md`. Do not assume the plugin is in
   the working directory.
3. Select one Python interpreter and use that exact executable for checks,
   installation, and decoding:

   ```text
   <python> -c "import cv2, numpy"
   ```

   If imports fail, do not use an unrelated bare `pip`. Check
   `<python> -m pip --version`, then install with:

   ```text
   <python> -m pip install -r "<plugin-root>/requirements.txt"
   ```

   Ask before installation when the environment requires approval. If that
   interpreter has no pip, select another interpreter with pip or report the
   dependency blocker; do not claim the decoder ran.
4. Create a scratch output directory unless the user requested a destination,
   then run:

   ```text
   <python> "<plugin-root>/decode.py" "<video>" -o "<out-dir>"
   ```

5. Inspect `revealed.png`, then use the full-frame
   `revealed_heatmap.png` to check for a faint leading, trailing, or separate
   glyph. Trust visual inspection over the printed OCR hint.
6. If the first result is weak or empty, allow one retry:

   ```text
   <python> "<plugin-root>/decode.py" "<video>" -o "<out-dir>" --method farneback --stride 2
   ```

   Use `--max-frames 200` for a long clip. Do not create diagnostic images or
   invent another algorithm.

## Required response

Render both images with absolute paths, then state the recovered text:

```markdown
![revealed.png](<absolute-path-to-revealed.png>)

![revealed_heatmap.png](<absolute-path-to-revealed_heatmap.png>)

Text in the video: **<RECOVERED TEXT>**
```

Mark only genuinely ambiguous glyphs as `(unclear: X)`. If neither image
contains letter shapes, say no text was recovered and still show both images.
