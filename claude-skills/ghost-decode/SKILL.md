---
name: ghost-decode
description: Use when a video hides text in moving dots or noise — "ghost font" clips, motion-defined text, random-dot kinematograms, TV-static videos with a secret message, text readable only while playing but invisible in any paused frame, or the user asks what a ghost-font video says.
argument-hint: "[video-path]"
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

1. Resolve the video from `$ARGUMENTS`, the request, or the most recently
   modified `.mp4`, `.mov`, `.avi`, or `.webm` in the working directory. Ask
   only when several candidates are plausible.
2. Resolve `${CLAUDE_PLUGIN_ROOT}` as the plugin install directory. On Windows
   PowerShell use `$env:CLAUDE_PLUGIN_ROOT`.
3. Use one Python interpreter consistently:

   ```text
   <python> -c "import cv2, numpy"
   ```

   If imports fail, do not use an unrelated bare `pip`. Check
   `<python> -m pip --version`, then install with:

   ```text
   <python> -m pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"
   ```

   Ask first when installation requires approval. If the selected interpreter
   has no pip, choose another interpreter with pip or report the blocker.
4. Run the bundled decoder:

   ```text
   <python> "${CLAUDE_PLUGIN_ROOT}/decode.py" "<video>" -o "<out-dir>"
   ```

   If the decoder is missing, report an incomplete plugin installation. Do not
   reconstruct a different decoder.
5. Inspect `revealed.png`, then use the full-frame
   `revealed_heatmap.png` to check faint leading, trailing, or separate glyphs.
   Trust visual inspection over the OCR hint.
6. If the first result is weak or empty, allow one retry:

   ```text
   <python> "${CLAUDE_PLUGIN_ROOT}/decode.py" "<video>" -o "<out-dir>" --method farneback --stride 2
   ```

   Add `--max-frames 200` for a long clip. Do not create diagnostic images or
   invent another algorithm.

## Required response

Show both images, then state the recovered text:

```markdown
![revealed.png](<absolute-path-to-revealed.png>)

![revealed_heatmap.png](<absolute-path-to-revealed_heatmap.png>)

Text in the video: **<RECOVERED TEXT>**
```

Mark only genuinely ambiguous glyphs as `(unclear: X)`. If neither image
contains letter shapes, say no text was recovered and still show both images.
