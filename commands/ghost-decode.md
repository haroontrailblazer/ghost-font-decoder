---
description: Decode a ghost-font video and reveal the hidden text
argument-hint: [video-path]
---

Immediately decode a ghost-font video (text hidden in moving dots / noise) by
running the bundled decoder. Do the work now — do NOT ask the user to set up a
folder, install the plugin differently, or point at files beyond the video path.

Target video: $ARGUMENTS

If `$ARGUMENTS` is empty, use the most recently modified `.mp4`, `.mov`, `.avi`,
or `.webm` file in the current directory. Only ask the user if there is more
than one plausible candidate.

Then, without further prompting:

1. Select one Python interpreter and check dependencies with
   `<python> -c "import cv2, numpy"`. If it fails, use the same interpreter:
   `<python> -m pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"`.
   Do not use an unrelated bare `pip`.
2. Run the decoder (it writes `revealed.png` and `revealed_heatmap.png`):

   ```
   <python> "${CLAUDE_PLUGIN_ROOT}/decode.py" "<video-path>" -o "<scratch-out-dir>"
   ```

   `${CLAUDE_PLUGIN_ROOT}` is the plugin's install directory. On Windows
   PowerShell it is `$env:CLAUDE_PLUGIN_ROOT`; if it expands empty, substitute
   the install path literally.
3. Read `revealed.png` with vision to confirm the text. Inspect the full-frame
   `revealed_heatmap.png` for faint leading, trailing, or separate glyphs.
4. Reply with the recovered text on its own line, plus the output image paths:

   ```
   Text in the video: **<DECODED TEXT>**
   ```

Never OCR a raw frame — every frame is uniform noise; the message is only in the
motion. Do not claim success unless the decoder actually ran and you inspected
`revealed.png`.
