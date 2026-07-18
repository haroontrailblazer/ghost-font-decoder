---
name: ghost-decode
description: Use when a video hides text in moving dots or noise — "ghost font" clips, motion-defined text, random-dot kinematograms, TV-static videos with a secret message, text readable only while playing but invisible in any paused frame, or the user asks what a ghost-font video says.
argument-hint: "[video-path]"
---

# Ghost-Font Video Decoder

Ghost-font videos hide a message as a random-dot field: every frame is uniform noise, but dots inside the letter shapes move against the background dots. The bundled decoder recovers the message from dense optical flow.

## Required behavior

- **Actually execute the Python decoder.** Do not stop at explaining the algorithm, suggesting a command, or returning file paths — run it and read its output.
- Use the bundled `decode.py` when present. If it is unavailable, write the complete program from `${CLAUDE_PLUGIN_ROOT}/prompts/decode-in-chat.md` to a scratch `.py` file and execute that.
- Produce and inspect `revealed.png` (and `revealed_heatmap.png`).
- A successful reply must state the recovered text explicitly using the format below. Paths or the algorithm alone are not a successful result.

## Steps

1. Resolve the video path: use `$ARGUMENTS` or the path in the user's request. If none given, look for a recently modified video file (`.mp4`, `.mov`, `.avi`, `.webm`) in the working directory; ask only if more than one candidate is plausible.
2. Check dependencies: `python -c "import cv2, numpy"`. If that fails, run `pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"`.
3. Run the decoder (writes `revealed.png` and `revealed_heatmap.png` to the output dir, creating it if needed — use a scratch dir unless the user wants them in the project):

   ```
   python "${CLAUDE_PLUGIN_ROOT}/decode.py" "<video-path>" -o "<output-dir>"
   ```

   `${CLAUDE_PLUGIN_ROOT}` is the plugin's install directory, not the working directory. On Windows PowerShell it is `$env:CLAUDE_PLUGIN_ROOT` — if that expands empty, substitute the install path literally.
4. If the output contains `Text in the video:`, Read `revealed.png` with vision to confirm the OCR matches the image.
5. If OCR was skipped (no Tesseract engine) or disagrees with the image, read the recovered text yourself from `revealed.png`. The heatmap `revealed_heatmap.png` preserves faint letter detail the cleaned mask may lose.
6. Reply using the required response format below. Note any characters you are unsure about.

## Required response format

Every successful decode must include, at minimum, the recovered text on its own line exactly like this:

```markdown
Text in the video: **<RECOVERED TEXT>**
```

Then also give the user the absolute paths of `revealed.png` and `revealed_heatmap.png`, and the exact command you ran. Do not claim the decode succeeded if the program did not run or the revealed image was not inspected.

## No-install / chat fallback

If `${CLAUDE_PLUGIN_ROOT}/decode.py` is not present — for example this skill's
steps are being run in a chat without the plugin installed — reproduce the
decoder yourself: write the complete Python program from
`prompts/decode-in-chat.md` (bundled at the plugin root) to a temporary file and
**run it** on the video with the same steps. Do not merely paste or describe the
program without executing it. The algorithm is identical: dense optical flow,
median background subtraction, phase-correlation drift registration,
accumulation, Otsu threshold, then OCR. Never OCR a raw frame — every frame is noise.

## Troubleshooting

- Weak or empty mask: retry with `--method farneback`; for high-fps clips also try `--stride 2`.
- Very long video: add `--max-frames 200` — a few seconds of footage is enough.
- OCR engine missing: the skill still works via step 5 (read the image visually). Installing Tesseract (`winget install UB-Mannheim.TesseractOCR`) enables automatic text extraction.
- No text found at all: the clip may not be a ghost-font video — check `revealed_heatmap.png`; if it is uniformly dark there is no coherent counter-motion to decode.
