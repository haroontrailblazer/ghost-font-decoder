---
name: ghost-decode
description: Decode videos that hide text in moving dots or noise using dense optical flow and optional OCR. Use for ghost-font clips, motion-defined text, random-dot kinematograms, TV-static videos with secret messages, text visible only during playback, or requests asking what a ghost-font video says.
---

# Ghost-Font Video Decoder

Recover motion-defined text with the decoder bundled at the plugin root.

## Required behavior

- Actually execute the Python decoder. Do not stop at explaining the algorithm, suggesting commands, or returning file paths.
- Use the bundled `decode.py` when present. If it is unavailable, write the complete program from `<plugin-root>/prompts/decode-in-chat.md` to a scratch `.py` file and execute it.
- Produce and inspect `revealed.png` and `revealed_heatmap.png`.
- A successful response must render `revealed.png` in the chat and state the recovered text explicitly. Paths alone are not a successful result.

## Workflow

1. Resolve the input video from the user's request. If no path is supplied, search the working directory for a recently modified `.mp4`, `.mov`, `.avi`, or `.webm` file. Ask only if more than one candidate is plausible.
2. Determine `<plugin-root>` from this skill's installed location. This file is at `<plugin-root>/codex-skills/ghost-decode/SKILL.md`; do not assume the plugin is installed in or invoked from the current working directory.
3. Check the runtime with:

   ```text
   python -c "import cv2, numpy"
   ```

   If imports fail, explain that the plugin needs the packages in `<plugin-root>/requirements.txt` and install them only with the user's approval when the environment requires approval.
4. Create a scratch output directory unless the user requested a specific destination, then run:

   ```text
   python "<plugin-root>/decode.py" "<video-path>" -o "<output-dir>"
   ```

5. Inspect both generated files. Prefer `revealed.png`; use `revealed_heatmap.png` to verify the decoder's conservative faint-glyph recovery when a lower-energy character would fall below the main Otsu threshold.
6. If the command prints `Text in the video:`, visually verify the OCR result against the revealed image. If OCR is unavailable or disagrees with the image, read the image directly.
7. Reply using the required chat format below. State uncertainty beside any ambiguous character.

## Required chat response

Every successful decode must include all of the following:

```markdown
Text in the video: **<RECOVERED TEXT>**

![Decoded text from the video](<ABSOLUTE-PATH-TO-REVEALED.PNG>)

Program run: `python "<plugin-root>/decode.py" "<video-path>" -o "<output-dir>"`

Outputs:
- Revealed image: `<absolute-path-to-revealed.png>`
- Heatmap: `<absolute-path-to-revealed_heatmap.png>`
```

Use the actual executed scratch-program path in `Program run` when the fallback is used. Ensure the Markdown image target is an absolute local path so Codex renders the image in chat. Do not say the decode succeeded if the program did not run or the revealed image was not inspected.

## No-install / chat fallback

If the bundled `decode.py` is not present — for example these steps are being
run in a chat without the plugin installed — reproduce the decoder yourself:
write the complete Python program from `prompts/decode-in-chat.md` (at the
plugin root) to a temporary file and run it on the video with the same steps.
Do not merely paste or describe the program without executing it. The algorithm
uses dense optical flow, median background subtraction, phase-correlation drift
registration, accumulation, a strong Otsu mask plus conservative faint-glyph
recovery, then OCR.
Never OCR a raw frame — every frame is noise.

## Troubleshooting

- Retry a weak or empty mask with `--method farneback`.
- Try `--stride 2` for high-frame-rate clips.
- Add `--max-frames 200` for long videos; a few seconds is usually enough.
- Continue with visual inspection when Tesseract is unavailable. Installing Tesseract is optional.
- Treat a uniformly dark heatmap as evidence that the clip may not contain coherent counter-motion.
