# Ghost Font Decoder — Claude Code plugin

"Ghost font" videos hide text as a random-dot field: every single frame is
uniform noise, but the dots inside the letter shapes drift against the
background dots, so the message is only visible while the video plays.
Pause it — or feed frames to an AI one by one — and there is nothing to read.

The trick has one weakness: the message *is* the motion. This plugin gives
Claude Code a skill that reads the motion directly with dense optical flow
and tells you what the video says.

![Revealed message](examples/revealed.png)

## Install (Claude Code)

```
/plugin marketplace add USER/ghost-font-decoder
/plugin install ghost-font-decoder@ghost-font-tools
```

Then just ask, in any session:

> what does ghost-video.mp4 say?

or invoke it directly:

```
/ghost-decode path/to/video.mp4
```

Claude runs the bundled decoder, OCRs the revealed mask (or reads it with
vision if Tesseract isn't installed), and replies with the hidden text.

**Requirements:** Python 3 with `opencv-python` and `numpy`
(`pip install -r requirements.txt` — Claude offers to do this for you).
Optional: the Tesseract engine (`winget install UB-Mannheim.TesseractOCR`)
for automatic OCR.

## Standalone use (no Claude)

```
pip install -r requirements.txt
python decode.py examples/ghost-message.mp4
```

Outputs `revealed_heatmap.png` (raw opposition score), `revealed.png` (clean
binary mask), and prints the OCR'd hidden message. Options: `-o OUT_DIR`,
`--method {dis,farneback}`, `--stride N`, `--max-frames N`, `--no-ocr`.

## How it works

1. **Dense optical flow** between consecutive frames (OpenCV DIS, or
   Farnebäck via `--method farneback`).
2. **Background subtraction**: the per-pair median flow is the background
   motion (the background dominates the frame area). Each pixel is scored by
   how strongly its residual flow points *against* the background.
3. **Drift registration**: the glyph region itself drifts slowly even though
   the dots inside it stream fast, so each pair's score map is aligned to the
   first frame via phase correlation before accumulating — otherwise the
   letters smear.
4. **Accumulation + cleanup**: scores summed over all frame pairs, Otsu
   threshold, morphological cleanup, small-component removal.
5. **OCR** (optional): Tesseract reads the revealed mask.

## Demo

`examples/ghost-message.mp4` is a real ghost-font clip. OCR on any paused
frame returns gibberish; `python decode.py examples/ghost-message.mp4`
prints `HELLO HUMAN`.
