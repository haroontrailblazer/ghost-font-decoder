---
name: ghost-decode
description: Use when a video hides text in moving dots or noise — "ghost font" clips, motion-defined text, random-dot kinematograms, TV-static videos with a secret message, text readable only while playing but invisible in any paused frame, or the user asks what a ghost-font video says.
---

# Ghost-Font Video Decoder

Ghost-font videos hide a message as a random-dot field: every frame is uniform noise, but dots inside the letter shapes move against the background dots, so the message is only visible in motion. Frame-by-frame reading and OCR see nothing. This skill recovers the message from the motion itself, using dense optical flow.

## Required behavior

- **Actually run the decoder in the code sandbox.** Do not stop at explaining the algorithm or describing the command — execute it and inspect the output.
- The decoder `decode.py` is bundled with this skill (in the same folder as this file). If you cannot locate it, recreate it from the **Bundled decoder** code block at the end of this file and run that.
- Produce `revealed.png` (and `revealed_heatmap.png`), then **view `revealed.png`** to read the message.
- A successful reply must state the recovered text on its own line: `Text in the video: **<decoded text>**`. Paths or the algorithm alone are not a success.

## Steps

1. Find the video the user attached to the conversation (a `.mp4`, `.mov`, `.avi`, or `.webm` file in the working directory). If several are plausible, ask which one.
2. Install dependencies in the sandbox (quietly), then confirm the import:

   ```bash
   pip install --quiet opencv-python-headless numpy pytesseract
   python -c "import cv2, numpy"
   ```

   `pytesseract` is optional — OCR only works if the `tesseract` binary is present. It usually is not in this sandbox, and that is fine: you will read `revealed.png` yourself.
3. Locate the bundled `decode.py` (it sits next to this `SKILL.md`). Run it, writing outputs to a scratch folder:

   ```bash
   python decode.py "<video-path>" -o out
   ```

4. If the program printed `Text in the video:`, note that value.
5. **Open and view `out/revealed.png`** and read the message directly from the image — this is the reliable path in this sandbox. Use `out/revealed_heatmap.png` if faint letters are unclear.
6. Reply with the recovered text on its own line, and attach/show `revealed.png`:

   ```markdown
   Text in the video: **<DECODED TEXT>**
   ```

   Note any characters you are unsure about.

## Troubleshooting

- Weak or empty mask: rerun with `--method farneback`; for high-frame-rate clips add `--stride 2`.
- Very long video: add `--max-frames 200` — a few seconds is enough.
- Uniformly dark `revealed_heatmap.png` means there is no coherent counter-motion — the clip is probably not a ghost-font video.

## Bundled decoder

The bundled `decode.py` is the authoritative program — prefer running it. If it
is missing from the sandbox, write this equivalent to `decode.py` and run it. It
is the same algorithm: dense optical flow, median background subtraction,
phase-correlation drift registration, accumulation, Otsu threshold, optional OCR.
Never OCR a raw frame — every frame is noise.

```python
import sys, os, shutil
import cv2, numpy as np

def iter_gray(path, stride=1, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {path}")
    i = y = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % stride == 0:
            yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            y += 1
            if max_frames and y >= max_frames: break
        i += 1
    cap.release()

def accumulate(frames):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    score = prev = prev_smooth = None
    drift = np.zeros(2); pairs = 0
    for gray in frames:
        if prev is not None:
            flow = dis.calc(prev, gray, None)
            bg = np.median(flow.reshape(-1, 2), axis=0)
            residual = flow - bg
            mag = float(np.hypot(*bg))
            ps = (residual @ (-bg / mag)) if mag > 0.15 else np.hypot(residual[...,0], residual[...,1])
            ps = np.clip(ps, 0, None).astype(np.float32)
            smooth = cv2.GaussianBlur(ps, (31, 31), 0)
            if prev_smooth is not None:
                (dx, dy), r = cv2.phaseCorrelate(prev_smooth, smooth)
                if r > 0.05 and np.hypot(dx, dy) < 30:
                    drift += (dx, dy)
            prev_smooth = smooth
            h, w = ps.shape
            M = np.float32([[1, 0, -drift[0]], [0, 1, -drift[1]]])
            score = (cv2.warpAffine(ps, M, (w, h)) if score is None
                     else score + cv2.warpAffine(ps, M, (w, h)))
            pairs += 1
        prev = gray
    if score is None:
        sys.exit("video has fewer than 2 usable frames")
    return score

def reveal(score):
    score = np.clip(score, 0, None)
    hi = np.percentile(score, 99.5)
    norm = np.clip(score / hi * 255, 0, 255).astype(np.uint8) if hi > 0 else score.astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (5, 5), 0)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < mask.size // 20000:
            mask[lab == i] = 0
    return norm, mask

def ocr(mask):
    try:
        import pytesseract
    except ImportError:
        return None
    exe = shutil.which("tesseract")
    if not exe:
        return None
    pytesseract.pytesseract.tesseract_cmd = exe
    t = pytesseract.image_to_string(cv2.bitwise_not(mask), config="--psm 6").strip()
    return t or None

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
OUT = "out"; os.makedirs(OUT, exist_ok=True)
score = accumulate(iter_gray(VIDEO))
heat, mask = reveal(score)
cv2.imwrite(os.path.join(OUT, "revealed_heatmap.png"), heat)
cv2.imwrite(os.path.join(OUT, "revealed.png"), mask)
text = ocr(mask)
print("Text in the video: " + " ".join(text.split()) if text else
      "OCR unavailable — open out/revealed.png and read it.")
```
