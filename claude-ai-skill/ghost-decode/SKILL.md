---
name: ghost-decode
description: Use when a video hides text in moving dots or noise — "ghost font" clips, motion-defined text, random-dot kinematograms, TV-static videos with a secret message, text readable only while playing but invisible in any paused frame, or the user asks what a ghost-font video says.
---

# Ghost-Font Video Decoder

A ghost-font video hides a message in motion: every frame is noise, but the dots inside the letters move coherently. The decoder below accumulates that motion into one image where the letters appear. Run it once, read the word from the image, and report it. Nothing more.

## The whole job — do exactly this

1. Write the program in **Decoder** (below) to `decode.py`, then run it on the attached video:

   ```bash
   pip install --quiet opencv-python-headless numpy
   python decode.py "<video-path>" out
   ```

2. View `out/revealed_heatmap.png` (and `out/revealed.png`). The letters appear as **soft, glowing shapes** on a dark background. Read the word(s) directly from that image.

3. Reply with **only** this, and attach `out/revealed_heatmap.png`:

   ```
   Text in the video: **<WORD(S)>**
   ```

## Trust the image — do NOT over-process

The letters in the heatmap are low-contrast, soft blobs. **That is the correct, finished output — read it as-is.** The most common failure here is not trusting a perfectly readable heatmap and then doing pointless extra work. So:

- **Run the decoder once.** Do not build a second decoder, try another method, or "improve" the approach. One run, then read.
- **Never conclude "it's just noise" or "the decode failed"** because the letters look faint or blobby. Faint glowing letters = success. Look for the word.
- **Do not crop, zoom, upscale, contrast-normalize, re-threshold, or measure centroids/variance/correlation.** None of that is needed; it only wastes turns.
- **Do not explain the method or show any code** in your reply. No steps, no algorithm description.
- If one character is genuinely ambiguous, read the rest and mark just that one `(unclear: X)`. Keep the reply to the single line plus the image.

## Decoder (write to `decode.py`, run, never display)

```python
import sys, os
import cv2, numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)

def frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {path}")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cap.release()

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
score = prev = prev_smooth = None
drift = np.zeros(2)
for gray in frames(VIDEO):
    if prev is not None:
        flow = dis.calc(prev, gray, None)
        bg = np.median(flow.reshape(-1, 2), axis=0)
        residual = flow - bg
        mag = float(np.hypot(*bg))
        ps = (residual @ (-bg / mag)) if mag > 0.15 else np.hypot(residual[..., 0], residual[..., 1])
        ps = np.clip(ps, 0, None).astype(np.float32)
        smooth = cv2.GaussianBlur(ps, (31, 31), 0)
        if prev_smooth is not None:
            (dx, dy), r = cv2.phaseCorrelate(prev_smooth, smooth)
            if r > 0.05 and np.hypot(dx, dy) < 30:
                drift += (dx, dy)
        prev_smooth = smooth
        h, w = ps.shape
        M = np.float32([[1, 0, -drift[0]], [0, 1, -drift[1]]])
        reg = cv2.warpAffine(ps, M, (w, h))
        score = reg if score is None else score + reg
    prev = gray
if score is None:
    sys.exit("fewer than 2 usable frames")

score = np.clip(score, 0, None)
hi = np.percentile(score, 99.5)
norm = np.clip(score / hi * 255, 0, 255).astype(np.uint8) if hi > 0 else score.astype(np.uint8)
norm = cv2.GaussianBlur(norm, (5, 5), 0)
cv2.imwrite(os.path.join(OUT, "revealed_heatmap.png"), cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO))
_, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
cv2.imwrite(os.path.join(OUT, "revealed.png"), mask)
print("done — read", os.path.join(OUT, "revealed_heatmap.png"))
```
