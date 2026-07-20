---
name: ghost-decode
description: Use when a video hides text in moving dots or noise — "ghost font" clips, motion-defined text, random-dot kinematograms, TV-static videos with a secret message, text readable only while playing but invisible in any paused frame, or the user asks what a ghost-font video says.
argument-hint: "[video-path]"
---

# Ghost-Font Video Decoder

Ghost-font videos hide a message as a random-dot field: every frame is uniform
noise, but the dots inside the letter shapes move against the background dots.
This skill accumulates that motion into **two images** where the letters appear,
then reads the message from them.

## Hard rules — the whole job is ONE run producing TWO images

These rules exist because the #1 failure of this skill is over-processing: an
agent that doesn't trust the output spawns a dozen diagnostic images, invents new
algorithms, and hallucinates a message out of noise. Do not do that.

- **Run the decoder exactly once.** The algorithm below is correct and complete.
  Do not write a second decoder, try another method (temporal variance, phase
  correlation, sub-pixel warping, weighted accumulation, per-line crops…), or
  "improve" the pipeline.
- **Produce exactly two images: `revealed.png` and `revealed_heatmap.png`.**
  Create NO other images — no diagnostic maps, no crops, no re-thresholded or
  contrast-boosted variants. Extra images mean you are off the rails; stop.
- **Never OCR a raw frame.** Every single frame is pure noise; the message exists
  only in accumulated motion.
- **Read the two images, then stop.** Soft, rounded, blobby letters are the
  normal, correct output — not a reason to re-process. If you can read the word,
  report it.

## Steps

1. **Resolve the video path** from `$ARGUMENTS`, the user's request, or the most
   recently modified video (`.mp4`, `.mov`, `.avi`, `.webm`) in the working
   directory. Ask only if several candidates are plausible.
2. **Check dependencies:** `python -c "import cv2, numpy"`. If that fails,
   `pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"` (or
   `pip install opencv-python-headless numpy`).
3. **Decode — run once.** Pick ONE:
   - If `${CLAUDE_PLUGIN_ROOT}/decode.py` exists:
     `python "${CLAUDE_PLUGIN_ROOT}/decode.py" "<video>" -o "<out-dir>"`
   - Otherwise (plugin files not present in this environment): write the
     **Decoder** program at the bottom of this file verbatim to a scratch
     `decode.py`, then `python decode.py "<video>" "<out-dir>"`.

   Either path writes exactly `revealed.png` and `revealed_heatmap.png` and
   nothing else. `${CLAUDE_PLUGIN_ROOT}` is the plugin's install dir (on Windows
   PowerShell, `$env:CLAUDE_PLUGIN_ROOT`); if it expands empty, use the embedded
   Decoder instead.
4. **Read the message.** Read `revealed.png` with vision (it's a black background
   with the message in white); use `revealed_heatmap.png` to confirm a faint or
   merged glyph. The printed `OCR hint` line is only a rough hint from Tesseract —
   trust your own reading of the image over it. Mark any single ambiguous glyph
   `(unclear: X)`.

## Required response format

Show **both** images, then the text — nothing else:

```markdown
![revealed.png](<absolute-path-to-revealed.png>)

![revealed_heatmap.png](<absolute-path-to-revealed_heatmap.png>)

Text in the video: **<RECOVERED TEXT>**
```

Use absolute local paths so the images render in chat. Do not claim success if the
program did not run or you did not inspect `revealed.png`. If the mask genuinely
has no letter shapes (just specks / a uniformly dark heatmap), say no text was
recovered — still show the two images.

## Troubleshooting (still one run, still two images)

- **Weak or empty mask:** rerun the SAME decoder once with `--method farneback`,
  and for high-fps clips add `--stride 2`. That is the only permitted retry. It
  still produces just the two images — do not switch algorithms or add diagnostic
  renders.
- **Long video:** add `--max-frames 200`; a few seconds of footage is enough.
- **No Tesseract:** fine — read the text from `revealed.png` yourself.

## Decoder (write to a scratch `decode.py` only if the bundled one is absent)

```python
import sys, os, shutil
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


def frame_to_text(mask, heat, pad_frac=0.08):
    # Crop both images tightly to the text and enlarge, so a small glyph (a lone
    # `I`, an accent, a short top line) is big and obvious instead of a few pixels
    # lost in a mostly-empty frame. The mask defines the box; the heatmap matches.
    ys, xs = np.where(mask > 127)
    if ys.size == 0:
        return mask, heat
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    hh, ww = mask.shape
    pad = int(pad_frac * max(x1 - x0, y1 - y0)) + 8
    y0, y1 = max(0, y0 - pad), min(hh, y1 + pad + 1)
    x0, x1 = max(0, x0 - pad), min(ww, x1 + pad + 1)
    mask, heat = mask[y0:y1, x0:x1], heat[y0:y1, x0:x1]
    long_side = max(mask.shape[:2])
    if long_side < 1000:
        f = min(4.0, 1000.0 / long_side)
        size = (int(mask.shape[1] * f), int(mask.shape[0] * f))
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        heat = cv2.resize(heat, size, interpolation=cv2.INTER_CUBIC)
    return mask, heat


# --- accumulate motion against the background (letters move, background drifts) ---
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

# --- build the two images: heatmap (raw score) + clean mask ---
score = np.clip(score, 0, None)
hi = np.percentile(score, 99.5)
norm = np.clip(score / hi * 255, 0, 255).astype(np.uint8) if hi > 0 else score.astype(np.uint8)
norm = cv2.GaussianBlur(norm, (5, 5), 0)

_, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
h_img, w_img = mask.shape
n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
for i in range(1, n):
    x, y, w, h, area = st[i]
    band = w >= 5 * h and h <= h_img // 18            # wide, short
    at_edge = x <= 2 or x + w >= w_img - 2            # drift bands hug an edge
    if area < mask.size // 20000 or (band and (at_edge or w >= w_img // 3)):
        mask[lab == i] = 0

# --- optional OCR hint on the full-frame mask (never authoritative) ---
try:
    import pytesseract
    exe = shutil.which("tesseract")
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
        t = pytesseract.image_to_string(cv2.bitwise_not(mask), config="--psm 6").strip()
        print("OCR hint (unreliable):", " ".join(t.split()) if t else "(none)")
except Exception:
    pass

# --- crop both images tightly to the text (a lone I stays visible), then save ---
mask, norm = frame_to_text(mask, norm)
cv2.imwrite(os.path.join(OUT, "revealed_heatmap.png"), norm)
cv2.imwrite(os.path.join(OUT, "revealed.png"), mask)
print("done — wrote revealed.png and revealed_heatmap.png (the only two outputs)")
```
