# Ghost Font Decoder — use it in any chat (no install)

Want to decode a ghost-font video inside a normal chat — Claude, ChatGPT, or
Codex — without installing the plugin? Copy everything in this file, paste it
into the chat, and attach/upload your video. Any assistant with Python
execution (Claude's analysis tool, ChatGPT Advanced Data Analysis, Codex) will
run the **same optical-flow process** the plugin uses and reply with the hidden
message.

---

## Prompt to paste

You are a ghost-font video decoder. The attached video hides text as a field of
moving dots: every frame is uniform noise, but dots inside the letter shapes
drift against the background dots, so the message is only visible in motion and
invisible in any single frame. Do NOT try to OCR a raw frame — it will fail.

Instead, run the Python program below exactly as written (install `opencv-python`
and `numpy` first if needed), pointing it at the uploaded video. Then:

1. Read the printed `Hidden message:` value.
2. Open `revealed.png` and confirm the text visually matches; if OCR is
   unavailable or disagrees, read the message directly from `revealed.png`.
3. Reply with the hidden message and show `revealed.png`.

```python
# Ghost-font decoder — dense optical flow + drift registration + OCR.
# Same algorithm as https://github.com/haroontrailblazer/ghost-font-decoder
import sys, os, shutil
import cv2, numpy as np

VIDEO = "REPLACE_WITH_VIDEO_PATH"   # e.g. the uploaded file's path
OUT   = "."                          # where revealed.png is written

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
            reg = cv2.warpAffine(ps, M, (w, h))
            score = reg if score is None else score + reg
            pairs += 1
        prev = gray
    if score is None:
        sys.exit("video has fewer than 2 usable frames")
    print(f"{pairs} frame pairs, drift ({drift[0]:+.0f},{drift[1]:+.0f}) px")
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

score = accumulate(iter_gray(VIDEO))
heat, mask = reveal(score)
cv2.imwrite(os.path.join(OUT, "revealed_heatmap.png"), heat)
cv2.imwrite(os.path.join(OUT, "revealed.png"), mask)
text = ocr(mask)
print("Hidden message:\n" + text if text else "OCR unavailable — open revealed.png and read it.")
```

If the mask is weak or empty, replace the DIS flow with Farnebäck
(`cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)`),
and for high-frame-rate clips pass `stride=2` to `iter_gray`.
