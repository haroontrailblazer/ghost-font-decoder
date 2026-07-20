"""Break "ghost font" videos with dense optical flow.

Ghost-font clips hide text as a random-dot field: every frame is uniform
noise, but dots inside the glyphs drift against the background dots, so the
message is only visible in motion. Frame-by-frame OCR sees nothing.

This decoder recovers the message from the motion itself:
  1. compute dense optical flow between consecutive frames,
  2. subtract the median (background) flow,
  3. score each pixel by how strongly it moves against the background,
  4. accumulate scores over the whole clip, threshold, and clean up,
  5. OCR the revealed mask (optional, needs Tesseract).

Usage:
  python decode.py ghost-message.mp4
  python decode.py clip.mp4 -o out_dir --method farneback --stride 2 --no-ocr
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np


def iter_gray_frames(path, stride=1, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"error: cannot open video: {path}")
    index = yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        index += 1
    cap.release()


def make_flow_fn(method):
    if method == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return lambda a, b: dis.calc(a, b, None)
    return lambda a, b: cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def accumulate_opposition(frames, flow_fn):
    """Sum, over all frame pairs, each pixel's motion against the background.

    Background motion is the per-pair median flow (the background covers most
    of the frame, so the median is immune to the text region). Text pixels
    moving opposite the background score strongly positive; incoherent noise
    averages out to ~0 over many pairs.

    The text itself drifts across the clip, so each pair's score map is
    shifted back by the text's cumulative displacement before accumulating —
    otherwise the letters smear into blobs.
    """
    score = None
    prev = None
    prev_smooth = None
    pairs = 0
    drift = np.zeros(2, dtype=np.float64)
    for gray in frames:
        if prev is not None:
            flow = flow_fn(prev, gray)
            bg = np.median(flow.reshape(-1, 2), axis=0)
            residual = flow - bg
            mag = float(np.hypot(*bg))
            if mag > 0.15:
                # signed projection of residual onto the anti-background axis
                pair_score = residual @ (-bg / mag)
            else:
                # background barely moves: fall back to residual magnitude
                pair_score = np.hypot(residual[..., 0], residual[..., 1])
            pair_score = np.clip(pair_score, 0, None).astype(np.float32)

            # the glyph REGION drifts slowly even though the dots inside it
            # stream fast — track the region via phase correlation between
            # consecutive score maps, dots average out under the blur
            smooth = cv2.GaussianBlur(pair_score, (31, 31), 0)
            if prev_smooth is not None:
                (dx, dy), response = cv2.phaseCorrelate(prev_smooth, smooth)
                if response > 0.05 and np.hypot(dx, dy) < 30:
                    drift += (dx, dy)
            prev_smooth = smooth

            # register this pair's score map against the first frame
            h, w = pair_score.shape
            shift = np.float32([[1, 0, -drift[0]], [0, 1, -drift[1]]])
            registered = cv2.warpAffine(pair_score, shift, (w, h))
            score = registered if score is None else score + registered

            pairs += 1
            if pairs % 50 == 0:
                print(f"  {pairs} frame pairs processed")
        prev = gray
    if score is None:
        sys.exit("error: video has fewer than 2 usable frames")
    print(f"  {pairs} frame pairs total, text drift compensated: "
          f"({drift[0]:+.0f}, {drift[1]:+.0f}) px")
    return score


def reveal_mask(score):
    """Build a strong mask, then recover coherent lower-energy glyphs."""
    score = np.clip(score, 0, None)
    hi = np.percentile(score, 99.5)
    norm = np.clip(score / hi * 255, 0, 255).astype(np.uint8) if hi > 0 else score.astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (5, 5), 0)
    otsu, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # A weak glyph can peak just below global Otsu while remaining far brighter
    # than its background. Recover only new, compact components; wide scan-line
    # and camera-pan bands are rejected.
    faint_cutoff = max(8, int(round(otsu * 0.5)))
    _, faint = cv2.threshold(norm, faint_cutoff, 255, cv2.THRESH_BINARY)
    faint_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    faint = cv2.morphologyEx(faint, cv2.MORPH_CLOSE, faint_kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(faint)
    height, width = mask.shape
    min_faint_area = max(20, mask.size // 4000)
    min_faint_height = max(5, height // 40)
    min_faint_peak = max(faint_cutoff + 1, int(round(otsu * 0.75)))
    recovered = 0
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        component = labels == i
        peak = int(norm[component].max())
        strong_overlap = np.count_nonzero(mask[component]) / area
        compact = w <= 4 * h and w <= width // 3
        coherent = area >= min_faint_area and h >= min_faint_height
        if compact and coherent and strong_overlap < 0.1 and peak >= min_faint_peak:
            mask[component] = 255
            recovered += 1
    if recovered:
        print(
            f"  recovered {recovered} faint coherent component(s) "
            f"below Otsu ({faint_cutoff} vs {otsu:.0f})"
        )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = mask.size // 20000
    for i in range(1, n):
        x, _, w, h, area = stats[i]
        # Drop drift-registration streaks: wide, short horizontal bands that hug
        # a frame edge or span a huge width. Real glyphs never do both — even a
        # dash or underscore mid-word sits away from the edges and isn't that
        # wide — so letters and punctuation are untouched.
        band = w >= 5 * h and h <= height // 18
        at_edge = x <= 2 or x + w >= width - 2
        if area < min_area or (band and (at_edge or w >= width // 3)):
            mask[labels == i] = 0
    return norm, mask


def find_tesseract():
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def has_isolated_i(mask):
    """Recognize a recovered uppercase I that page-layout OCR may ignore."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    min_area = max(20, mask.size // 4000)
    components = [
        stats[i] for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    for x, y, w, h, area in components:
        aspect = w / h
        fill = area / (w * h)
        if aspect > 0.5 or fill < 0.45 or h < height // 10:
            continue
        cx = x + w / 2
        below = [item for item in components if item[1] > y + h * 0.75]
        if len(below) < 2:
            continue
        below_left = min(item[0] for item in below)
        below_right = max(item[0] + item[2] for item in below)
        if below_left <= cx <= below_right and width * 0.1 < cx < width * 0.9:
            return True
    return False


def run_ocr(mask):
    try:
        import pytesseract
    except ImportError:
        print("OCR skipped: pytesseract not installed (pip install pytesseract)")
        return None
    exe = find_tesseract()
    if not exe:
        print("OCR skipped: tesseract binary not found (winget install UB-Mannheim.TesseractOCR)")
        return None
    pytesseract.pytesseract.tesseract_cmd = exe
    ocr_img = cv2.bitwise_not(mask)  # tesseract wants dark text on white
    text = pytesseract.image_to_string(ocr_img, config="--psm 6").strip()
    uppercase = pytesseract.image_to_string(
        ocr_img,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ).strip()
    alnum = lambda value: sum(char.isalnum() for char in value)
    has_punctuation_token = any(
        not any(char.isalnum() for char in token) for token in text.split()
    )
    if uppercase and has_punctuation_token and alnum(uppercase) >= alnum(text):
        text = uppercase
    text = " ".join(
        token for token in text.split()
        if any(char.isalnum() for char in token)
    )
    if text and has_isolated_i(mask) and text.split()[0].upper() != "I":
        text = "I " + text
    return text or None


def main():
    ap = argparse.ArgumentParser(description="Decode ghost-font (motion-defined text) videos via optical flow")
    ap.add_argument("video", help="path to the ghost-font video")
    ap.add_argument("-o", "--out-dir", default=".", help="directory for output images (default: current dir)")
    ap.add_argument("--method", choices=["dis", "farneback"], default="dis", help="dense optical flow method")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (default 1)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap on frames processed")
    ap.add_argument("--no-ocr", action="store_true", help="skip the OCR step")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Decoding {args.video} ({args.method} flow, stride {args.stride})")
    frames = iter_gray_frames(args.video, args.stride, args.max_frames)
    score = accumulate_opposition(frames, make_flow_fn(args.method))
    heat, mask = reveal_mask(score)

    heat_path = os.path.join(args.out_dir, "revealed_heatmap.png")
    mask_path = os.path.join(args.out_dir, "revealed.png")
    cv2.imwrite(heat_path, heat)
    cv2.imwrite(mask_path, mask)
    print(f"Wrote {heat_path} (raw opposition score) and {mask_path} (clean mask)")

    if not args.no_ocr:
        text = run_ocr(mask)
        if text:
            single_line = " ".join(text.split())  # collapse OCR line breaks
            print("\nText in the video: " + single_line)
        elif text is None:
            pass  # reason already printed (no OCR engine)
        else:
            print("\nText in the video: (none found — check revealed.png visually)")


if __name__ == "__main__":
    main()
