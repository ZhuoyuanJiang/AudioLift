"""
Preprocess an in-the-wild photo into a StableAvatar training-distribution reference.

Reference preprocessing for the five-view video generator.

Pipeline (per image):
  1. Matting (RVM by default; rembg fallback; 'none' for already-white-bg images)
  2. Composite onto white:  out = fg * alpha + white * (1 - alpha)
     -- same formula as the training converter (convert_nersemble_multiview_512.py)
  3. Framing, two modes (the Session-69 A/B experiment):
       face   (mode A): MediaPipe face landmarks -> face bbox -> build a canvas in
               which the face bbox height-fraction and center position MATCH the
               NeRSemble training references (calibrated stats, not guessed).
               Regions of the canvas outside the source image are filled WHITE.
       fitpad (mode B): scale the whole image to fit inside the target aspect,
               pad the remainder with white. Never crops, never distorts.
  4. Resize to --width x --height (default 832x1216, the S63/S65 training size)
  5. Optional pose gate: reject if |yaw| > --pose_reject_deg

Framing uses the provided configs/stage1/nersemble_framing_calibration.json.

Process:
  python scripts/preprocess/preprocess_reference.py \
      --input photo.jpg --output_dir out/ --mode both --matting rvm

Outputs: <stem>_face.png / <stem>_fitpad.png + <stem>_meta.json sidecar.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDIAPIPE_MODEL_PATH = os.path.join(REPO_ROOT, "external", "mediapipe_models", "face_landmarker.task")
DEFAULT_CALIB_JSON = os.path.join(REPO_ROOT, "configs", "stage1", "nersemble_framing_calibration.json")


# ============================================================================
# Matting
# ============================================================================

def load_rvm(device):
    import torch
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")
    return model.eval().to(device)


def matte_rvm(img_pil, model, device):
    """RVM on a single image -> alpha in [0,1] float32 (H, W)."""
    import torch
    import torchvision.transforms.functional as TF
    src = TF.to_tensor(img_pil.convert("RGB")).unsqueeze(0).to(device)
    h, w = src.shape[-2:]
    # Repo-recommended ratio: matting runs internally around ~512px on the long side
    downsample_ratio = min(512.0 / max(h, w), 1.0)
    rec = [None] * 4
    with torch.no_grad():
        fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)
    alpha = pha[0, 0].clamp(0, 1).cpu().numpy().astype(np.float32)
    fg = fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return fg, alpha


def matte_rembg(img_pil):
    """rembg fallback -> (fg, alpha). fg is the original RGB (rembg gives no clean fg)."""
    from rembg import remove
    rgba = remove(img_pil.convert("RGB"))
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    return arr[..., :3], arr[..., 3]


def composite_white(fg, alpha):
    """Training-converter formula: fg * alpha + white * (1 - alpha)."""
    a3 = alpha[..., None]
    out = fg * a3 + 1.0 * (1.0 - a3)
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8), "RGB")


# ============================================================================
# Face detection (MediaPipe FaceLandmarker — same model file the repo already
# uses for lip masks) + yaw estimate
# ============================================================================

def create_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.isfile(MEDIAPIPE_MODEL_PATH):
        raise FileNotFoundError(f"MediaPipe model not found: {MEDIAPIPE_MODEL_PATH}")
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
        num_faces=1,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_face(img_pil, landmarker):
    """Returns dict {bbox: (x0,y0,x1,y1) px, yaw/pitch/roll deg} or None."""
    import mediapipe as mp
    rgb = np.asarray(img_pil.convert("RGB"))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    w, h = img_pil.size
    xs = [lm.x * w for lm in res.face_landmarks[0]]
    ys = [lm.y * h for lm in res.face_landmarks[0]]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    yaw = pitch = roll = None
    if res.facial_transformation_matrixes:
        R = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))))
        roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return {"bbox": bbox, "yaw": yaw, "pitch": pitch, "roll": roll}


# ============================================================================
# Framing
# ============================================================================

def frame_fitpad(comp, out_w, out_h):
    """Mode B: scale whole image to fit inside out_w x out_h, pad with white."""
    w, h = comp.size
    scale = min(out_w / w, out_h / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = comp.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    canvas.paste(resized, ((out_w - nw) // 2, (out_h - nh) // 2))
    return canvas


def frame_face(comp, det, calib, out_w, out_h):
    """
    Mode A: build a crop window (in source pixels) such that, after resize to
    out_w x out_h, the face bbox lands at the calibrated height-fraction and
    center position. Window regions outside the source are filled white.
    """
    x0, y0, x1, y1 = det["bbox"]
    face_h = y1 - y0
    face_cx = (x0 + x1) / 2.0
    face_cy = (y0 + y1) / 2.0

    win_h = face_h / calib["face_h_frac"]
    win_w = win_h * (out_w / out_h)
    win_x0 = face_cx - calib["face_cx_frac"] * win_w
    win_y0 = face_cy - calib["face_cy_frac"] * win_h

    # Crop with white out-of-bounds fill
    win_w_i, win_h_i = round(win_w), round(win_h)
    canvas = Image.new("RGB", (win_w_i, win_h_i), (255, 255, 255))
    src_x0, src_y0 = round(win_x0), round(win_y0)
    w, h = comp.size
    ix0, iy0 = max(src_x0, 0), max(src_y0, 0)
    ix1, iy1 = min(src_x0 + win_w_i, w), min(src_y0 + win_h_i, h)
    if ix1 > ix0 and iy1 > iy0:
        canvas.paste(comp.crop((ix0, iy0, ix1, iy1)), (ix0 - src_x0, iy0 - src_y0))
    return canvas.resize((out_w, out_h), Image.LANCZOS)


# ============================================================================
# Calibration
# ============================================================================



# ============================================================================
# Main processing
# ============================================================================

def process_images(args):
    inputs = []
    for pat in args.input:
        hits = sorted(glob.glob(pat))
        inputs.extend(hits if hits else [pat])
    os.makedirs(args.output_dir, exist_ok=True)

    modes = ["face", "fitpad"] if args.mode == "both" else [args.mode]

    calib = None
    if "face" in modes:
        if not os.path.isfile(args.calib_json):
            sys.exit(f"Calibration JSON missing ({args.calib_json}). Restore the provided calibration file.")
        with open(args.calib_json) as f:
            calib = json.load(f)

    rvm = device = None
    if args.matting == "rvm":
        import torch
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        rvm = load_rvm(device)
    landmarker = create_landmarker() if ("face" in modes or args.pose_reject_deg) else None

    # Disambiguate identical basenames (e.g. every examples/case-*/reference.png)
    # by prefixing the parent directory name.
    base_stems = [os.path.splitext(os.path.basename(p))[0] for p in inputs]
    dup = {s for s in base_stems if base_stems.count(s) > 1}

    for path in inputs:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in dup:
            stem = f"{os.path.basename(os.path.dirname(path))}_{stem}"
        img = Image.open(path).convert("RGB")
        meta = {"source": path, "matting": args.matting, "size_in": img.size,
                "size_out": [args.width, args.height]}

        # 1-2. matting + white composite
        if args.matting == "rvm":
            fg, alpha = matte_rvm(img, rvm, device)
            comp = composite_white(fg, alpha)
        elif args.matting == "rembg":
            fg, alpha = matte_rembg(img)
            comp = composite_white(fg, alpha)
        else:
            comp = img

        # face detection (on the composite — background noise already removed)
        det = detect_face(comp, landmarker) if landmarker else None
        if det:
            meta.update({"yaw": det["yaw"], "pitch": det["pitch"], "roll": det["roll"]})
        if args.pose_reject_deg and det and det["yaw"] is not None and abs(det["yaw"]) > args.pose_reject_deg:
            print(f"[reject] {stem}: |yaw|={abs(det['yaw']):.1f} > {args.pose_reject_deg}")
            meta["rejected"] = "pose"
            with open(os.path.join(args.output_dir, f"{stem}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            continue

        # 3-4. framing + resize
        for mode in modes:
            if mode == "face":
                if det is None:
                    print(f"[warn] {stem}: no face detected, skipping face mode")
                    meta["face_mode"] = "skipped_no_face"
                    continue
                out = frame_face(comp, det, calib, args.width, args.height)
            else:
                out = frame_fitpad(comp, args.width, args.height)
            out_path = os.path.join(args.output_dir, f"{stem}_{mode}.png")
            out.save(out_path)
            meta[f"output_{mode}"] = out_path
            print(f"[ok] {out_path}" + (f"  (yaw={det['yaw']:.1f})" if det and det["yaw"] is not None else ""))

        with open(os.path.join(args.output_dir, f"{stem}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    if landmarker is not None:
        # MediaPipe's FaceLandmarker.__del__ raises a harmless TypeError at
        # interpreter shutdown when it is left to the garbage collector; close it
        # here so the gate's output is not followed by a bogus traceback.
        try:
            landmarker.close()
        except Exception:
            pass


# ============================================================================
# The gate — importable, so no inference entry point can silently take a raw photo
# ============================================================================
# Until 2026-08-28 this module was a tool you had to REMEMBER to run. The compliant
# references in the repo were a frozen directory produced by one manual run back in
# Session 69; every later script hardcoded those paths, so anyone reusing them was
# right by accident and anyone cutting a fresh frame fell off the path with nothing
# to catch them. These two functions are what Stage-1 entry points
# call so that "was this photo preprocessed?" is answered by the code, not by memory.
#
# What actually has to hold (measured 2026-08-28 over 12 real training references,
# 4 fixed in-the-wild references, and 4 raw frames):
#
#                            size          white%   top-strip%   border-ring%
#   training refs (12)     2200x3208       38-47      100          67-70
#   fixed wild refs (4)     832x1216       35-47      100          68-90
#   raw frames (4)      1280/1920 landscape  0-20       0-30           0-16
#
# So the exact pixel size is NOT the invariant -- real training references are
# 2200x3208 and get resized at load time. The two invariants are:
#   1. aspect ratio ~= 0.684 (2200/3208 = .6858, 832/1216 = .6842). A landscape
#      photo violates it, and Stage 1 resizes without preserving aspect, so the
#      face arrives at the model squashed.
#   2. the strip above the head is background, hence pure white after matting.
#      Compliant: 100% on all 16. Non-compliant: 30% at best. The gap is wide
#      enough that a 90% threshold needs no tuning.

TOP_STRIP_FRAC = 0.05      # fraction of image height sampled above the head
WHITE_LEVEL = 235          # min(R,G,B) above this counts as "white backdrop"
TOP_WHITE_MIN = 90.0       # percent; compliant refs measure 100
ASPECT_TOL = 0.03          # relative tolerance on w/h

# Reference-gate validation: the top-strip rule above was calibrated on 16
# bare-headed references and misfires on headdresses -- 9 of the 42 frozen
# RenderMe360 references (0048, 0156, 0168, 0195, 0262) have a hat or crown in
# the top 5% and measured 37-89% strip white although they are cleanly matted.
# What is background in every compliant reference is the region BESIDE the
# head, so the decision now uses the two top corners:
#
#                                 top-left / top-right corner white%
#   84 known-compliant refs          min 96 (0195_s2), the rest 100
#   (NeRSemble 10, RM360 42, HDTF 32)
#   7 known-raw photos (v1 studio     max 0
#   frames, case1 sources)
#
# Gap 96 vs 0: no tuning needed. The strip percentage stays in stats for
# information only. Corner side = CORNER_FRAC of the shorter image side
# (~50 px on 832x1216), so a hat may touch the top edge as long as it does not
# spread into both corners.
CORNER_FRAC = 0.06
CORNER_WHITE_MIN = 95.0


def check_reference(image, width=832, height=1216):
    """Is this image already a training-distribution reference?

    Accepts a path or a PIL image. Returns (ok, problems, stats) and never raises
    on a normal image -- callers warn, they do not abort.
    """
    img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
    a = np.asarray(img, np.float32)
    h, w, _ = a.shape
    white = a.min(-1) > WHITE_LEVEL
    stats = {
        "size": [w, h],
        "aspect": round(w / h, 4),
        "aspect_target": round(width / height, 4),
        "white_pct": round(float(white.mean() * 100), 1),
        "top_white_pct": round(float(white[:max(1, int(h * TOP_STRIP_FRAC))].mean() * 100), 1),
    }
    c = max(1, int(min(h, w) * CORNER_FRAC))
    stats["top_corner_white_pct"] = [round(float(white[:c, :c].mean() * 100), 1),
                                     round(float(white[:c, -c:].mean() * 100), 1)]
    problems = []
    target = width / height
    if abs(w / h - target) / target > ASPECT_TOL:
        problems.append(
            "长宽比 %.3f 偏离目标 %.3f（横版/方图会被非等比拉伸，脸会压扁）" % (w / h, target))
    tl, tr = stats["top_corner_white_pct"]
    if min(tl, tr) < CORNER_WHITE_MIN:
        problems.append(
            "头两侧的上角只有 %.0f%% / %.0f%% 是白底（合规参考 ≥%.0f%%）—— 像是没抠背景的原图"
            % (tl, tr, CORNER_WHITE_MIN))
    return (not problems), problems, stats


class ReferenceGateError(RuntimeError):
    """The reference needs strict review or cannot be read for inference."""


def preprocess_fitpad(image, width=832, height=1216):
    """Reuse the evaluated ITW fallback: rembg, or original pixels, then fitpad."""
    details = {"method": "rembg_fitpad"}
    try:
        fg, alpha = matte_rembg(image)
        coverage = float(alpha.mean())
        details["alpha_mean"] = coverage
        if coverage >= 0.08:
            comp = composite_white(fg, alpha)
        else:
            comp = image
            details.update(method="original_fitpad", reason="rembg retained less than 8% foreground")
    except Exception as error:
        comp = image
        details.update(method="original_fitpad", reason=f"rembg failed: {type(error).__name__}: {error}")
    return frame_fitpad(comp, width, height), details


def ensure_reference(path, width=832, height=1216, cache_dir=None, auto=True, tag="reference",
                     allow_noncompliant=False, strict=False, skip=False, cpu=False):
    """Prepare a reference automatically and save the exact RGB image used.

    Compliant images keep their pixels. Other images use the existing human
    preprocessing, then the evaluated rembg/fitpad fallback if necessary.
    Strict mode preserves the original candidate-review pause instead.
    """
    def say(msg):
        print("[%s] %s" % (tag, msg), flush=True)

    path = os.path.abspath(os.fspath(path))
    cache_dir = os.path.abspath(cache_dir or "_gate_candidate")
    output_dir = os.path.dirname(cache_dir)
    os.makedirs(output_dir, exist_ok=True)
    record_path = os.path.join(output_dir, "reference_processing.json")
    record = {"source": path, "mode": "strict" if strict else "auto", "attempts": []}

    def save_record():
        with open(record_path, "w") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def use(image, method):
        used_path = os.path.join(output_dir, "reference_used.png")
        image.save(used_path, format="PNG")
        record.update(status="ready", method=method, reference_used=used_path,
                      output_size=list(image.size))
        save_record()
        say(f"Using {method}: {used_path}; processing record: {record_path}")
        return used_path

    try:
        with Image.open(path) as source:
            original = source.convert("RGB")
    except (OSError, ValueError) as error:
        record.update(status="error", error=f"Cannot read reference: {error}")
        save_record()
        raise ReferenceGateError(record["error"]) from error

    if skip:
        record["mode"] = "skip_check"
        return use(original, "original")

    ok, problems, stats = check_reference(original, width, height)
    record["input_check"] = {"passed": ok, "problems": problems, "stats": stats}
    if ok:
        return use(original, "original")

    if allow_noncompliant:
        record["mode"] = "allow_noncompliant"
        say("WARNING: explicitly using the original image despite its reference-check results")
        return use(original, "original")

    for pr in problems:
        say("Preprocessing needed: " + pr)

    candidate = candidate_image = None
    if auto:
        attempt = {"method": "rvm_face"}
        try:
            candidate = preprocess_one(path, cache_dir, width=width, height=height, cpu=cpu)
            if candidate:
                with Image.open(candidate) as prepared:
                    candidate_image = prepared.convert("RGB")
            attempt["status"] = "prepared" if candidate else "no_face"
        except (Exception, SystemExit) as error:
            candidate = None
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["attempts"].append(attempt)

    if strict or not auto:
        record.update(status="review_required", candidate=candidate)
        save_record()
        if candidate:
            say(f"Strict review: inspect {candidate}, then pass it explicitly as --reference")
        else:
            say("Strict review: no face-framed candidate was produced; see the processing record")
        raise ReferenceGateError("; ".join(problems))

    if candidate:
        return use(candidate_image, "rvm_face")

    say("Human preprocessing did not produce a reference; trying rembg and fitpad on the original image")
    prepared, details = preprocess_fitpad(original, width, height)
    record["attempts"].append(details)
    if details["method"] == "original_fitpad":
        say("WARNING: keeping the original background and fitting the full image: " + details["reason"])
    return use(prepared, details["method"])


def preprocess_one(path, output_dir, width=832, height=1216, matting="rvm", cpu=False):
    """Single-image face-mode preprocessing, reusing the exact same code as the CLI.

    Returns the output path, or None if no face was detected.
    """
    args = argparse.Namespace(
        input=[path], output_dir=output_dir, width=width, height=height,
        mode="face", matting=matting, pose_reject_deg=None, cpu=cpu,
        calib_json=DEFAULT_CALIB_JSON,
    )
    process_images(args)
    stem = os.path.splitext(os.path.basename(path))[0]
    # A previous run may have left a face PNG in this directory. Only accept an
    # output recorded by the current call, which also writes metadata on no-face.
    with open(os.path.join(output_dir, "%s_meta.json" % stem)) as handle:
        out = json.load(handle).get("output_face")
    return out if out and os.path.isfile(out) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib_json", type=str, default=DEFAULT_CALIB_JSON)

    p.add_argument("--input", type=str, nargs="*", default=[], help="Image path(s) or glob(s)")
    p.add_argument("--output_dir", type=str, default="./preprocessed_refs")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1216)
    p.add_argument("--mode", choices=["face", "fitpad", "both"], default="face",
                   help="face = calibrated face-detector framing (default; user decision 2026-06-12). "
                        "fitpad kept as fallback for detection failures / A-B experiments.")
    p.add_argument("--matting", choices=["rvm", "rembg", "none"], default="rvm")
    p.add_argument("--pose_reject_deg", type=float, default=None,
                   help="Reject images with |yaw| above this (off by default)")
    p.add_argument("--cpu", action="store_true", help="Force CPU for matting")
    args = p.parse_args()

    if args.input:
        process_images(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
