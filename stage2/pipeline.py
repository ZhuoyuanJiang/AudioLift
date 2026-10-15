#!/usr/bin/env python3
"""RELEASE-16: reconstruct each Stage-1 video frame from five calibrated views.

The model, view preparation and renderer are the existing RELEASE-16 code.
The camera paths come from StableAvatar/scripts/s79_orbit4d.py; this entrypoint
fixes the final checkpoint/config and checks media before and after rendering.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_models import MODELS, ensure_checkpoint

FPS = 25
CHECKPOINT_SHA256 = "dcbef48d163f880133d1bd2adbf46140c6d45d0b9cbcebb12276b3bea5a537dc"
CONFIG = "configs/stage2/inference.yaml"
GRID_CAMERAS = "configs/stage2/cameras/opencv_cameras_nersemble_017_832x1216.json"
FRONT_CAMERA = "configs/stage2/cameras/opencv_cameras_nersemble_frontal_canonical_832x1216.json"
PINNED_FILES = {
    CONFIG: "bbaf3c6c25c5a1b853164a07600f9b7c7ef2246051c14900effe4cd4a171655a",
    GRID_CAMERAS: "7a55a91bd7c4453d76069a2be9db224110cea4b684c4e337e31e236ee2aabe38",
    FRONT_CAMERA: "1afb898f84290c14e4d8da860b938919f959c1ebff9c674144bfe50c3f146a92",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n",
                    *map(str, args)], check=True)


def probe_video(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    if result.stderr.strip():
        raise ValueError(f"Cannot decode {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No video stream in {path}")
    stream = data["streams"][0]
    frames = int(stream.get("nb_read_frames", 0))
    timestamps = data.get("frames", [])
    if frames < 1 or len(timestamps) != frames:
        raise ValueError(f"Cannot decode all frames of {path}")
    for index, frame in enumerate(timestamps):
        timestamp = float(frame.get("best_effort_timestamp_time", "nan"))
        if not math.isfinite(timestamp) or abs(timestamp - index / FPS) > 1e-5:
            raise ValueError(f"{path}: frame {index} must be at {index / FPS}s (25 fps, start 0)")
    return {"width": stream["width"], "height": stream["height"], "frames": frames}


def probe_audio(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=duration,start_time,sample_rate", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    if result.stderr.strip():
        raise ValueError(f"Cannot inspect {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No audio stream in {path}")
    duration = float(streams[0].get("duration", "nan"))
    start = float(streams[0].get("start_time", 0))
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(start) or abs(start) > 1e-5:
        raise ValueError(f"{path}: audio must have a known positive duration and start at 0")
    return {"duration": duration, "sample_rate": int(streams[0]["sample_rate"])}


def validate_inputs(stage1_dir, audio, max_frames=None):
    grid = probe_video(stage1_dir / "grid.mp4")
    front = probe_video(stage1_dir / "front.mp4")
    sound = probe_audio(audio)
    if (grid["width"], grid["height"]) != (832, 1216):
        raise ValueError("grid.mp4 must be 832x1216 for the fixed 416x608 input camera intrinsics")
    if grid["frames"] != front["frames"]:
        raise ValueError("grid.mp4 and front.mp4 must have the same frame count")
    if abs(sound["duration"] - grid["frames"] / FPS) > 1 / FPS + 1e-5:
        raise ValueError("Audio duration must match the input videos within one frame (0.04s)")
    if max_frames is not None and not 1 <= max_frames <= grid["frames"]:
        raise ValueError(f"--max-frames must be between 1 and {grid['frames']}")
    return {"grid": grid, "front": front, "audio": sound,
            "frames": max_frames or grid["frames"]}


def coverage_azimuth(frame, num_frames):
    """Historical four cosine-eased legs: 0 -> +51 -> 0 -> -40 -> 0."""
    import numpy as np

    if num_frames <= 1 or frame == num_frames - 1:
        return 0.0
    phase = 4.0 * frame / (num_frames - 1)
    segment = min(int(phase), 3)
    weight = 0.5 - 0.5 * np.cos(np.pi * (phase - segment))
    keys = (0.0, 51.0, 0.0, -40.0, 0.0)
    return keys[segment] * (1.0 - weight) + keys[segment + 1] * weight


def rotation_z(degrees):
    import numpy as np

    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1] = c, -s, s, c
    return matrix


def orbit_azimuth(frame, num_frames, turns):
    if num_frames <= 1:
        return 0.0
    return 360.0 * turns * frame / (num_frames - 1 if turns == 1 else num_frames)


def render_frames(stage1_dir, checkpoint, work, frames, turns, gpu):
    import numpy as np
    from PIL import Image
    import torch
    from diff_gaussian_rasterization import _C as rasterizer
    from stage2.view_inputs import extract_views_from_grid, initialize_gslrm_model, prepare_views
    from stage2.view_inputs import fit_pad, load_cams, build_batch
    from stage2.rendering import render_opencv_cam

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 reconstruction requires CUDA; use --dry-run for CPU preflight")
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    model = initialize_gslrm_model(str(checkpoint), str(ROOT / CONFIG), device)
    model.eval()
    intrinsics, poses = load_cams(ROOT / GRID_CAMERAS, ROOT / FRONT_CAMERA, device, True)
    camera = json.loads((ROOT / FRONT_CAMERA).read_text())["frames"][0]
    frontal = np.asarray(camera["c2w"], dtype=np.float32)
    render_intrinsics = torch.tensor([camera[key] * 2 for key in ("fx", "fy", "cx", "cy")],
                                     dtype=torch.float32, device=device)
    for name in ("grid", "front"):
        ffmpeg("-i", stage1_dir / f"{name}.mp4", "-map", "0:v:0", "-vsync", "0",
               "-frames:v", frames, "-start_number", "0", work / f"{name}_%06d.png")
    with torch.no_grad():
        for frame in range(frames):
            with Image.open(work / f"grid_{frame:06d}.png") as grid:
                views = extract_views_from_grid(grid.convert("RGB"))
            with Image.open(work / f"front_{frame:06d}.png") as front:
                views.append(fit_pad(front.convert("RGB"), 416, 608))
            batch = build_batch(prepare_views(views, target_size=0), intrinsics, poses, device)
            with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
                result = model.forward(batch, create_visual=False, split_data=False)
            gaussians = result.gaussians[0]  # No filtering: RELEASE-16 is evaluated unfiltered.
            targets = {
                "frontal": frontal,
                "coverage": rotation_z(coverage_azimuth(frame, frames)) @ frontal,
                "orbit360": rotation_z(orbit_azimuth(frame, frames, turns)) @ frontal,
            }
            for name, pose in targets.items():
                rendered = render_opencv_cam(gaussians, 1216, 832,
                                             torch.from_numpy(pose).to(device), render_intrinsics)
                rgb = rendered["render"].detach().cpu().permute(1, 2, 0).numpy() * 255
                Image.fromarray(rgb.clip(0, 255).astype("uint8")).save(work / f"{name}_{frame:06d}.png")
            del result, gaussians, batch, rendered
            if frame % 20 == 0 or frame == frames - 1:
                print(f"Reconstructed frame {frame + 1}/{frames}", flush=True)
    return {"torch": torch.__version__, "numpy": np.__version__, "pillow": Image.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "rasterizer_sha256": sha256(rasterizer.__file__),
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}


def encode_audio(audio, work, trim_seconds=None):
    # Encode the full audio independently. Combining -frames:v and -shortest
    # with audio encoding used to lose the audio tail on the 81-frame clips.
    trim = [] if trim_seconds is None else ["-t", str(trim_seconds)]
    target = work / "audio.m4a"
    ffmpeg("-i", audio, "-map", "0:a:0", "-vn", *trim, "-c:a", "aac", "-b:a", "192k", target)
    return target


def encode_video(work, name, frames, audio, output):
    silent = work / f"{name}.mp4"
    ffmpeg("-framerate", FPS, "-i", work / f"{name}_%06d.png", "-frames:v", frames,
           "-c:v", "libx264", "-qp", "0", "-preset", "veryslow", "-pix_fmt", "yuv420p", silent)
    ffmpeg("-i", silent, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
           "-c", "copy", "-movflags", "+faststart", output)
    video = probe_video(output)
    sound = probe_audio(output)
    encoded_sound = probe_audio(audio)
    if video["frames"] != frames or abs(sound["duration"] - encoded_sound["duration"]) > .002:
        raise ValueError(f"Incomplete video or audio in {output}")
    return {"video": video, "audio": sound, "sha256": sha256(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", required=True, type=Path, help="directory with grid.mp4 and front.mp4")
    parser.add_argument("--audio", required=True, type=Path, help="matching audio, aligned at time zero")
    parser.add_argument("--checkpoint", type=Path,
                        help="checkpoint path; default: checkpoints/AudioLift-gaussianlifter.pt (download if missing)")
    parser.add_argument("--no-download", action="store_true", help="require local weights without downloading")
    parser.add_argument("--out", required=True, type=Path, help="new output directory (must not exist)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--turns", type=int, choices=(1, 2, 3), help="default: 2 for audio <25s, otherwise 3")
    parser.add_argument("--max-frames", type=int, help="explicit smoke run: render a prefix and trim its audio")
    parser.add_argument("--dry-run", action="store_true", help="check checkpoint, config, cameras and media without CUDA")
    args = parser.parse_args(argv)
    try:
        use_release_checkpoint = args.checkpoint is None
        if use_release_checkpoint:
            args.checkpoint = ROOT / "checkpoints" / MODELS["2"][0]
        for key in ("stage1_dir", "audio", "checkpoint", "out"):
            setattr(args, key, getattr(args, key).expanduser().resolve())
        if args.out.exists():
            raise ValueError(f"Output already exists: {args.out}; choose a new directory")
        if args.gpu < 0:
            raise ValueError("--gpu must be non-negative")
        inputs = validate_inputs(args.stage1_dir, args.audio, args.max_frames)
        for relative, expected in PINNED_FILES.items():
            verify_hash(ROOT / relative, expected)
        if use_release_checkpoint and not args.checkpoint.is_file():
            ensure_checkpoint("2", ROOT / "checkpoints", allow_download=not (args.no_download or args.dry_run))
        verify_hash(args.checkpoint, CHECKPOINT_SHA256)
        frames = inputs["frames"]
        turns = args.turns or (2 if inputs["audio"]["duration"] < 25 else 3)
        receipt = {
            "release": "RELEASE-16 / R5@150", "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256, "config": CONFIG,
            "inputs": inputs, "fps": FPS, "turns": turns, "filters": False,
            "precision": "fp16", "input_view_order": ["top_left", "top_right", "bottom_left", "bottom_right", "frontal"],
            "coverage_degrees": [0, 51, 0, -40, 0], "output_size": [832, 1216],
            "smoke_prefix": frames < inputs["grid"]["frames"],
            "input_files": {str(path): sha256(path) for path in
                            (args.stage1_dir / "grid.mp4", args.stage1_dir / "front.mp4", args.audio)},
            "source_sha256": {name: sha256(ROOT / name) for name in
                              (*PINNED_FILES, "stage2/pipeline.py", "stage2/inference.sh",
                               "stage2/view_inputs.py",
                               "stage2/models/gslrm.py", "stage2/rendering.py", "stage2/models/transform_data.py")},
        }
        print(json.dumps(receipt, indent=2), flush=True)
        if args.dry_run:
            print("PREFLIGHT PASSED; no output directory created and no GPU used.")
            return 0
        args.out.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="frames-", dir=args.out) as temporary:
            work = Path(temporary)
            receipt["runtime"] = render_frames(args.stage1_dir, args.checkpoint, work, frames, turns, args.gpu)
            trim = frames / FPS if receipt["smoke_prefix"] else None
            audio = encode_audio(args.audio, work, trim)
            receipt["outputs"] = {name: encode_video(work, name, frames, audio, args.out / f"{name}.mp4")
                                  for name in ("frontal", "coverage", "orbit360")}
        (args.out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Completed {frames} frames: {args.out}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Stage-2 inference failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
