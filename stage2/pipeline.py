#!/usr/bin/env python3
"""RELEASE-16: reconstruct each Stage-1 video frame from five calibrated views.

The model, view preparation and renderer are the existing RELEASE-16 code.
Frame extraction, view preparation and camera paths follow the paper renderer.
Audio delivery follows the paper's complete-audio remux.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_models import MODELS, ensure_checkpoint

FPS = 25
CONFIG = "configs/stage2/inference.yaml"
GRID_CAMERAS = "configs/stage2/cameras/opencv_cameras_nersemble_017_832x1216.json"
FRONT_CAMERA = "configs/stage2/cameras/opencv_cameras_nersemble_frontal_canonical_832x1216.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                    *map(str, args)], check=True)


def probe_video(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No video stream in {path}")
    stream = data["streams"][0]
    frames = int(stream.get("nb_read_frames", 0))
    return {"width": stream["width"], "height": stream["height"], "frames": frames}


def probe_audio(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=duration,start_time,sample_rate", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No audio stream in {path}")
    duration = float(streams[0].get("duration", "nan"))
    return {"duration": duration, "sample_rate": int(streams[0]["sample_rate"])}


def inspect_inputs(stage1_dir, audio):
    """Record media metadata without imposing extra alignment requirements."""
    grid = probe_video(stage1_dir / "grid.mp4")
    front = probe_video(stage1_dir / "front.mp4")
    sound = probe_audio(audio)
    return {"grid": grid, "front": front, "audio": sound, "frames": grid["frames"]}


def gen_frames(stage1_dir, work):
    """Decode both videos as in the paper renderer; the grid determines length."""
    for name, prefix in (("grid", "g"), ("front", "f")):
        ffmpeg("-i", stage1_dir / f"{name}.mp4", "-start_number", "0",
               work / f"{prefix}_%03d.png")
    return len(list(work.glob("g_*.png")))


def cells(img):
    """The paper renderer's grid split, in calibrated camera order."""
    W, H = img.size
    w, h = W // 2, H // 2
    return [img.crop((c * w, r * h, (c + 1) * w, (r + 1) * h)) for r in range(2) for c in range(2)]


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
    from stage2.view_inputs import initialize_gslrm_model, prepare_views
    from stage2.view_inputs import fit_pad, load_cams, build_batch
    from stage2.rendering import render_opencv_cam

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    model = initialize_gslrm_model(str(checkpoint), str(ROOT / CONFIG), device)
    model.eval()
    intrinsics, poses = load_cams(ROOT / GRID_CAMERAS, ROOT / FRONT_CAMERA, device, True)
    camera = json.loads((ROOT / FRONT_CAMERA).read_text())["frames"][0]
    frontal = (np.array(camera["c2w"]) if "c2w" in camera
               else np.linalg.inv(np.array(camera["w2c"]))).astype(np.float32)
    render_width, render_height = int(camera["w"] * 2), int(camera["h"] * 2)
    render_intrinsics = torch.tensor([camera[key] * 2 for key in ("fx", "fy", "cx", "cy")],
                                     dtype=torch.float32, device=device)
    with Image.open(work / "g_000.png") as grid:
        cell_width, cell_height = cells(grid.convert("RGB"))[0].size
    with torch.no_grad():
        for frame in range(frames):
            with Image.open(work / f"g_{frame:03d}.png") as grid:
                views = cells(grid.convert("RGB"))
            with Image.open(work / f"f_{frame:03d}.png") as front:
                views.append(fit_pad(front.convert("RGB"), cell_width, cell_height))
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
                rendered = render_opencv_cam(gaussians, render_height, render_width,
                                             torch.from_numpy(pose).to(device), render_intrinsics)
                rgb = rendered["render"].detach().cpu().permute(1, 2, 0).numpy() * 255
                Image.fromarray(rgb.clip(0, 255).astype("uint8")).save(work / f"{name}_{frame:06d}.png")
            del result, gaussians, batch, rendered
            torch.cuda.empty_cache()
            if frame % 20 == 0 or frame == frames - 1:
                print(f"Reconstructed frame {frame + 1}/{frames}", flush=True)
    return {"torch": torch.__version__, "numpy": np.__version__, "pillow": Image.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "rasterizer_sha256": sha256(rasterizer.__file__),
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}


def encode_audio(audio, work):
    # Encode the full audio independently. Combining -frames:v and -shortest
    # with audio encoding used to lose the audio tail on the 81-frame clips.
    target = work / "audio.m4a"
    ffmpeg("-i", audio, "-vn", "-c:a", "aac", "-b:a", "192k", target)
    return target


def encode_video(work, name, audio, output):
    silent = work / f"{name}.mp4"
    ffmpeg("-framerate", FPS, "-i", work / f"{name}_%06d.png",
           "-c:v", "libx264", "-qp", "0", "-preset", "veryslow", "-pix_fmt", "yuv420p", silent)
    ffmpeg("-i", silent, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
           "-c", "copy", output)
    video = probe_video(output)
    sound = probe_audio(output)
    return {"video": video, "audio": sound, "sha256": sha256(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", required=True, type=Path, help="directory with grid.mp4 and front.mp4")
    parser.add_argument("--audio", required=True, type=Path, help="the original Stage-1 driving audio")
    parser.add_argument("--checkpoint", type=Path,
                        help="checkpoint path; default: checkpoints/AudioLift-gaussianlifter.pt (download if missing)")
    parser.add_argument("--no-download", action="store_true", help="require local weights without downloading")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--turns", type=float, help="default: 2 for audio <25s, otherwise 3")
    parser.add_argument("--dry-run", action="store_true", help="print media and model paths without CUDA")
    args = parser.parse_args(argv)
    try:
        use_release_checkpoint = args.checkpoint is None
        if use_release_checkpoint:
            args.checkpoint = ROOT / "checkpoints" / MODELS["2"][0]
        for key in ("stage1_dir", "audio", "checkpoint", "out"):
            setattr(args, key, getattr(args, key).expanduser().resolve())
        outputs = {name: args.out / f"{name}.mp4" for name in ("frontal", "coverage", "orbit360")}
        if not args.dry_run and all(path.is_file() for path in outputs.values()):
            print(f"EXISTS {args.out}")
            return 0
        inputs = inspect_inputs(args.stage1_dir, args.audio)
        if use_release_checkpoint and not args.checkpoint.is_file():
            ensure_checkpoint("2", ROOT / "checkpoints", allow_download=not (args.no_download or args.dry_run))
        turns = args.turns if args.turns is not None else (2 if inputs["audio"]["duration"] < 25 else 3)
        camera = json.loads((ROOT / FRONT_CAMERA).read_text())["frames"][0]
        receipt = {
            "release": "RELEASE-16 / R5@150", "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint), "config": CONFIG,
            "inputs": inputs, "fps": FPS, "turns": turns, "filters": False,
            "precision": "fp16", "input_view_order": ["top_left", "top_right", "bottom_left", "bottom_right", "frontal"],
            "coverage_degrees": [0, 51, 0, -40, 0],
            "output_size": [int(camera["w"] * 2), int(camera["h"] * 2)],
            "input_files": {str(path): sha256(path) for path in
                            (args.stage1_dir / "grid.mp4", args.stage1_dir / "front.mp4", args.audio)},
            "source_sha256": {name: sha256(ROOT / name) for name in
                              (CONFIG, GRID_CAMERAS, FRONT_CAMERA, "stage2/pipeline.py", "stage2/inference.sh",
                               "stage2/view_inputs.py",
                               "stage2/models/gslrm.py", "stage2/rendering.py", "stage2/models/transform_data.py")},
        }
        print(json.dumps(receipt, indent=2), flush=True)
        if args.dry_run:
            print("Dry run complete; no output directory created and no GPU used.")
            return 0
        args.out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="frames-", dir=args.out) as temporary:
            work = Path(temporary)
            frames = gen_frames(args.stage1_dir, work)
            if frames < 10:
                print("NO_FRAMES")
                return 1
            receipt["frames"] = frames
            receipt["runtime"] = render_frames(args.stage1_dir, args.checkpoint, work, frames, turns, args.gpu)
            audio = encode_audio(args.audio, work)
            receipt["outputs"] = {name: encode_video(work, name, audio, path)
                                  for name, path in outputs.items()}
        (args.out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Completed {frames} frames: {args.out}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Stage-2 inference failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
