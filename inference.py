#!/usr/bin/env python3
"""Run the existing Stage-1 and Stage-2 entrypoints in separate environments."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def python_for_stage(stage, override):
    executable = override or os.environ.get(f"STAGE{stage}_PYTHON")
    executable = executable or str(ROOT / f".venv-stage{stage}" / "bin" / "python")
    executable = os.path.expanduser(executable)
    if os.sep not in executable:
        executable = shutil.which(executable) or executable
    # Preserve the venv's Python symlink: resolving it can select the base environment.
    executable = os.path.abspath(executable)
    if not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise ValueError(
            f"Stage {stage} Python not found or not executable: {executable}. "
            f"Install .venv-stage{stage} (see docs/installation.md), or pass "
            f"--stage{stage}-python /path/to/python."
        )
    return executable


def cuda_device(index):
    if index < 0:
        raise ValueError("--gpu must be non-negative")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    devices = [device.strip() for device in visible.split(",")]
    if index >= len(devices) or any(device in ("", "-1") for device in devices):
        raise ValueError(f"--gpu {index} is unavailable in CUDA_VISIBLE_DEVICES={visible!r}")
    return devices[index]


def run_stage(stage, command, env):
    print(f"[AudioLift] Stage {stage}: {shlex.join(command)}", flush=True)
    code = subprocess.run(command, env=env).returncode
    return code if code >= 0 else 128 - code


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate five views, then reconstruct and render a Gaussian head.",
        epilog="Each stage uses its own Python environment; no manual activation is needed.",
    )
    parser.add_argument("--reference", required=True, type=Path, help="reference portrait")
    parser.add_argument("--audio", required=True, type=Path, help="driving speech recording")
    parser.add_argument("--out", required=True, type=Path, help="new output directory (must not exist)")
    for stage in (1, 2):
        parser.add_argument(
            f"--stage{stage}-python",
            help=f"Python executable (default: STAGE{stage}_PYTHON or .venv-stage{stage}/bin/python)",
        )
        parser.add_argument(f"--stage{stage}-checkpoint", type=Path, help=f"explicit Stage-{stage} weights")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index within CUDA_VISIBLE_DEVICES if set, otherwise CUDA device index (default: 0)")
    parser.add_argument("--steps", type=int, default=30, help="Stage-1 denoising steps (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Stage-1 sampling seed (default: 42)")
    parser.add_argument("--turns", type=int, choices=(1, 2, 3), help="Stage-2 orbit turns (default: based on audio duration)")
    parser.add_argument("--no-download", action="store_true", help="require local checkpoints and base-model assets")
    args = parser.parse_args(argv)

    try:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        for name in ("reference", "audio", "stage1_checkpoint", "stage2_checkpoint"):
            path = getattr(args, name)
            if path is not None:
                path = path.expanduser().resolve()
                if not path.is_file():
                    raise ValueError(f"--{name.replace('_', '-')} file not found: {path}")
                setattr(args, name, path)
        args.out = args.out.expanduser().absolute()
        if os.path.lexists(args.out):
            raise ValueError(f"Output already exists: {args.out}. Choose a new --out directory.")
        stage1_python = python_for_stage(1, args.stage1_python)
        stage2_python = python_for_stage(2, args.stage2_python)
        device = cuda_device(args.gpu)
        env = dict(os.environ, STAGE1_PYTHON=stage1_python, STAGE2_PYTHON=stage2_python,
                   CUDA_VISIBLE_DEVICES=device)
        stage1_dir = args.out / "stage1"
        stage2_dir = args.out / "stage2"
        stage1 = [stage1_python, str(ROOT / "inference_stage1.py"),
                  "--reference", str(args.reference), "--audio", str(args.audio),
                  "--out", str(stage1_dir), "--gpu", device,
                  "--steps", str(args.steps), "--seed", str(args.seed)]
        # Both processes see the same single GPU. Stage 1's launcher sets its own
        # visibility mask; Stage 2 addresses device 0 within that mask.
        stage2 = [stage2_python, str(ROOT / "inference_stage2.py"),
                  "--stage1-dir", str(stage1_dir), "--audio", str(args.audio),
                  "--out", str(stage2_dir), "--gpu", "0"]
        for command, checkpoint in ((stage1, args.stage1_checkpoint), (stage2, args.stage2_checkpoint)):
            if checkpoint is not None:
                command.extend(["--checkpoint", str(checkpoint)])
            if args.no_download:
                command.append("--no-download")
        if args.turns is not None:
            stage2.extend(["--turns", str(args.turns)])

        code = run_stage(1, stage1, env)
        if code:
            print(f"[AudioLift] Stage 1 stopped (exit {code}); Stage 2 was not started.", file=sys.stderr)
            if code == 3:
                print(
                    f"Inspect the proposed portrait in {stage1_dir / '_gate_candidate'}. "
                    "If acceptable, rerun with that image as --reference and a new --out directory.",
                    file=sys.stderr,
                )
            return code
        for name in ("grid.mp4", "front.mp4"):
            path = stage1_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Stage 1 did not produce a non-empty {path}; Stage 2 was not started.")

        code = run_stage(2, stage2, env)
        if code:
            print(
                f"[AudioLift] Stage 2 stopped (exit {code}). Stage-1 videos remain in {stage1_dir}. "
                "Use inference_stage2.py to retry with a new Stage-2 output directory.",
                file=sys.stderr,
            )
            return code
        print(f"[AudioLift] Done: {stage2_dir} (frontal.mp4, coverage.mp4, orbit360.mp4)")
        return 0
    except (OSError, ValueError) as error:
        print(f"[AudioLift] ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[AudioLift] Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
