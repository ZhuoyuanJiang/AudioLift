#!/usr/bin/env python3
"""Download AudioLift inference models to their expected local paths."""

import argparse
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "DanJZY/AudioLift"
MODEL_REVISION = "9e1f36db0df254f6d948a3c5f3a5bfc6abad58b1"
MODELS = {
    "1": ("AudioLift-mvdiffusion.pt", "8b871f1344d0edcc06f59b501e02146ed52080544e5992ce32be36d392738e6a"),
    "2": ("AudioLift-gaussianlifter.pt", "dcbef48d163f880133d1bd2adbf46140c6d45d0b9cbcebb12276b3bea5a537dc"),
}
UPSTREAM_REPO = "FrancisRing/StableAvatar"
UPSTREAM_REVISION = "140cbd9e2b7b331898e66a8f07e74e1d04359f2f"
# Check every required file, so a partially downloaded directory is not accepted.
# The PyTorch runtime uses model.safetensors; TensorFlow and duplicate .bin
# weights are not needed. Sizes are from this pinned upstream revision.
UPSTREAM_FILES = {
    "Wan2.1-Fun-V1.1-1.3B-InP/LICENSE.txt": 11357,
    "Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth": 507609880,
    "Wan2.1-Fun-V1.1-1.3B-InP/config.json": 249,
    "Wan2.1-Fun-V1.1-1.3B-InP/configuration.json": 47,
    "Wan2.1-Fun-V1.1-1.3B-InP/diffusion_pytorch_model.safetensors": 3128957992,
    "Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/special_tokens_map.json": 6623,
    "Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/spiece.model": 4548313,
    "Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/tokenizer.json": 16837417,
    "Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/tokenizer_config.json": 61728,
    "Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": 4772359047,
    "Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth": 11361920418,
    "Wan2.1-Fun-V1.1-1.3B-InP/xlm-roberta-large/sentencepiece.bpe.model": 5069051,
    "Wan2.1-Fun-V1.1-1.3B-InP/xlm-roberta-large/special_tokens_map.json": 280,
    "Wan2.1-Fun-V1.1-1.3B-InP/xlm-roberta-large/tokenizer.json": 17082660,
    "Wan2.1-Fun-V1.1-1.3B-InP/xlm-roberta-large/tokenizer_config.json": 418,
    "wav2vec2-base-960h/config.json": 1596,
    "wav2vec2-base-960h/feature_extractor_config.json": 158,
    "wav2vec2-base-960h/model.safetensors": 377607901,
    "wav2vec2-base-960h/preprocessor_config.json": 159,
    "wav2vec2-base-960h/special_tokens_map.json": 85,
    "wav2vec2-base-960h/tokenizer_config.json": 163,
    "wav2vec2-base-960h/vocab.json": 291,
}


def require_download(allow_download, missing):
    offline = os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}
    if not allow_download or offline:
        raise FileNotFoundError(
            f"Missing model assets (downloads disabled): {missing}. "
            "Prepare them with python scripts/download_models.py on a connected machine."
        )


def ensure_checkpoint(stage, checkpoints_dir=ROOT / "checkpoints", *, allow_download=True):
    filename, expected_sha256 = MODELS[str(stage)]
    checkpoints_dir = Path(checkpoints_dir)
    path = checkpoints_dir / filename
    if not path.is_file():
        require_download(allow_download, path)
        from huggingface_hub import hf_hub_download

        print(f"Downloading {filename} from {MODEL_REPO}", flush=True)
        try:
            hf_hub_download(
                repo_id=MODEL_REPO, filename=filename, revision=MODEL_REVISION,
                local_dir=checkpoints_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {filename} from {MODEL_REPO} ({type(exc).__name__}). "
                "Check connectivity and retry; released AudioLift models do not require a Hugging Face login."
            ) from exc
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {digest.hexdigest()}. "
            "The existing file was not overwritten; move it aside before downloading again."
        )
    print(f"Verified {filename}", flush=True)
    return path


def ensure_upstream(checkpoints_dir=ROOT / "checkpoints", *, allow_download=True):
    checkpoints_dir = Path(checkpoints_dir)

    def missing_files():
        return [name for name, size in UPSTREAM_FILES.items()
                if not (checkpoints_dir / name).is_file()
                or (checkpoints_dir / name).stat().st_size != size]

    missing = missing_files()
    if missing:
        require_download(allow_download, ", ".join(missing))
        from huggingface_hub import snapshot_download

        print("Downloading missing Wan and wav2vec2 assets", flush=True)
        try:
            snapshot_download(
                repo_id=UPSTREAM_REPO, revision=UPSTREAM_REVISION,
                local_dir=checkpoints_dir, allow_patterns=missing,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download Wan/wav2vec2 from {UPSTREAM_REPO} ({type(exc).__name__})."
            ) from exc
        missing = missing_files()
        if missing:
            raise ValueError("Incomplete upstream model assets: " + ", ".join(missing))
    print("Wan and wav2vec2 assets are ready", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--check-only", action="store_true", help="verify local assets without downloading")
    components = parser.add_mutually_exclusive_group()
    components.add_argument("--checkpoint-only", action="store_true", help="skip the Stage-1 upstream assets")
    components.add_argument("--upstream-only", action="store_true", help="prepare only Wan and wav2vec2")
    args = parser.parse_args(argv)
    if args.upstream_only and args.stage == "2":
        parser.error("--upstream-only applies to Stage 1")
    try:
        directory = args.checkpoints_dir.expanduser().resolve()
        if not args.upstream_only:
            stages = ("1", "2") if args.stage == "all" else (args.stage,)
            for stage in stages:
                ensure_checkpoint(stage, directory, allow_download=not args.check_only)
        if args.stage in {"1", "all"} and not args.checkpoint_only:
            ensure_upstream(directory, allow_download=not args.check_only)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Model setup failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
