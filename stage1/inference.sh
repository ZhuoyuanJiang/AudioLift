#!/usr/bin/env bash
# Canonical five-view Stage-1 inference: one reference image + one audio track
# -> four-view grid, separate frontal stream, and a side-by-side preview.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FINAL_STAGE1_SHA256="8b871f1344d0edcc06f59b501e02146ed52080544e5992ce32be36d392738e6a"
FINAL_STAGE1_RELATIVE_PATH="checkpoints/AudioLift-mvdiffusion.pt"

usage() {
  cat <<'EOF'
Canonical five-view Stage-1 inference

Usage:
  bash stage1/inference.sh \
    --reference photo.jpg --audio speech.wav --out OUTPUT_DIR [options]

Required:
  --reference PATH   Reference portrait. The default gate requires the Stage-1
                     training distribution: 832x1216 portrait framing with the
                     person matted over white.
  --audio PATH       Driving speech. Output length follows the audio.
  --out DIR          Output directory.

Options:
  --checkpoint PATH  Final Stage-1 transformer checkpoint. If omitted, search
                     checkpoints/AudioLift-mvdiffusion.pt in this repository;
                     download the released model if missing.
  --no-download      Require local assets; do not download missing models.
  --gpu N            CUDA device index (default: 0).
  --steps N          Denoising steps (default: 30).
  --seed N           Sampling seed (default: 42).
  --no_side_by_side_preview
                     Write only grid.mp4 and front.mp4.
  --dry_run          Run the reference gate, print the complete window plan and
                     cost estimate, then stop before loading Stage-1 models.
  --skip_reference_check
                     Bypass the gate for an already audited frozen dataset.
  --allow_noncompliant_reference
                     Deliberately run the original rejected image. A/B tests
                     only; output may be invalid.
  --allow_noncanonical_checkpoint
                     Permit a checkpoint whose SHA-256 differs from the frozen
                     release. A/B tests only; the mismatch is printed loudly.
  -h, --help         Show this help.

Reference refusal:
  A rejected image is never replaced silently. The gate writes
    OUTPUT_DIR/_gate_candidate/<reference_stem>_face.png
  when repair succeeds, prints the path, and exits 3 before model loading. Open
  that candidate, inspect hair/headwear/framing/shoulders, and pass it explicitly
  as --reference in a new command if it is acceptable.

Environment overrides:
  CKPT                         Same as --checkpoint.
  STAGE1_ASSET_ROOT            Root containing checkpoints/, configs/
                               (default: this repository).
  STAGE1_PYTHON or PY          Python interpreter.
  ALLOW_NONCANONICAL_STAGE1_CKPT=1
                               Same as --allow_noncanonical_checkpoint.

Frozen release checkpoint:
  checkpoints/AudioLift-mvdiffusion.pt
  SHA-256 8b871f1344d0edcc06f59b501e02146ed52080544e5992ce32be36d392738e6a
EOF
}

die() {
  echo "[stage1-5view] ERROR: $*" >&2
  exit 1
}

REFERENCE=""
AUDIO=""
OUT=""
GPU="0"
STEPS="30"
SEED="42"
CKPT="${CKPT:-}"
DRY_RUN=0
NO_DOWNLOAD=0
ALLOW_NONCANONICAL="${ALLOW_NONCANONICAL_STAGE1_CKPT:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference) [[ $# -ge 2 ]] || die "$1 requires a value"; REFERENCE="$2"; shift 2 ;;
    --audio) [[ $# -ge 2 ]] || die "$1 requires a value"; AUDIO="$2"; shift 2 ;;
    --out) [[ $# -ge 2 ]] || die "$1 requires a value"; OUT="$2"; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || die "$1 requires a value"; CKPT="$2"; shift 2 ;;
    --gpu) [[ $# -ge 2 ]] || die "$1 requires a value"; GPU="$2"; shift 2 ;;
    --steps) [[ $# -ge 2 ]] || die "$1 requires a value"; STEPS="$2"; shift 2 ;;
    --seed) [[ $# -ge 2 ]] || die "$1 requires a value"; SEED="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; EXTRA_ARGS+=(--dry_run); shift ;;
    --no-download) NO_DOWNLOAD=1; shift ;;
    --skip_reference_check) EXTRA_ARGS+=(--skip_reference_check); shift ;;
    --allow_noncompliant_reference) EXTRA_ARGS+=(--allow_noncompliant_reference); shift ;;
    --allow_noncanonical_checkpoint) ALLOW_NONCANONICAL=1; shift ;;
    --no_side_by_side_preview) EXTRA_ARGS+=(--no_side_by_side_preview); shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$REFERENCE" ]] || die "--reference is required"
[[ -n "$AUDIO" ]] || die "--audio is required"
[[ -n "$OUT" ]] || die "--out is required"
[[ -f "$REFERENCE" ]] || die "reference not found: $REFERENCE"
[[ -f "$AUDIO" ]] || die "audio not found: $AUDIO"

PYTHON_BIN="${STAGE1_PYTHON:-${PY:-python}}"
DOWNLOAD_ARGS=()
if [[ "$NO_DOWNLOAD" == "1" || "$DRY_RUN" == "1" ]]; then
  DOWNLOAD_ARGS+=(--check-only)
fi
if [[ -z "$CKPT" ]]; then
  CKPT="${REPO_ROOT}/${FINAL_STAGE1_RELATIVE_PATH}"
  if [[ ! -f "$CKPT" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/download_models.py" \
      --stage 1 --checkpoint-only "${DOWNLOAD_ARGS[@]}"
  fi
fi

[[ -n "$CKPT" && -f "$CKPT" ]] || die "final Stage-1 checkpoint not found.
Pass --checkpoint PATH (or CKPT=PATH). Expected ${FINAL_STAGE1_RELATIVE_PATH}."
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required to verify the checkpoint"

actual_sha256="$(sha256sum "$CKPT" | awk '{print $1}')"
if [[ "$actual_sha256" != "$FINAL_STAGE1_SHA256" ]]; then
  if [[ "$ALLOW_NONCANONICAL" == "1" ]]; then
    echo "[stage1-5view] WARNING: non-canonical checkpoint explicitly allowed" >&2
    echo "[stage1-5view] expected SHA-256: $FINAL_STAGE1_SHA256" >&2
    echo "[stage1-5view] actual   SHA-256: $actual_sha256" >&2
  else
    die "checkpoint SHA-256 mismatch.
Expected: ${FINAL_STAGE1_SHA256}
Actual:   ${actual_sha256}
Path:     ${CKPT}
Use --allow_noncanonical_checkpoint only for an intentional A/B test."
  fi
fi

# Resolve caller-relative paths before running from the repository root.
REFERENCE="$(realpath -- "$REFERENCE")"
AUDIO="$(realpath -- "$AUDIO")"
CKPT="$(realpath -- "$CKPT")"
OUT="$(realpath -m -- "$OUT")"
ASSET_ROOT="$(realpath -m -- "${STAGE1_ASSET_ROOT:-$REPO_ROOT}")"
if [[ -n "${STAGE1_PYTHON:-}" ]]; then
  PYTHON_BIN="$STAGE1_PYTHON"
elif [[ -n "${PY:-}" ]]; then
  PYTHON_BIN="$PY"
else
  PYTHON_BIN="python"
fi

CONFIG_PATH="${ASSET_ROOT}/configs/stage1/wan_civitai.yaml"
PRETRAINED_MODEL="${ASSET_ROOT}/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
WAV2VEC_PATH="${ASSET_ROOT}/checkpoints/wav2vec2-base-960h"

[[ -f "$CONFIG_PATH" ]] || die "config not found: $CONFIG_PATH (set STAGE1_ASSET_ROOT)"
if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/download_models.py" \
    --stage 1 --upstream-only --checkpoints-dir "$ASSET_ROOT/checkpoints" \
    "${DOWNLOAD_ARGS[@]}"
  [[ -e "$PRETRAINED_MODEL" ]] || die "pretrained model not found: $PRETRAINED_MODEL"
  [[ -e "$WAV2VEC_PATH" ]] || die "wav2vec model not found: $WAV2VEC_PATH"
fi

mkdir -p "$OUT"
echo "[stage1-5view] repository : $REPO_ROOT"
echo "[stage1-5view] assets     : $ASSET_ROOT"
echo "[stage1-5view] reference  : $REFERENCE"
echo "[stage1-5view] audio      : $AUDIO"
echo "[stage1-5view] checkpoint : $CKPT"
echo "[stage1-5view] ckpt sha256: $actual_sha256"
echo "[stage1-5view] output     : $OUT"
echo "[stage1-5view] sampling   : gpu=$GPU steps=$STEPS seed=$SEED"

CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/stage1/sampling.py"
  --config_path "$CONFIG_PATH"
  --pretrained_model_name_or_path "$PRETRAINED_MODEL"
  --transformer_path "$CKPT"
  --pretrained_wav2vec_path "$WAV2VEC_PATH"
  --validation_reference_path "$REFERENCE"
  --validation_driven_audio_path "$AUDIO"
  --output_dir "$OUT"
  --height 1216
  --width 832
  --num_inference_steps "$STEPS"
  --s76_variant selfattn
  --front_stream_mode random
  --frames_per_batch 81
  --overlap_window_length 5
  --seed "$SEED"
  "${EXTRA_ARGS[@]}"
)

if (
  cd "$REPO_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false "${CMD[@]}"
); then
  shopt -s nullglob
  outputs=("$OUT"/*.mp4)
  echo "[stage1-5view] done -> ${#outputs[@]} mp4 file(s) in $OUT"
else
  rc=$?
  echo "[stage1-5view] FAILED (exit $rc)" >&2
  exit "$rc"
fi
