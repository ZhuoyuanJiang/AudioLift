#!/usr/bin/env bash
set -euo pipefail

# Unfiltered full-resolution orbits have large, varying rasterizer buffers.
# Match the training launchers' allocator setting; retain caller overrides.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

STAGE2_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${STAGE2_PYTHON:-python}" "$STAGE2_ROOT/stage2/pipeline.py" "$@"
