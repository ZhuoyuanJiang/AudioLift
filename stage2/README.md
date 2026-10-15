# Stage 2: Gaussian head reconstruction

The five-view reconstructor takes `grid.mp4`, `front.mp4` and matching audio
from Stage 1. It produces frontal, coverage-sweep and full-orbit videos.
Run the entrypoints from the AudioLift repository root:

- Inference: `python inference_stage2.py --help`

See [installation](../docs/installation.md), [checkpoints](../docs/checkpoints.md),
[input preparation](../data/README.md). Reconstruction uses the four grid views
and the separate frontal view with the provided camera calibrations.

CPU regression checks: `python -m unittest discover -s tests/stage2 -v`.

This stage builds on [FaceLift](https://github.com/weijielyu/FaceLift).
The original [source license](LICENSE) and
[Adobe Research License](<Adobe Research License v1.2.txt>) are retained.
