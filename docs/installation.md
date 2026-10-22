# Installation

Run commands from the repository root on Linux with Python 3.10, an NVIDIA
GPU and FFmpeg/ffprobe. Each stage uses its own Python environment.
Use a GPU with at least 48 GB VRAM for the default five-view settings.
The unified launcher runs both stages sequentially on that one GPU.

## Stage 1

```bash
python3.10 -m venv .venv-stage1
source .venv-stage1/bin/activate
python -m pip install -r requirements-stage1.txt
python -m pip check
```

This environment uses PyTorch 2.7.0 with CUDA 12.6 wheels. The environment
versions are pinned separately from Stage 2. FlashAttention and xFuser are
optional and are not required by the provided single-GPU inference entrypoint.

Reference preprocessing uses MediaPipe and Robust Video Matting. Download the
Face Landmarker bundle linked from the
[MediaPipe model guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
to `external/mediapipe_models/face_landmarker.task`. The RVM matting model is
loaded and cached by PyTorch Hub on first use from
[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting).

## Stage 2

```bash
python3.10 -m venv .venv-stage2
source .venv-stage2/bin/activate
python -m pip install -r requirements-stage2.txt
```

Build the pinned rasterizer using a CUDA 12.4 toolkit containing `bin/nvcc`:

```bash
export CUDA_HOME=/path/to/cuda-12.4
TORCH_CUDA_ARCH_LIST="8.6;8.9" python -m pip install \
  --no-build-isolation --no-deps \
  git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
python -m pip check
```

The example architecture list covers RTX 3090 and RTX 6000 Ada. For an A100,
use `TORCH_CUDA_ARCH_LIST="8.0"` in the build command instead. Set this list for
the GPU you will use. A Conda CUDA toolkit may also need:

```bash
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

Set these before building the extension. The pip CUDA libraries alone do not
provide the compiler.

After both environments are installed, `python inference.py` selects
`.venv-stage1/bin/python` and `.venv-stage2/bin/python` automatically and runs the
stages in sequence. For environments elsewhere, pass `--stage1-python` and
`--stage2-python`, or set `STAGE1_PYTHON` / `STAGE2_PYTHON` to their Python
executables. See the [inference command](../README.md#inference). The individual
`inference_stage1.py` and `inference_stage2.py` entrypoints still require the
appropriate active environment or Python override.

## Checking a Colab GPU

In a GPU notebook, first check the assigned device and its memory:

```python
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

[A100 GPUs have 40 GB and 80 GB variants](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf),
and [Colab hardware availability varies](https://research.google.com/colaboratory/faq.html#resource-limits).
The default Stage-1 run reached about 44.2 GiB on an RTX 6000 Ada before an
external interruption. A 40 GB device has not been validated for this workflow;
the A100 name alone does not establish that it has enough memory.

Use Python 3.10 environments and the pinned dependencies above, even if the
notebook's default Python differs. The Stage-2 rasterizer needs the CUDA 12.4
toolkit and the A100 architecture setting above. Model downloads currently
require access to the private Hugging Face repository; see
[checkpoint access](checkpoints.md).

## Validation status

Both pinned environments were installed from scratch. Stage 2 completed all
81 frames and all three output videos on one RTX 3090, using saved inputs from
the evaluated Stage-1 workflow; the full audio was preserved. Gaussian outputs
and rendered images matched the original implementation exactly at three
checked frames on the same GPU. The complete unified Stage-1-to-Stage-2 run is
still pending: its Stage-1 process received SIGTERM after 7 of 30 denoising steps.
This release has not yet completed an A100 or Colab run.
