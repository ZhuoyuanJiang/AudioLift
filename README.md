# AudioLift

**Audio-Driven Full-Head Gaussian Avatar via Multi-view Video Generation**

Generate an animated 3D Gaussian head from a portrait and a speech recording.
Stage 1 generates five synchronized views; Stage 2 reconstructs and renders
an animated head from those views.

> This release provides inference code. Training code is not included.
> Model downloads currently require an authorized Hugging Face login because
> the model repository remains private.

## Installation

Use Linux, Python 3.10, an NVIDIA GPU and FFmpeg. The two stages have separate
Python environments within this repository:

```bash
python3.10 -m venv .venv-stage1
source .venv-stage1/bin/activate
python -m pip install -r requirements-stage1.txt
```

```bash
python3.10 -m venv .venv-stage2
source .venv-stage2/bin/activate
python -m pip install -r requirements-stage2.txt
```

Stage 2 also needs the pinned CUDA rasterizer. See the
[installation guide](docs/installation.md) for its build command and the
reference-image preprocessing models.

## Model checkpoints

The inference entrypoints automatically download missing models from
[DanJZY/AudioLift](https://huggingface.co/DanJZY/AudioLift):

- **AudioLift-mvdiffusion** generates the five synchronized video views.
- **AudioLift-gaussianlifter** reconstructs the Gaussian head.

Weights are saved in `checkpoints/` and reused on subsequent runs. Stage 1 also
downloads the required Wan and wav2vec2 assets from their upstream repository.
To download everything ahead of time, run `python scripts/download_models.py`.
See [model paths, private-review access and checksums](docs/checkpoints.md).

## Inference

After installing both environments, run both stages with one command from the
repository root:

```bash
python inference.py \
  --reference /path/to/portrait.png \
  --audio /path/to/speech.wav \
  --out outputs/example
```

The launcher uses `.venv-stage1/bin/python` and `.venv-stage2/bin/python`
automatically. You do not need to activate or switch environments between stages.
It runs the existing Stage-1 entrypoint, waits for it to finish, then passes
`grid.mp4`, `front.mp4` and the same audio to the existing Stage-2 entrypoint.
The model, sampling and rendering implementations are unchanged.

Use a **new output directory** for each run. The main outputs are:

```text
outputs/example/
├── stage1/
│   ├── grid.mp4       Four synchronized views in a 2×2 grid
│   └── front.mp4      Separate frontal view
└── stage2/
    ├── frontal.mp4   Frontal rendering with audio
    ├── coverage.mp4  Camera sweep with audio
    ├── orbit360.mp4  Full orbit with audio
    └── run.json      Input, checkpoint and rendering details
```

The reference should be a white-background 832×1216 portrait with the expected
face framing. If Stage 1 proposes a preprocessed replacement, inference stops
with exit code 3 before loading the generation models. Inspect the image under
`outputs/example/stage1/_gate_candidate/`. If it is acceptable, rerun with that
image as `--reference` and a new `--out` directory. The launcher never accepts a
replacement for you. Output duration follows the audio.

For Conda or environments installed elsewhere, specify their Python executables:

```bash
python inference.py \
  --reference /path/to/portrait.png --audio /path/to/speech.wav \
  --out outputs/example \
  --stage1-python /path/to/stage1/env/bin/python \
  --stage2-python /path/to/stage2/env/bin/python
```

`STAGE1_PYTHON` and `STAGE2_PYTHON` are also supported; explicit command-line
options take precedence. `--gpu 0` selects the first visible GPU for both stages
(within `CUDA_VISIBLE_DEVICES` when set). Optional controls include `--steps`,
`--seed`, `--turns`, `--stage1-checkpoint`, `--stage2-checkpoint` and
`--no-download`. Run `python inference.py --help` for details.

<details>
<summary>Run a stage separately or retry Stage 2</summary>

The individual entrypoints remain available. To generate only the five views:

```bash
source .venv-stage1/bin/activate
python inference_stage1.py \
  --reference /path/to/portrait.png --audio /path/to/speech.wav \
  --out outputs/example/stage1
```

To reconstruct and render from completed Stage-1 videos:

```bash
source .venv-stage2/bin/activate
python inference_stage2.py \
  --stage1-dir outputs/example/stage1 --audio /path/to/speech.wav \
  --out outputs/example/stage2
```

Use synchronized 25 fps inputs, matching audio starting at time zero, and a
new Stage-2 output directory. Add `--dry-run` to check Stage-2 inputs and assets
before using a GPU. Dry runs require local weights and do not download them.

If Stage 2 fails during a combined run, the Stage-1 videos remain available.
Retry with `inference_stage2.py`, the same audio, and a new output directory
such as `outputs/example/stage2-retry`; there is no need to regenerate Stage 1.

</details>

## Repository layout

```text
inference.py                    Both stages in one command
inference_stage1.py              Stage 1 only
inference_stage2.py              Stage 2 only
requirements-stage1.txt          Stage-1 environment
requirements-stage2.txt          Stage-2 environment
stage1/
  models/                       Video model components
  dist/                         Optional distributed inference support
  utils/                        Video, latent and scheduler helpers
stage2/
  models/                       Gaussian reconstruction model
configs/
  stage1/                       Wan settings and reference framing
  stage2/                       Reconstruction settings and calibrated cameras
data/                           Local portrait and audio inputs
scripts/
  download_models.py            Model download and verification
  preprocess/                   Reference portrait preparation
tests/                          Inference and download regression checks
docs/                           Installation and checkpoint details
```

Each stage owns its model code under `models/` and its inference pipeline next
to it. `configs/` holds configuration values: model settings, expected portrait
framing and camera parameters. The trained weights live in `checkpoints/`.
`stage1/dist/` means **distributed computation**, specifically optional multi-GPU
support; the default single-GPU command does not require xFuser.

See [input preparation](data/README.md) for the portrait and audio layout. Only
`data/README.md` is tracked in Git; local inputs, downloaded weights and generated
outputs stay in the ignored `data/`, `checkpoints/` and `outputs/` directories.
`tests/` contains checks for developers and is not needed to run inference.

<details>
<summary>File-by-file guide</summary>

### Root entrypoints, environments and notices

| File | Purpose |
| --- | --- |
| [inference.py](inference.py) | Selects each stage's Python environment, runs the stages in sequence, and passes Stage-1 videos to Stage 2. |
| [inference_stage1.py](inference_stage1.py) | Small Python entrypoint that forwards arguments to `stage1/inference.sh`. |
| [inference_stage2.py](inference_stage2.py) | Small Python entrypoint that forwards arguments to `stage2/inference.sh`. |
| [requirements-stage1.txt](requirements-stage1.txt) | Python dependencies for video generation and portrait preprocessing. |
| [requirements-stage2.txt](requirements-stage2.txt) | Python dependencies for Gaussian reconstruction and rendering. |
| [.gitignore](.gitignore) | Keeps local media, model weights, environments, outputs and caches out of Git. |
| [README.md](README.md) | Project overview, installation, inference commands and this file guide. |
| [LICENSE.txt](LICENSE.txt) | Retained Stage-1 upstream MIT license notice. |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Upstream code and model sources, with their license references. |

### Configuration, data and setup documentation

| File | Purpose |
| --- | --- |
| [configs/stage1/wan_civitai.yaml](configs/stage1/wan_civitai.yaml) | Wan component settings, asset subpaths and diffusion scheduler configuration. |
| [configs/stage1/nersemble_framing_calibration.json](configs/stage1/nersemble_framing_calibration.json) | Target face size and position used when preparing a reference portrait. |
| [configs/stage2/inference.yaml](configs/stage2/inference.yaml) | Gaussian reconstruction model architecture and inference settings. |
| [configs/stage2/cameras/opencv_cameras_nersemble_017_832x1216.json](configs/stage2/cameras/opencv_cameras_nersemble_017_832x1216.json) | Calibrated intrinsics and poses for the four views in the generated grid. |
| [configs/stage2/cameras/opencv_cameras_nersemble_frontal_canonical_832x1216.json](configs/stage2/cameras/opencv_cameras_nersemble_frontal_canonical_832x1216.json) | Camera calibration for the separate frontal view, the fifth reconstruction input. |
| [data/README.md](data/README.md) | Local portrait/audio layout and the required dimensions, frame rate and alignment of intermediate videos. |
| [docs/installation.md](docs/installation.md) | Environment installation, CUDA rasterizer build and preprocessing model setup. |
| [docs/checkpoints.md](docs/checkpoints.md) | Model sources, authentication, download commands, local paths and checksums. |

### Download and preprocessing helpers

| File | Purpose |
| --- | --- |
| [scripts/download_models.py](scripts/download_models.py) | Downloads the two AudioLift checkpoints and Stage-1 base assets, and verifies the pinned files. |
| [scripts/preprocess/preprocess_reference.py](scripts/preprocess/preprocess_reference.py) | Checks portrait framing; uses matting and face landmarks to propose a white-background, correctly framed reference when needed. |
| [scripts/__init__.py](scripts/__init__.py) | Makes the download helpers importable as a Python package. |

### Stage 1: five-view video generation

| File | Purpose |
| --- | --- |
| [stage1/inference.sh](stage1/inference.sh) | Checks inputs and checkpoint identity, obtains missing assets, selects the GPU and launches sampling. |
| [stage1/sampling.py](stage1/sampling.py) | Main five-view inference loop: audio-aligned windows, denoising, overlapping-window handling, streaming decoding and video output. |
| [stage1/pipeline.py](stage1/pipeline.py) | Holds the Wan model components and shared prompt, latent and conditioning helpers used by sampling. |
| [stage1/models/wan_fantasy_transformer3d_1B.py](stage1/models/wan_fantasy_transformer3d_1B.py) | Audio-conditioned video transformer used by the released Stage-1 model. |
| [stage1/models/vocal_projector_fantasy_1B.py](stage1/models/vocal_projector_fantasy_1B.py) | Projects audio features into conditioning for that transformer. |
| [stage1/models/vocal_projector_fantasy.py](stage1/models/vocal_projector_fantasy.py) | Upstream audio components, including sequence slicing and padding helpers reused by the 1B projector. |
| [stage1/models/wan_transformer3d.py](stage1/models/wan_transformer3d.py) | Wan backbone definitions and shared attention code used by other components. The generation loop uses the audio-conditioned model above. |
| [stage1/models/wan_image_encoder.py](stage1/models/wan_image_encoder.py) | CLIP image encoder for reference-image features. |
| [stage1/models/wan_text_encoder.py](stage1/models/wan_text_encoder.py) | T5 text encoder for prompt features. |
| [stage1/models/wan_xlm_roberta.py](stage1/models/wan_xlm_roberta.py) | XLM-RoBERTa component of the upstream CLIP implementation. |
| [stage1/models/wan_vae.py](stage1/models/wan_vae.py) | Encodes images/video into latents and decodes generated latents into frames. |
| [stage1/models/cache_utils.py](stage1/models/cache_utils.py) | Optional TeaCache acceleration helpers; not enabled by the default inference command. |
| [stage1/dist/__init__.py](stage1/dist/__init__.py) | Loads optional xFuser support and initializes distributed GPU execution when requested. |
| [stage1/dist/wan_xfuser.py](stage1/dist/wan_xfuser.py) | Attention and positional-encoding helpers for distributed Wan inference. |
| [stage1/utils/utils.py](stage1/utils/utils.py) | Shared image/video saving and reference-latent preparation helpers. |
| [stage1/utils/fm_solvers.py](stage1/utils/fm_solvers.py) | Retained Flow DPM solver and timestep helpers. Default sampling uses the Diffusers FlowMatch Euler scheduler. |
| [stage1/utils/fm_solvers_unipc.py](stage1/utils/fm_solvers_unipc.py) | Retained alternative Flow UniPC scheduler. |
| [stage1/utils/__init__.py](stage1/utils/__init__.py) | Exposes the retained solver helpers to Python imports. |
| [stage1/__init__.py](stage1/__init__.py) | Marks Stage 1 as a Python package. |
| [stage1/models/__init__.py](stage1/models/__init__.py) | Marks the Stage-1 model directory as a Python package. |

### Stage 2: Gaussian reconstruction and rendering

| File | Purpose |
| --- | --- |
| [stage2/inference.sh](stage2/inference.sh) | Launches the pipeline with the Stage-2 Python environment and memory/thread settings. |
| [stage2/pipeline.py](stage2/pipeline.py) | Validates videos and audio, reconstructs each frame, renders the camera trajectories, combines videos with audio, and records `run.json`. |
| [stage2/view_inputs.py](stage2/view_inputs.py) | Splits the four-view grid, adds the frontal view, loads calibrated cameras, builds model inputs and loads reconstruction weights. |
| [stage2/models/gslrm.py](stage2/models/gslrm.py) | Core Gaussian reconstruction network: converts the five images and their camera information into Gaussian parameters. |
| [stage2/models/transform_data.py](stage2/models/transform_data.py) | Converts camera and image information into the input representations and rays used by the network. |
| [stage2/models/utils_transformer.py](stage2/models/utils_transformer.py) | Attention, MLP and transformer blocks for reconstruction. |
| [stage2/rendering.py](stage2/rendering.py) | Gaussian representation, camera helpers and Gaussian rasterization used to render images. |
| [stage2/README.md](stage2/README.md) | Short Stage-2 usage guide and upstream references. |
| [stage2/LICENSE](stage2/LICENSE) | Retained upstream source license. |
| [stage2/Adobe Research License v1.2.txt](<stage2/Adobe Research License v1.2.txt>) | Retained Adobe Research license for the corresponding upstream components. |
| [stage2/__init__.py](stage2/__init__.py) | Marks Stage 2 as a Python package. |
| [stage2/models/__init__.py](stage2/models/__init__.py) | Marks the Stage-2 model directory as a Python package. |

### Regression checks

These are software checks, not a training or evaluation dataset. They verify
input handling and launch behavior without running full GPU generation.

| File | Purpose |
| --- | --- |
| [tests/test_inference.py](tests/test_inference.py) | Checks environment selection, stage ordering, video handoff, GPU mapping, failure propagation and output protection using subprocess stand-ins. |
| [tests/test_download_models.py](tests/test_download_models.py) | Checks model download, checksum verification and offline/missing-asset behavior. |
| [tests/stage1/test_stage1_entrypoints.py](tests/stage1/test_stage1_entrypoints.py) | Checks path handling, checkpoint rejection and reference-gate exit status. |
| [tests/stage2/test_stage2_infer.py](tests/stage2/test_stage2_infer.py) | Checks video geometry, frame counts, audio alignment, camera trajectories and preflight behavior. |
| [tests/__init__.py](tests/__init__.py) | Makes the top-level tests importable by the test runner. |
| [tests/stage1/__init__.py](tests/stage1/__init__.py) | Marks the Stage-1 test package. |
| [tests/stage2/__init__.py](tests/stage2/__init__.py) | Marks the Stage-2 test package. |

The main execution path is:

```text
inference.py
  → inference_stage1.py → stage1/inference.sh → stage1/sampling.py
      → grid.mp4 + front.mp4
  → inference_stage2.py → stage2/inference.sh → stage2/pipeline.py
      → frontal.mp4 + coverage.mp4 + orbit360.mp4
```

</details>

## Acknowledgements

Built on [StableAvatar](https://github.com/Francis-Rings/StableAvatar),
[FaceLift](https://github.com/weijielyu/FaceLift) and their dependencies.
See [third-party notices](THIRD_PARTY_NOTICES.md) for the retained source and
model license references.
