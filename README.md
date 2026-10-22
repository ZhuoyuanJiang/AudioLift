# AudioLift

**Audio-Driven Full-Head Gaussian Avatar via Multi-view Video Generation**

Generate an animated 3D Gaussian head from a portrait and a speech recording.
Stage 1 generates five synchronized views; Stage 2 reconstructs and renders
an animated head from those views.

> This release provides inference code. Training code is not included.
> Model downloads currently require an authorized Hugging Face login because
> the model repository remains private.

## Installation

Use Linux, Python 3.10, FFmpeg and an NVIDIA GPU with at least 48 GB VRAM for
the default settings. Both stages run sequentially on the same GPU, using
separate Python environments within this repository:

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
Stage 1 uses the sampler from the evaluated generation workflow. Stage 2 uses
the same five-view preparation, cameras and renderer as the evaluated outputs.

The main outputs are:

```text
outputs/example/
├── stage1/
│   ├── grid.mp4       Four synchronized views in a 2×2 grid
│   ├── front.mp4      Separate frontal view
│   ├── reference_used.png          Actual reference supplied to Stage 1
│   └── reference_processing.json   Input checks, processing and fallbacks
└── stage2/
    ├── frontal.mp4   Frontal rendering with audio
    ├── coverage.mp4  Camera sweep with audio
    ├── orbit360.mp4  Full orbit with audio
    └── run.json      Input, checkpoint and rendering details
```

Reference preparation is automatic. Images that pass the existing portrait-aspect
and white-background checks keep their original pixels. Other images use the
existing matting and calibrated face-framing code. If that cannot produce a face
reference, the original image goes through rembg and whole-image fitpad: scaling
with preserved proportions and white padding. If rembg fails or retains less than
8% foreground, the original pixels, including their background, are fitted instead.
The fallback is logged and inference continues; it does not guarantee the same
quality as a well-prepared portrait.

Before generation, the selected reference is saved as `stage1/reference_used.png`.
`stage1/reference_processing.json` records the source path, input checks,
processing and any fallbacks. The source file is left untouched; a separate copy
of the original file is not made. Inspect these outputs when the generated
appearance is unexpected. For manual review, add `--strict-reference`: an image
that needs preprocessing produces a face-framed
candidate when possible, then stops with exit code 3 before generation. Inspect
`stage1/_gate_candidate/` and pass the accepted candidate as `--reference` to rerun.
The standalone `scripts/preprocess/preprocess_reference.py` remains available
for debugging matting and framing. Existing `--skip_reference_check` and
`--allow_noncompliant_reference` overrides still preserve the original input pixels.

Output length follows the original Stage-1 frame plan; Stage 2 renders every
generated frame and retains the full audio, including when the audio is slightly
longer than the video.

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
`--seed`, `--turns`, `--strict-reference`, `--stage1-checkpoint`, `--stage2-checkpoint` and
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

Pass the Stage-1 videos and their original driving audio. Existing output
directories are accepted; Stage 2 keeps its results if all three videos already
exist, as in the paper renderer. Add `--dry-run` to print Stage-2 input metadata
and model paths without using a GPU. Dry runs use local weights and do not download them.

If Stage 2 fails during a combined run, the Stage-1 videos remain available.
Retry with `inference_stage2.py`, the same audio, and
`--out outputs/example/stage2`; there is no need to regenerate Stage 1.

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
to it. `stage1/pipeline.py` runs five-view video generation and uses
`stage1/wan_pipeline.py` for the Wan model wrapper and shared conditioning
helpers. `stage2/pipeline.py` runs Gaussian reconstruction and rendering.
`configs/` holds configuration values: model settings, expected portrait
framing and camera parameters. The trained weights live in `checkpoints/`.
`stage1/dist/` means **distributed computation**, specifically optional multi-GPU
support; the default single-GPU command does not require xFuser.

See [input preparation](data/README.md) for the portrait and audio layout. Only
`data/README.md` is tracked in Git; local inputs, downloaded weights and generated
outputs stay in the ignored `data/`, `checkpoints/` and `outputs/` directories.
`tests/` contains checks for developers and is not needed to run inference.

<details>
<summary>Detailed repository tree</summary>

```text
AudioLift/
├── inference.py                          # Run Stage 1 → Stage 2 in one command
├── inference_stage1.py                   # Run Stage 1 only
├── inference_stage2.py                   # Run Stage 2 only
├── requirements-stage1.txt               # Video generation dependencies
├── requirements-stage2.txt               # Reconstruction/rendering dependencies
│
├── configs/                             # Model settings, framing and cameras
│   ├── stage1/
│   │   ├── wan_civitai.yaml              # Wan components, asset paths, scheduler
│   │   └── nersemble_framing_calibration.json  # Target face size and position
│   └── stage2/
│       ├── inference.yaml               # Gaussian reconstruction settings
│       └── cameras/                     # Four grid cameras + separate frontal camera
│           ├── opencv_cameras_nersemble_017_832x1216.json
│           └── opencv_cameras_nersemble_frontal_canonical_832x1216.json
│
├── stage1/                              # Portrait + audio → five-view video
│   ├── inference.sh                     # Check inputs/assets and launch sampling
│   ├── pipeline.py                      # Audio windows, denoising and video output
│   ├── wan_pipeline.py                  # Wan model, conditioning and latent helpers
│   ├── models/
│   │   ├── wan_fantasy_transformer3d_1B.py  # Released audio-conditioned transformer
│   │   ├── vocal_projector_fantasy_1B.py    # Audio feature conditioning
│   │   ├── vocal_projector_fantasy.py    # Shared audio slicing/padding helpers
│   │   ├── wan_transformer3d.py          # Wan backbone and shared attention
│   │   ├── wan_image_encoder.py          # Reference-image features (CLIP)
│   │   ├── wan_text_encoder.py           # Prompt features (T5)
│   │   ├── wan_xlm_roberta.py            # Upstream CLIP text component
│   │   ├── wan_vae.py                    # Image/video ↔ latent conversion
│   │   ├── cache_utils.py                # Optional TeaCache acceleration
│   │   └── __init__.py                   # Python package marker
│   ├── dist/                            # Optional distributed/multi-GPU support
│   │   ├── __init__.py                   # xFuser imports and distributed setup
│   │   └── wan_xfuser.py                 # Distributed Wan attention helpers
│   ├── utils/
│   │   ├── utils.py                      # Video saving and reference preparation
│   │   ├── fm_solvers.py                 # Retained alternative Flow DPM scheduler
│   │   ├── fm_solvers_unipc.py           # Retained alternative Flow UniPC scheduler
│   │   └── __init__.py                   # Solver helper exports
│   └── __init__.py                       # Python package marker
│
├── stage2/                              # Five-view video → animated Gaussian head
│   ├── inference.sh                     # Launch with Stage-2 environment settings
│   ├── pipeline.py                      # Validate, reconstruct, render, attach audio
│   ├── view_inputs.py                   # Split views, load cameras and checkpoint
│   ├── rendering.py                     # Gaussian representation and rasterization
│   ├── models/
│   │   ├── gslrm.py                      # Five-view Gaussian reconstruction network
│   │   ├── transform_data.py             # Camera rays and model input preparation
│   │   ├── utils_transformer.py          # Attention, MLP and transformer blocks
│   │   └── __init__.py                   # Python package marker
│   ├── README.md                        # Stage-2 usage and upstream references
│   ├── LICENSE                          # Upstream source license
│   ├── Adobe Research License v1.2.txt   # Retained upstream component license
│   └── __init__.py                       # Python package marker
│
├── scripts/
│   ├── download_models.py               # Download and verify model assets
│   ├── preprocess/
│   │   └── preprocess_reference.py      # Matting, face framing and reference checks
│   └── __init__.py                       # Python package marker
├── data/
│   └── README.md                        # Local portrait/audio input instructions
├── docs/
│   ├── installation.md                  # Environments, CUDA rasterizer, preprocessing
│   └── checkpoints.md                   # Model sources, paths and checksums
├── tests/                               # Software checks; not needed for inference
│   ├── test_inference.py                # Two-stage orchestration checks
│   ├── test_download_models.py          # Download, checksum and offline checks
│   ├── stage1/
│   │   ├── test_stage1_entrypoints.py    # Paths, checkpoint and portrait checks
│   │   └── __init__.py                   # Test package marker
│   ├── stage2/
│   │   ├── test_stage2_infer.py          # Video/audio, camera and preflight checks
│   │   └── __init__.py                   # Test package marker
│   └── __init__.py                       # Test package marker
│
├── .gitignore                           # Exclude local inputs, weights and outputs
├── README.md                            # Project overview and usage
├── LICENSE.txt                          # Retained Stage-1 upstream MIT notice
└── THIRD_PARTY_NOTICES.md                # Upstream sources and license references
```

The main execution path is:

```text
inference.py
  → inference_stage1.py → stage1/inference.sh → stage1/pipeline.py
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
