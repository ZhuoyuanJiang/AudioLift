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

In the Stage-1 environment:

```bash
source .venv-stage1/bin/activate
python inference_stage1.py \
  --reference /path/to/portrait.png --audio /path/to/speech.wav \
  --out outputs/example/stage1
```

The reference should be a white-background 832×1216 portrait with the expected
face framing. If preprocessing proposes a replacement, inspect it and pass the
accepted image explicitly as `--reference`. Stage 1 writes `grid.mp4` and
`front.mp4`; output duration follows the audio.

Then reconstruct and render in the Stage-2 environment:

```bash
source .venv-stage2/bin/activate
python inference_stage2.py \
  --stage1-dir outputs/example/stage1 --audio /path/to/speech.wav \
  --out outputs/example/stage2
```

Outputs are `frontal.mp4`, `coverage.mp4` and `orbit360.mp4`, with audio.
Use synchronized 25 fps inputs, matching audio starting at time zero, and a
new Stage-2 output directory. Add `--dry-run` to check Stage-2 inputs and assets
before using a GPU. Dry runs require local weights and do not download them.

## Repository layout

```text
inference_stage1.py / inference_stage2.py
                                      Inference entrypoints
stage1/                               Video model and sampling pipeline
stage2/                               Reconstruction model and rendering
configs/stage1/ / configs/stage2/       Inference model and camera configurations
data/README.md                        Portrait and audio input layout
scripts/download_models.py            Model download and verification
scripts/preprocess/                   Reference portrait preparation
tests/                                Inference and download checks
docs/                                 Installation and checkpoints
```

The stage directories contain the inference launchers used by the root
entrypoints. See [input preparation](data/README.md) for the expected portrait
and audio layout. Local inputs, weights and generated outputs stay in the ignored
`data/`, `checkpoints/` and `outputs/` directories.

## Acknowledgements

Built on [StableAvatar](https://github.com/Francis-Rings/StableAvatar),
[FaceLift](https://github.com/weijielyu/FaceLift) and their dependencies.
See [third-party notices](THIRD_PARTY_NOTICES.md) for the retained source and
model license references.
