# Model checkpoints

The two final models are hosted at
[DanJZY/AudioLift](https://huggingface.co/DanJZY/AudioLift). This repository is
currently **private for release review**. Log in with an account that has access:

```bash
hf auth login
```

The inference entrypoints download their missing default checkpoint automatically.
Stage 1 also prepares its Wan and wav2vec2 assets. To prepare both stages before
inference, run this from either installed stage environment:

```bash
python scripts/download_models.py
```

Use `--stage 1` or `--stage 2` to download only one stage. All paths default to
this checkout's `checkpoints/` directory, even when invoked from another directory.
Use `--checkpoints-dir /path/to/checkpoints` with the download script when preparing
assets elsewhere. The download script verifies final weights by SHA-256 and
never silently replaces them. Wan and wav2vec2 downloads use a pinned upstream
revision and check that every required file has its expected size. The Stage-1
launcher also checks these upstream assets, so an interrupted download can be
completed on the next run.

For offline use, download the assets beforehand and pass `--no-download` to an
inference entrypoint, or set `HF_HUB_OFFLINE=1`. Dry runs also disable downloads.
An explicit `--checkpoint PATH` uses that local file with the original model
loader. Inference does not impose a release-hash whitelist on local checkpoints
or configuration files. Run the download script with `--check-only` to verify
the released assets separately.

| Asset | Path | Availability |
| --- | --- | --- |
| AudioLift-mvdiffusion | `checkpoints/AudioLift-mvdiffusion.pt` | Hugging Face; private review |
| AudioLift-gaussianlifter | `checkpoints/AudioLift-gaussianlifter.pt` | Hugging Face; private review |
| Wan2.1 base assets | `checkpoints/Wan2.1-Fun-V1.1-1.3B-InP/` | Upstream StableAvatar |
| Speech encoder | `checkpoints/wav2vec2-base-960h/` | Upstream StableAvatar |

The [StableAvatar weight instructions](https://github.com/Francis-Rings/StableAvatar#-download-weights)
provide the Wan and wav2vec2 assets. The Wan directory must retain its transformer,
tokenizers, VAE, CLIP and T5 files. Upstream checkpoints do not replace AudioLift's
two final checkpoints. No training initialization or teacher model is needed
for inference.

The download script verifies the released weights against these SHA-256 values:

| Checkpoint | SHA-256 |
| --- | --- |
| AudioLift-mvdiffusion | `8b871f1344d0edcc06f59b501e02146ed52080544e5992ce32be36d392738e6a` |
| AudioLift-gaussianlifter | `dcbef48d163f880133d1bd2adbf46140c6d45d0b9cbcebb12276b3bea5a537dc` |

The original checkpoint bytes are unchanged; the two final models have descriptive
release filenames. Downloads are pinned to model revision
`9e1f36db0df254f6d948a3c5f3a5bfc6abad58b1` and StableAvatar asset revision
`140cbd9e2b7b331898e66a8f07e74e1d04359f2f`.

The Gaussian reconstruction weights retain the
upstream [Adobe Research License](<../stage2/Adobe Research License v1.2.txt>)
for noncommercial research. The checkpoints, input media and generated
videos remain external assets and are not embedded in the source repository.
