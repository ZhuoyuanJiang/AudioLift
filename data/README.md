# Inference inputs

Place a portrait and its driving audio anywhere locally, for example:

```text
data/
  example/
    portrait.png
    speech.wav
```

The portrait should have a white background and the expected 832×1216 framing.
Stage 1 checks its framing before generation. If it proposes a replacement,
inspect the image and pass the accepted image explicitly as `--reference`.
See the [installation guide](../docs/installation.md) for preprocessing assets.

Stage 1 generates `grid.mp4` and `front.mp4`. Pass their directory and the same
audio to Stage 2. Videos must be synchronized at 25 fps, with matching frame
counts and audio beginning at time zero. The grid is 832×1216 and the separate
frontal view is 416×608.

No training dataset is needed for inference. Local media in this directory is
ignored by Git. See the [root README](../README.md#inference) for both commands.
