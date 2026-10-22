# Inference inputs

Place a portrait and its driving audio anywhere locally, for example:

```text
data/
  example/
    portrait.png
    speech.wav
```

Pass your image directly as `--reference`. Stage 1 checks its aspect ratio and
white background, uses prepared references directly, and preprocesses other
images automatically before continuing. If face framing fails, it uses
whole-image matting and fitpad; if matting also fails, it fits the original image
with white padding. Inspect `stage1/reference_used.png` and
`stage1/reference_processing.json` to see the image and processing actually used.
The source image is left untouched.

For manual review, add `--strict-reference`. An image that needs preprocessing
then stops before generation. Inspect the candidate, if one was produced, and
pass the accepted image explicitly as `--reference` to rerun.
See the [installation guide](../docs/installation.md) for preprocessing assets.

Stage 1 generates `grid.mp4` and `front.mp4`. The unified `inference.py` entrypoint
passes their directory and the same audio to Stage 2 automatically. When running
the stages separately, supply these explicitly. Videos must be synchronized at
25 fps, with matching frame counts and audio beginning at time zero. The grid
is 832×1216 and the separate frontal view is 416×608.

No training dataset is needed for inference. Local media in this directory is
ignored by Git. See the [root README](../README.md#inference) for the unified
command and the separate stage commands.
