# Third-party notices

AudioLift contains adapted source from the following projects:

| Component | Source | Retained notice |
| --- | --- | --- |
| Stage-1 video model and inference implementation | [StableAvatar](https://github.com/Francis-Rings/StableAvatar) | [MIT license](LICENSE.txt) |
| Stage-2 Gaussian reconstructor | [FaceLift](https://github.com/weijielyu/FaceLift) | [Apache 2.0 license](stage2/LICENSE) |
| FaceLift pretrained model | [FaceLift](https://github.com/weijielyu/FaceLift) | [Adobe Research License](<stage2/Adobe Research License v1.2.txt>) |

Copyright headers in the source files are retained. AudioLift adds five-view
video generation, Gaussian reconstruction, and the associated inference entrypoints.
The StableAvatar license at the root records the Stage-1 upstream notice;
the Stage-2 notices are retained separately.

External models, datasets and separately installed dependencies retain their
own terms. These include Wan2.1, wav2vec2, MediaPipe, Robust Video Matting,
the Gaussian rasterizer. Model weights and datasets are not
included in this source tree; see the linked providers in the installation,
checkpoint and input guides.
