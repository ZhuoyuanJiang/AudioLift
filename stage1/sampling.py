#!/usr/bin/env python3
"""Long-form frontal-token inference: the Session 77 dual-stream sampler, windowed.

Session 77's sampler denoises the whole clip in one window, and refuses anything
other than 81 pixel frames. That guard is not an architectural limit -- seq_len is
computed from the actual latent shape -- but removing it alone does not buy long
video, because attention is quadratic in sequence length. At 832x1216 an 81-frame
clip is already ~109k tokens; one minute would be ~1.86M, roughly 290x the
attention cost. It does not fit on a 48 GB card by two orders of magnitude.

So long video needs windowing, and the windowing that exists in this repo
(session68_inference_frame_anchor_long.py) predates the frontal-token stream and
knows nothing about it -- it drives a single latent stream. This script is the
Session 77 sampler with Session 68's window loop wrapped around it, which is the
one combination neither file had.

The structure, unchanged from Session 68: allocate the FULL-length latents, and at
every timestep denoise overlapping windows of them, blending each window's overlap
with the one before it. Windows advance inside the timestep loop, not outside, so
neighbouring windows are always at the same noise level when they are blended --
which is what keeps the seam from being a discontinuity in time.

What this file has to add on top of that:

  * TWO streams. The grid and frontal latents have different spatial shapes and
    therefore cannot share a scheduler, so Session 77 runs a second scheduler
    instance in lockstep. Windowing doubles that: both streams are sliced on the
    same temporal window, both are stepped, both are blended, and BOTH schedulers
    have their _step_index reset per window -- missing either one silently
    desynchronises the pair and the assert at the bottom of the loop catches it.

  * Per-window seq_len and video_sample_n_frames. Both are derived from the
    window's own frame count; passing the full-clip values would make the model
    compute positional structure for frames that are not in its input.

  * Per-window audio. Session 77 encodes the whole clip's audio in one wav2vec
    pass; here each window gets its own slice, cached up front so wav2vec runs
    once rather than once per timestep.
"""

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import imageio
import librosa
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage1.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from stage1.models.wan_image_encoder import CLIPModel
from stage1.models.wan_text_encoder import WanT5EncoderModel
from stage1.models.wan_vae import AutoencoderKLWan
from stage1.pipeline import WanI2VFantasyPipeline, retrieve_timesteps
from stage1.utils.utils import get_image_to_video_latent, save_videos_grid

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocess"))
from preprocess_reference import ReferenceGateError, ensure_reference  # noqa: E402

# Measured 2026-09-09 on RTX 6000 Ada with the B checkpoint (and identical in
# August with A): one 81-frame window costs this many seconds per denoising
# step. Used only by --dry_run for the cost estimate.
SEC_PER_WINDOW_STEP = 53.0
# Wan2.1 VAE temporal stride. --dry_run plans windows before any model is
# loaded, so it needs the number up front; main() asserts it against the VAE.
TEMPORAL_COMPRESSION_RATIO = 4


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def filter_kwargs(cls, kwargs):
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def parse_args():
    parser = argparse.ArgumentParser(description="Session75 dual-stream frame-anchor inference")
    parser.add_argument("--config_path", type=str, default="configs/stage1/wan_civitai.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--pretrained_wav2vec_path", type=str, required=True)
    parser.add_argument("--validation_reference_path", type=str, required=True)
    parser.add_argument("--skip_reference_check", action="store_true",
                        help="Do not check/auto-preprocess the reference into the "
                             "training distribution. For deliberate A/B tests only.")
    parser.add_argument("--allow_noncompliant_reference", action="store_true",
                        help="If the reference is not a training-distribution image, warn "
                             "and use it verbatim instead of stopping. For deliberate A/B "
                             "tests, or for a photo you know is a matted portrait that the "
                             "corner rule misjudges.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Check the reference, plan the windows, print the cost "
                             "estimate and exit before loading any model.")
    parser.add_argument("--validation_driven_audio_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--video_sample_n_frames", type=int, default=81,
                        help="ignored unless --max_output_frames is set; length comes from the audio")
    parser.add_argument("--frames_per_batch", type=int, default=81,
                        help="pixel frames per window; keep at the trained 81")
    parser.add_argument("--overlap_window_length", type=int, default=5,
                        help="overlap between windows, in LATENT frames")
    parser.add_argument("--overlapping_weight_scheme", type=str, default="uniform",
                        choices=["uniform", "log"])
    parser.add_argument("--max_output_frames", type=int, default=0,
                        help="0 = use the audio's full natural length")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="The protagonist is talking")
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--prompt_cfg_scale", type=float, default=3.0)
    parser.add_argument("--audio_cfg_scale", type=float, default=5.0)
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Only controls whether 3-branch CFG is enabled.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--save_side_by_side_preview",
        action="store_true",
        default=True,
        help="Save front+grid side-by-side preview video in addition to separate streams.",
    )
    parser.add_argument(
        "--no_side_by_side_preview",
        dest="save_side_by_side_preview",
        action="store_false",
        help="Disable the front+grid side-by-side preview video.",
    )
    parser.add_argument(
        "--anchor_vae_encode_mode",
        type=str,
        default="sample",
        choices=["sample", "mode"],
        help="sample matches Session65 training encode; mode is deterministic.",
    )
    parser.add_argument(
        "--front_stream_mode",
        type=str,
        default="random",
        choices=["random", "zero"],
        help="Use random frontal latents for normal S75 inference, or zeros for A/B debugging.",
    )
    parser.add_argument(
        "--s76_variant",
        type=str,
        default=os.environ.get("S76_VARIANT", "selfattn"),
        choices=["selfattn", "m7", "f8"],
        help=(
            "Must match the training variant so the view modules are applied at inference. "
            "m7/f8 enable block_view_embedding(5) + view_projector; selfattn keeps them off."
        ),
    )
    return parser.parse_args()


def load_models(args, weight_dtype):
    config = OmegaConf.load(args.config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.pretrained_wav2vec_path)
    wav2vec = Wav2Vec2Model.from_pretrained(args.pretrained_wav2vec_path).eval()

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    ).eval()

    transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    # Session 77: frontal rides as its own tokens; head stays 16ch, so S65/S77
    # checkpoints load natively (no head expansion).
    transformer_kwargs["frontal_token_stream"] = True
    transformer3d = WanTransformer3DFantasyModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=transformer_kwargs,
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )

    print(f"Loading Session77/S65-compatible transformer checkpoint: {args.transformer_path}")
    state_dict = torch.load(args.transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    # S65-lineage checkpoints carry a 4-row view_embedding; this model uses 5
    # rows. Pad (rows 0-3 copied, row 4 keeps zero init) to avoid a
    # load_state_dict size-mismatch RuntimeError — same guard the train script
    # uses. S77-trained checkpoints already have 5 rows and pass through.
    state_dict = WanTransformer3DFantasyModel.maybe_pad_view_embedding_state_dict(state_dict, transformer3d)
    missing, unexpected = transformer3d.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded - missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing:
        print(f"First missing keys: {missing[:5]}")
    if unexpected:
        print(f"First unexpected keys: {unexpected[:5]}")

    # Session76: enable the M7/F8 view modules at inference so the trained
    # block_view_embedding + view_projector weights are actually applied.
    # These are runtime flags (not stored in the state_dict), so they MUST be
    # set here to match the training variant, else the previews are wrong.
    s76_variant = getattr(args, "s76_variant", "selfattn")
    use_view_modules = s76_variant in ("m7", "f8")
    transformer3d.use_view_embedding = False
    transformer3d.use_block_view_embedding = use_view_modules
    transformer3d.use_view_projector = use_view_modules
    for block in transformer3d.blocks:
        block.use_view_projector = use_view_modules
    print(f"[Session76 inference] variant={s76_variant} view_modules_on={use_view_modules}")

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = WanI2VFantasyPipeline(
        vae=vae.to(weight_dtype),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer3d,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )

    return pipeline, wav2vec_processor, wav2vec


def encode_reference_anchor(pipeline, reference_path, height, width, weight_dtype, device, encode_mode):
    input_video, _, clip_image = get_image_to_video_latent(
        reference_path,
        None,
        video_length=1,
        sample_size=[height, width],
    )
    reference_video = input_video[:, :, :1]
    reference_video = pipeline.image_processor.preprocess(
        rearrange(reference_video, "b c f h w -> (b f) c h w"),
        height=height,
        width=width,
    )
    reference_video = rearrange(reference_video, "(b f) c h w -> b c f h w", f=1)
    reference_video = reference_video.to(device=device, dtype=pipeline.vae.dtype)

    posterior = pipeline.vae.encode(reference_video)[0]
    if encode_mode == "mode":
        clean_ref_anchor = posterior.mode()
    else:
        clean_ref_anchor = posterior.sample()
    return clean_ref_anchor.to(device=device, dtype=weight_dtype), clip_image


def build_windows(infer_length, frames_per_batch, overlap_window_length):
    """Overlapping [start, end) spans over the LATENT frame axis.

    Copied from session68_inference_frame_anchor_long.py rather than reinvented:
    the blending weights and stride have to agree with it for the seams to behave
    the same way, and that file is what produced the long videos this repo already
    trusts.

    One deliberate departure: when the last
    window would come out shorter than frames_per_batch, it is slid back so that
    it is full length. The checkpoint only ever trained on 81-frame windows, and
    the short tail is where the largest measured frame-to-frame jumps of a clip
    sat (BobbySchilling_40s frame 960, DianeBlack0_60s frame 1480), although they
    were not visible by eye. The crossfade in the denoise loop already blends over
    whatever overlap results, so no other code path changes.
    """
    if infer_length < 1 or frames_per_batch < 1:
        raise ValueError(f"bad lengths: infer={infer_length}, per_batch={frames_per_batch}")
    if not 0 <= overlap_window_length < frames_per_batch:
        raise ValueError(
            "overlap_window_length must satisfy 0 <= overlap < frames_per_batch, "
            f"got overlap={overlap_window_length}, frames_per_batch={frames_per_batch}"
        )
    windows, stride, start = [], frames_per_batch - overlap_window_length, 0
    while start < infer_length:
        end = min(start + frames_per_batch, infer_length)
        if end - start < frames_per_batch and start > 0:
            # Short tail: slide the window back to full length instead of running
            # the model at a size it never trained on. `start > 0` keeps the
            # single-window case (audio shorter than one window) untouched.
            start = infer_length - frames_per_batch
            end = infer_length
        windows.append((start, end))
        if end == infer_length:
            break
        start += stride
    return windows


def plan_windows(n_pixel_total, tcr, frames_per_batch, overlap_window_length):
    """Turn an audio length into the sampler's frame plan: (video_length,
    infer_length, latent_per_window, windows). One place for the arithmetic so
    --dry_run and the real run cannot disagree.
    """
    video_length = int((n_pixel_total - 1) // tcr * tcr) + 1
    infer_length = (video_length - 1) // tcr + 1          # latent frames
    latent_per_window = (frames_per_batch - 1) // tcr + 1
    windows = build_windows(infer_length, latent_per_window, overlap_window_length)
    return video_length, infer_length, latent_per_window, windows


def build_overlap_weight(overlap_length, scheme, device, dtype):
    """Ramp from 0 to 1 across the overlap, so the seam is a crossfade not a cut."""
    if overlap_length <= 0:
        return None
    if overlap_length == 1:
        return torch.full((1, 1, 1, 1, 1), 0.5, device=device, dtype=dtype)
    if scheme == "uniform":
        values = torch.linspace(0, 1, overlap_length, device=device, dtype=dtype)
    else:
        init = torch.linspace(0, 1, overlap_length, device=device, dtype=dtype)
        values = torch.log1p(init * (torch.exp(torch.tensor(1.0, device=device, dtype=dtype)) - 1))
        values = (values - values.min()) / (values.max() - values.min())
    return values.view(1, 1, overlap_length, 1, 1)


def window_pixel_frames(latent_window_length, temporal_compression_ratio):
    return int((latent_window_length - 1) * temporal_compression_ratio + 1)


def cache_window_audio(windows, audio_path, fps, temporal_compression_ratio,
                       wav2vec_processor, wav2vec, device, weight_dtype,
                       do_classifier_free_guidance):
    """One wav2vec pass per window, cached before denoising starts.

    Each window needs the audio for ITS frames, and the denoising loop visits every
    window at every timestep -- so encoding on demand would run wav2vec
    num_windows x num_inference_steps times instead of num_windows.
    """
    sr = 16000
    audio_input, _ = librosa.load(audio_path, sr=sr)
    tokens_per_frame = int(sr / fps)
    cache = {}
    for start, end in windows:
        first_pixel = int(start * temporal_compression_ratio)
        n_pixel = window_pixel_frames(end - start, temporal_compression_ratio)
        seg = audio_input[first_pixel * tokens_per_frame:(first_pixel + n_pixel) * tokens_per_frame]
        if len(seg) == 0:
            seg = audio_input[-tokens_per_frame:]
        vals = wav2vec_processor(seg, sampling_rate=sr, return_tensors="pt").input_values.to(device)
        emb = wav2vec(vals).last_hidden_state.to(dtype=weight_dtype)
        if do_classifier_free_guidance:
            emb = torch.cat([torch.zeros_like(emb), torch.zeros_like(emb), emb], dim=0)
        cache[start] = emb.cpu()
    return cache


def audio_frames_available(audio_path, fps):
    audio_input, _ = librosa.load(audio_path, sr=16000)
    return int(len(audio_input) / 16000 * fps)


def decode_latents_to_mp4(pipeline, latents, path, fps):
    """Decode latents one latent frame at a time and append the pixels straight to
    the mp4, so host memory stays flat instead of holding the whole clip as
    float32 (12 MB per 832x1216 frame: 18 GB per minute, 55 GB per minute once
    the old side-by-side preview upsampled and concatenated it). The frames are
    converted exactly as save_videos_grid did -- clamp, /2 + 0.5, x255, truncate
    to uint8 -- and written with the same imageio backend, and the optional
    S77_DUMP_PNG_DIR lossless dump is honoured. Returns the frame count.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    dump_dir = os.environ.get("S77_DUMP_PNG_DIR")
    if dump_dir:
        dump_dir = os.path.join(dump_dir, os.path.splitext(os.path.basename(path))[0])
        os.makedirs(dump_dir, exist_ok=True)
    count = [0]

    def on_chunk(chunk):
        # chunk: [1, 3, t, H, W] raw decoder output on the GPU; same math as
        # AutoencoderKLWan._decode (clamp) + decode_latents (/2 + 0.5, clamp).
        # Same dtype order as the old path -- clamp and /2 + 0.5 in the VAE dtype
        # (bf16), float32 only on the CPU, then x255 and truncate -- so the frames
        # are bit-identical to decode_latents + save_videos_grid.
        frames = (chunk[0].clamp_(-1, 1) / 2 + 0.5).clamp(0, 1)
        frames = frames.permute(1, 2, 3, 0).cpu().float()
        frames = (frames * 255).numpy().astype(np.uint8)
        for f in frames:
            writer.append_data(f)
            if dump_dir:
                Image.fromarray(f).save(os.path.join(dump_dir, f"{count[0]:03d}.png"))
            count[0] += 1

    with torch.no_grad():
        pipeline.vae.decode(latents.to(pipeline.vae.dtype), frame_callback=on_chunk)
    writer.close()
    return count[0]


def write_side_by_side_preview(front_path, grid_path, preview_path):
    """front (416x608) upsampled 2x nearest, beside grid (832x1216) -- the same
    layout the in-memory preview had, built by ffmpeg from the two finished mp4s
    so it costs no RAM. It is a preview: the front half is re-encoded from the
    already-compressed front.mp4.
    """
    import subprocess
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
           "-i", front_path, "-i", grid_path,
           "-filter_complex", "[0:v]scale=iw*2:ih*2:flags=neighbor[f];[f][1:v]hstack=inputs=2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", preview_path]
    subprocess.run(cmd, check=True)


def save_decoded_videos(grid_video, front_video, output_dir, fps, save_side_by_side_preview=True):
    os.makedirs(output_dir, exist_ok=True)

    grid_path = os.path.join(output_dir, "grid.mp4")
    front_path = os.path.join(output_dir, "front.mp4")
    save_videos_grid(grid_video, grid_path, fps=fps)
    save_videos_grid(front_video, front_path, fps=fps)
    print(f"Saved grid video to: {grid_path}")
    print(f"Saved frontal video to: {front_path}")

    if save_side_by_side_preview:
        if grid_video.shape != front_video.shape:
            # Session 77: the frontal is half resolution; upsample 2x so the
            # side-by-side preview keeps the familiar layout.
            front_video = torch.nn.functional.interpolate(
                front_video.float(), scale_factor=(1, 2, 2), mode="nearest"
            ).to(front_video.dtype)
        if grid_video.shape != front_video.shape:
            raise ValueError(
                "grid_video and front_video must have the same shape for side-by-side preview, "
                f"got grid={tuple(grid_video.shape)}, front={tuple(front_video.shape)}"
            )
        preview = torch.cat([front_video, grid_video], dim=-1)
        preview_path = os.path.join(output_dir, "preview_5view_side_by_side.mp4")
        save_videos_grid(preview, preview_path, fps=fps)
        print(f"Saved 5-view side-by-side preview to: {preview_path}")


def main():
    args = parse_args()
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16

    # The reference has to be a training-distribution one: portrait aspect, subject
    # matted onto white, face framed to the NeRSemble calibration. A raw photo used
    # to sail straight through here and get non-uniformly squashed to height x width
    # by get_image_to_video_latent, which previously accepted an unprocessed portrait. Warn and fix
    # rather than abort -- a run that dies six hours in helps nobody.
    # the gate now stops the job when the photo cannot be
    # repaired. It runs before any model is loaded, so stopping here costs seconds,
    # not hours; --allow_noncompliant_reference restores the warn-and-continue path.
    # 2026-09-11: the gate no longer repairs anything on its own. A non-compliant
    # photo is refused and a repaired candidate is written to _gate_candidate/
    # for the user to inspect and pass in explicitly (the automatic swap had eaten
    # the black hat of RM360 subject 0168).
    if not args.skip_reference_check:
        try:
            args.validation_reference_path = ensure_reference(
                args.validation_reference_path,
                width=args.width, height=args.height,
                cache_dir=os.path.join(args.output_dir, "_gate_candidate"),
                tag="reference",
                allow_noncompliant=args.allow_noncompliant_reference,
            )
        except ReferenceGateError as e:
            print(f"[reference] REFUSED: {e}", flush=True)
            sys.exit(3)

    # The 81-frame guard from the single-window sampler is gone: length is now set
    # by the audio, and each WINDOW is what has to stay near the trained size.
    if args.frames_per_batch != 81:
        print(f"WARNING: frames_per_batch={args.frames_per_batch}, but this checkpoint was "
              "trained at 81. Windows far from that size are extrapolation.")

    # Plan the clip before touching any model, so --dry_run can report it and a
    # real run shows the same numbers it will act on.
    tcr = TEMPORAL_COMPRESSION_RATIO
    audio_frames = audio_frames_available(args.validation_driven_audio_path, args.fps)
    n_pixel_total = args.max_output_frames or audio_frames
    video_length, infer_length, latent_per_window, windows = plan_windows(
        n_pixel_total, tcr, args.frames_per_batch, args.overlap_window_length)
    print(f"Long inference: {video_length} pixel frames ({video_length/args.fps:.1f}s) "
          f"= {infer_length} latent frames, {len(windows)} windows of "
          f"{latent_per_window} (overlap {args.overlap_window_length})")
    # Print the actual spans: since the tail-alignment change the last window
    # can overlap its predecessor by more than overlap_window_length.
    print(f"Window spans (latent): {windows}")
    short = audio_frames - video_length
    if short > 0:
        print(f"Output is {short} frame(s) ({short/args.fps:.2f}s) shorter than the audio: "
              f"the VAE emits 4k+1 frames, so the audio's last {short} frame(s) get no picture.")
    if len(windows) > 1 and (windows[-1][1] - windows[-1][0]) < latent_per_window:
        print(f"WARNING: last window is {windows[-1][1] - windows[-1][0]} latent frames, "
              f"shorter than the trained {latent_per_window}.")
    gpu_h = len(windows) * args.num_inference_steps * SEC_PER_WINDOW_STEP / 3600
    print(f"Estimated cost: {len(windows)} windows x {args.num_inference_steps} steps x "
          f"{SEC_PER_WINDOW_STEP:.0f}s = {gpu_h:.1f} GPU-hours on one card (denoising is "
          f"sequential; more cards do not speed up one clip). VRAM ~45 GB regardless of "
          f"length; host RAM flat (streaming decode).")
    if args.dry_run:
        print("--dry_run: stopping before loading models.")
        return

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    print("Loading models...")
    pipeline, wav2vec_processor, wav2vec = load_models(args, weight_dtype)
    pipeline = pipeline.to(device)
    pipeline.transformer.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.clip_image_encoder.eval()

    do_classifier_free_guidance = args.guidance_scale > 1.0
    # Length comes from the audio, not from a flag. The single-window sampler took
    # 81 because that was all it could do; here the clip runs as long as there is
    # speech to drive it, which is what "natural length" has meant in this project
    # since the in-the-wild work. The plan itself was made above, before loading.
    assert pipeline.vae.config.temporal_compression_ratio == tcr, \
        f"VAE temporal stride {pipeline.vae.config.temporal_compression_ratio} != planned {tcr}"

    print("Encoding clean reference anchor...")
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        clean_ref_anchor, clip_image = encode_reference_anchor(
            pipeline,
            args.validation_reference_path,
            args.height,
            args.width,
            weight_dtype,
            device,
            args.anchor_vae_encode_mode,
        )

        print("Encoding prompt and CLIP reference...")
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            args.prompt,
            args.negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
            dtype=weight_dtype,
        )
        if do_classifier_free_guidance:
            # Dual-CFG branches must be [uncond, neg-text+audio, full]. The audio adapter
            # internally forces audio to [off, on, on] (it uses only vocal_embeddings[-1:]),
            # so the ONLY thing that separates branch 1 from branch 2 is the text prompt.
            # [neg, neg, pos] keeps both the audio term (b1-b0) and text term (b2-b1) live and
            # matches the channel-conditioning baseline; [neg, pos, pos] makes b1==b2 and
            # silently zeroes out audio_cfg.
            prompt_embeds = negative_prompt_embeds + negative_prompt_embeds + prompt_embeds

        clip_tensor = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
        clip_context = pipeline.clip_image_encoder([clip_tensor[:, None, :, :]])
        if do_classifier_free_guidance:
            clip_context = torch.cat([clip_context, clip_context, clip_context], dim=0)

        print(f"Encoding audio for {len(windows)} windows...")
        wav2vec.to(device)
        window_audio = cache_window_audio(
            windows, args.validation_driven_audio_path, args.fps, tcr,
            wav2vec_processor, wav2vec, device, weight_dtype, do_classifier_free_guidance)
        wav2vec.to("cpu")
        torch.cuda.empty_cache()
        vocal_embeddings = None  # per-window now; see the denoise loop
        # CFG branches for audio are built inside cache_window_audio.

        print("Preparing grid and frontal target latents...")
        grid_latents = pipeline.prepare_latents(
            1,
            pipeline.vae.config.latent_channels,
            video_length,
            args.height,
            args.width,
            weight_dtype,
            device,
            generator,
            latents=None,
        )
        front_latents = pipeline.prepare_latents(
            1,
            pipeline.vae.config.latent_channels,
            video_length,
            args.height // 2,
            args.width // 2,
            weight_dtype,
            device,
            generator,
            latents=None,
        )
        if args.front_stream_mode == "zero":
            print("Using zero frontal stream latents for debugging.")
            front_latents = torch.zeros_like(front_latents)

        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler,
            args.num_inference_steps,
            device,
            timesteps=None,
            mu=1,
        )
        # Session 77: the two streams have different shapes, so they cannot be
        # channel-concatenated into one scheduler.step call. A second scheduler
        # instance (identical config + timesteps) advances the frontal stream;
        # FlowMatchEulerDiscreteScheduler.step is stateless apart from
        # _step_index, so the two stay in lockstep.
        front_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        front_timesteps, _ = retrieve_timesteps(
            front_scheduler,
            args.num_inference_steps,
            device,
            timesteps=None,
            mu=1,
        )
        assert torch.equal(timesteps, front_timesteps), "grid/front schedulers must share timesteps"
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta=0.0)

        latent_h, latent_w = grid_latents.shape[-2:]
        front_h, front_w = front_latents.shape[-2:]
        patch_h, patch_w = pipeline.transformer.config.patch_size[1:]

        def seq_len_for(n_latent_frames):
            """Exact token count for a window of this many latent frames.

            Recomputed per window rather than once: SDPA ignores the padding mask,
            so an over-allocated seq_len does not just waste memory, it changes what
            the model attends over. The +1 is the anchor frame prepended to the grid
            stream; the frontal stream carries a zero frame in the same slot, so both
            contribute tokens for the same frame count.
            """
            return (n_latent_frames + 1) * (
                (latent_h // patch_h) * (latent_w // patch_w)
                + (front_h // patch_h) * (front_w // patch_w)
            )

        print(
            "Session77-long shapes: "
            f"anchor={tuple(clean_ref_anchor.shape)}, "
            f"grid={tuple(grid_latents.shape)}, front={tuple(front_latents.shape)}, "
            f"front_stream_mode={args.front_stream_mode}, "
            f"windows={len(windows)}, seq_len/window={seq_len_for(latent_per_window)}"
        )

        print(f"Generating {video_length} target frames at {args.width}x{args.height}...")

        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Windows advance INSIDE the timestep loop. Denoising each window
                # to completion before starting the next would blend a finished
                # window against a barely-started one, and the seam would be a jump
                # in time rather than a crossfade.
                grid_pred_all = torch.zeros_like(grid_latents)
                front_pred_all = torch.zeros_like(front_latents)
                index_previous_end = None

                for window_idx, (index_start, index_end) in enumerate(windows):
                    # BOTH schedulers, every window. FlowMatchEuler picks its sigma
                    # from _step_index; leaving it advanced from the previous window
                    # would step this one at the wrong noise level, and forgetting
                    # either one desynchronises the pair -- which the assert below
                    # catches, but only after the step has already been taken wrong.
                    pipeline.scheduler._step_index = None
                    front_scheduler._step_index = None

                    grid_window = grid_latents[:, :, index_start:index_end].clone()
                    front_window = front_latents[:, :, index_start:index_end].clone()
                    n_latent = grid_window.shape[2]
                    local_pixel_frames = window_pixel_frames(n_latent, tcr)

                    grid_model_input = (
                        torch.cat([grid_window] * 3) if do_classifier_free_guidance else grid_window
                    )
                    front_model_input = (
                        torch.cat([front_window] * 3) if do_classifier_free_guidance else front_window
                    )
                    if hasattr(pipeline.scheduler, "scale_model_input"):
                        grid_model_input = pipeline.scheduler.scale_model_input(grid_model_input, t)
                        front_model_input = pipeline.scheduler.scale_model_input(front_model_input, t)

                    anchor_model_input = (
                        torch.cat([clean_ref_anchor] * 3) if do_classifier_free_guidance else clean_ref_anchor
                    )
                    latent_model_input = torch.cat([anchor_model_input, grid_model_input], dim=2)
                    y = torch.zeros(
                        latent_model_input.shape[0], 20, latent_model_input.shape[2],
                        latent_model_input.shape[3], latent_model_input.shape[4],
                        device=device, dtype=weight_dtype,
                    )
                    # y stays all-zero; the frontal enters as its own tokens via
                    # x_front, with a zero latent frame filling the anchor slot.
                    front_zero_anchor = torch.zeros_like(front_model_input[:, :, :1])
                    x_front_input = torch.cat([front_zero_anchor, front_model_input], dim=2)
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = pipeline.transformer(
                        x=latent_model_input,
                        context=prompt_embeds,
                        t=timestep,
                        seq_len=seq_len_for(n_latent),
                        y=y,
                        clip_fea=clip_context,
                        vocal_embeddings=window_audio[index_start].to(device),
                        is_clip_level_modeling=False,
                        video_sample_n_frames=local_pixel_frames,
                        frame_conditioning_anchor=True,
                        x_front=x_front_input,
                    )
                    grid_noise_pred, front_noise_pred = noise_pred
                    grid_noise_pred = grid_noise_pred[:, :, 1:]
                    front_noise_pred = front_noise_pred[:, :, 1:]

                    if do_classifier_free_guidance:
                        # branches: 0=uncond(neg, no audio), 1=neg-text+audio, 2=full(pos+audio).
                        def _cfg_combine(pred):
                            pred_uncond, pred_drop_audio, pred_cond = pred.chunk(3)
                            return (
                                pred_uncond
                                + args.audio_cfg_scale * (pred_drop_audio - pred_uncond)
                                + args.prompt_cfg_scale * (pred_cond - pred_drop_audio)
                            )

                        grid_noise_pred = _cfg_combine(grid_noise_pred)
                        front_noise_pred = _cfg_combine(front_noise_pred)

                    grid_updated = pipeline.scheduler.step(
                        grid_noise_pred, t, grid_window, **extra_step_kwargs, return_dict=False,
                    )[0]
                    front_updated = front_scheduler.step(
                        front_noise_pred, t, front_window, **extra_step_kwargs, return_dict=False,
                    )[0]
                    assert pipeline.scheduler._step_index == front_scheduler._step_index, \
                        "grid/front schedulers drifted out of lockstep"

                    # Crossfade the overlap with what the previous window wrote.
                    # Skipped at i == 0 because at the first timestep every window
                    # still holds the same initial noise, so blending would only
                    # attenuate it.
                    if window_idx != 0 and i != 0 and index_previous_end is not None:
                        actual_overlap = max(0, index_previous_end - index_start)
                        if actual_overlap > 0:
                            w = build_overlap_weight(
                                actual_overlap, args.overlapping_weight_scheme,
                                grid_updated.device, grid_updated.dtype)
                            for updated, pred_all in ((grid_updated, grid_pred_all),
                                                      (front_updated, front_pred_all)):
                                prev = pred_all[:, :, index_start:index_start + actual_overlap]
                                cur = updated[:, :, :actual_overlap]
                                updated[:, :, :actual_overlap] = cur * w + prev * (1 - w)

                    grid_pred_all[:, :, index_start:index_end] = grid_updated.to(grid_pred_all.dtype)
                    front_pred_all[:, :, index_start:index_end] = front_updated.to(front_pred_all.dtype)
                    index_previous_end = index_end

                grid_latents = grid_pred_all
                front_latents = front_pred_all
                progress_bar.update()

    # decode straight into the mp4s instead of holding the
    # whole clip in host memory; the preview is assembled by ffmpeg afterwards.
    grid_path = os.path.join(args.output_dir, "grid.mp4")
    front_path = os.path.join(args.output_dir, "front.mp4")
    n = decode_latents_to_mp4(pipeline, grid_latents, grid_path, args.fps)
    print(f"Saved grid video to: {grid_path} ({n} frames)")
    n = decode_latents_to_mp4(pipeline, front_latents, front_path, args.fps)
    print(f"Saved frontal video to: {front_path} ({n} frames)")
    if args.save_side_by_side_preview:
        preview_path = os.path.join(args.output_dir, "preview_5view_side_by_side.mp4")
        write_side_by_side_preview(front_path, grid_path, preview_path)
        print(f"Saved 5-view side-by-side preview to: {preview_path}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
