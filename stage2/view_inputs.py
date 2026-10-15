"""Five-view image and calibrated-camera preparation for the reconstructor."""
import importlib
import json
import numpy as np
import torch
import yaml
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict
from rich import print

DEFAULT_IMG_SIZE = 512


def initialize_gslrm_model(gslrm_checkpoint_path: str, gslrm_config_path: str, device: torch.device):
    """Initialize GSLRM model."""
    model_config = edict(yaml.safe_load(open(gslrm_config_path, "r")))
    module_name, class_name = model_config.model.class_name.rsplit(".", 1)
    print(f"Loading model from {module_name} -> {class_name}")

    ModelClass = importlib.import_module(module_name).__dict__[class_name]
    gslrm_model = ModelClass(model_config)
    model_checkpoint = torch.load(gslrm_checkpoint_path, map_location="cpu", weights_only=True)
    # The original checkpoint also contains perceptual-loss weights. Only the
    # reconstruction network is needed here; its keys still load strictly.
    model_state = {
        key: value for key, value in model_checkpoint["model"].items()
        if not key.startswith("loss_calculator.")
    }
    gslrm_model.load_state_dict(model_state, strict=True)
    gslrm_model = gslrm_model.to(device)

    return gslrm_model

def extract_views_from_grid(grid_image: Image.Image) -> list:
    """
    Extract 4 views from a 512x512 grid image.

    Layout (RenderMe360):
    ┌─────────┬─────────┐
    │  FRONT  │  RIGHT  │  cam_28 (-7°)   cam_37 (-68°)
    ├─────────┼─────────┤
    │  BACK   │  LEFT   │  cam_49 (-155°) cam_54 (70°)
    └─────────┴─────────┘

    Returns views in order: [Front, Right, Back, Left]
    """
    w, h = grid_image.size
    half_w, half_h = w // 2, h // 2

    # Extract each quadrant (matching RenderMe360 grid layout)
    front = grid_image.crop((0, 0, half_w, half_h))           # top-left: cam_28
    right = grid_image.crop((half_w, 0, w, half_h))           # top-right: cam_37
    back = grid_image.crop((0, half_h, half_w, h))            # bottom-left: cam_49
    left = grid_image.crop((half_w, half_h, w, h))            # bottom-right: cam_54

    return [front, right, back, left]

def prepare_views(views: list, target_size: int = 512) -> list:
    """Resize views to target size if needed.

    Args:
        views: List of PIL Images
        target_size: Target square size (e.g. 512).
                     If 0 or None, use native resolution (no resize, no padding).
                     Requires split_data=False in GS-LRM forward to avoid
                     TransformTarget's square crop assumption.
    """
    if not target_size:
        return views
    prepared = []
    for view in views:
        if view.size != (target_size, target_size):
            view = view.resize((target_size, target_size), Image.LANCZOS)
        prepared.append(view)
    return prepared

def fit_pad(img, W, H):
    """Resize-fit (preserve aspect) into W x H with white padding — matches the grid-cell prep."""
    sw, sh = img.size
    s = min(W / sw, H / sh)
    nw, nh = round(sw * s), round(sh * s)
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img, ((W - nw) // 2, (H - nh) // 2))
    return canvas

def load_cams(grid_json, frontal_json, device, include_frontal):
    """4 grid cameras (order front/right/back/left) + optionally the frontal cam (view_0).
    Both JSONs store c2w directly. Returns fxfycxcy [1,V,4], c2w [1,V,4,4]."""
    frames = list(json.load(open(grid_json))["frames"])
    if include_frontal:
        frames = frames + list(json.load(open(frontal_json))["frames"])
    intr, extr = [], []
    for fr in frames:
        intr.append(np.array([fr["fx"], fr["fy"], fr["cx"], fr["cy"]]))
        c2w = np.array(fr["c2w"]) if "c2w" in fr else np.linalg.inv(np.array(fr["w2c"]))
        extr.append(c2w)
    fxfycxcy = torch.from_numpy(np.stack(intr).astype(np.float32))[None].to(device)
    c2w = torch.from_numpy(np.stack(extr).astype(np.float32))[None].to(device)
    return fxfycxcy, c2w

def build_batch(views, fxfycxcy, c2w, device):
    lrm = np.stack([np.array(v) for v in views], axis=0)
    lrm = torch.from_numpy(lrm).float()[None].to(device) / 255.0
    lrm = rearrange(lrm, "b v h w c -> b v c h w")
    index = torch.stack(
        [torch.zeros(lrm.size(1)).long(), torch.arange(lrm.size(1)).long()], dim=-1
    )[None].to(device)
    return edict(image=lrm, c2w=c2w, fxfycxcy=fxfycxcy, index=index)
