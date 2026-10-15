# Copyright 2025 Adobe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# transform_data.py is under the Adobe Research License. Copyright 2025 Adobe Inc.

"""
Data transformation utilities for GSLRM model.

This module transforms calibrated input images into camera-ray representations.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

# =============================================================================
# Utility Functions
# =============================================================================

def compute_camera_rays(
    fxfycxcy: torch.Tensor, 
    c2w: torch.Tensor, 
    h: int, 
    w: int, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute camera rays for given intrinsics and extrinsics.
    
    Args:
        fxfycxcy: Camera intrinsics [b*v, 4]
        c2w: Camera-to-world matrices [b*v, 4, 4]
        h: Image height
        w: Image width
        device: Target device
        
    Returns:
        Tuple of (ray_origins, ray_directions, ray_directions_camera)
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(device), x.to(device)
    
    b_v = fxfycxcy.size(0)
    x = x[None, :, :].expand(b_v, -1, -1).reshape(b_v, -1)
    y = y[None, :, :].expand(b_v, -1, -1).reshape(b_v, -1)
    
    # Convert to normalized camera coordinates
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    
    ray_d_cam = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    ray_d_cam = ray_d_cam / torch.norm(ray_d_cam, dim=2, keepdim=True)
    
    # Transform to world coordinates
    ray_d = torch.bmm(ray_d_cam, c2w[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)
    
    return ray_o, ray_d, ray_d_cam


def sample_patch_rays(
    image: torch.Tensor,
    fxfycxcy: torch.Tensor,
    c2w: torch.Tensor,
    patch_size: int,
    h: int,
    w: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample rays at patch centers for efficient processing.
    
    Args:
        image: Input images [b*v, c, h, w]
        fxfycxcy: Camera intrinsics [b*v, 4]
        c2w: Camera-to-world matrices [b*v, 4, 4]
        patch_size: Size of patches
        h: Image height
        w: Image width
        
    Returns:
        Tuple of (colors, ray_origins, ray_directions, xy_norm, projection_matrices)
    """
    b_v, c = image.shape[:2]
    device = image.device
    
    start_patch_center = patch_size / 2.0
    y, x = torch.meshgrid(
        torch.arange(h // patch_size) * patch_size + start_patch_center,
        torch.arange(w // patch_size) * patch_size + start_patch_center,
        indexing="ij",
    )
    y, x = y.to(device), x.to(device)
    
    x_flat = x[None, :, :].expand(b_v, -1, -1).reshape(b_v, -1)
    y_flat = y[None, :, :].expand(b_v, -1, -1).reshape(b_v, -1)
    
    # Sample colors at patch centers
    ray_color = F.grid_sample(
        image,
        torch.stack([x_flat / w * 2.0 - 1.0, y_flat / h * 2.0 - 1.0], dim=2).reshape(
            b_v, -1, 1, 2
        ),
        align_corners=False,
    ).squeeze(-1).permute(0, 2, 1).contiguous()
    
    # Compute normalized coordinates
    ray_xy_norm = torch.stack([x_flat / w, y_flat / h], dim=2)
    
    # Compute projection matrices
    K_norm = torch.eye(3, device=device).unsqueeze(0).repeat(b_v, 1, 1)
    K_norm[:, 0, 0] = fxfycxcy[:, 0] / w
    K_norm[:, 1, 1] = fxfycxcy[:, 1] / h
    K_norm[:, 0, 2] = fxfycxcy[:, 2] / w
    K_norm[:, 1, 2] = fxfycxcy[:, 3] / h
    
    w2c = torch.inverse(c2w)
    proj_mat = torch.bmm(K_norm, w2c[:, :3, :4])
    proj_mat = proj_mat.reshape(b_v, 12)
    proj_mat = proj_mat / (proj_mat.norm(dim=1, keepdim=True) + 1e-6)
    proj_mat = proj_mat.reshape(b_v, 3, 4)
    proj_mat = proj_mat * proj_mat[:, 0:1, 0:1].sign()
    
    # Compute ray directions
    x_norm = (x_flat - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y_norm = (y_flat - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z_norm = torch.ones_like(x_norm)
    
    ray_d = torch.stack([x_norm, y_norm, z_norm], dim=2)
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)
    
    return ray_color, ray_o, ray_d, ray_xy_norm, proj_mat


# =============================================================================
# Main Classes
# =============================================================================



class TransformInput(nn.Module):
    """
    Transform input data for feeding into the transformer network.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, data_batch: edict, patch_size: Optional[int] = None) -> edict:
        """
        Transform input images to rays and other representations.
        
        Args:
            data_batch: Input data batch
            patch_size: Optional patch size for patch-based processing
            
        Returns:
            Transformed input data
        """
        self._validate_input(data_batch)
        
        image, fxfycxcy, c2w, index = (
            data_batch.image, data_batch.fxfycxcy, 
            data_batch.c2w, data_batch.index
        )
        
        b, v, c, h, w = image.size()
        
        # Reshape for processing
        image_flat = image.reshape(b * v, c, h * w)
        fxfycxcy_flat = fxfycxcy.reshape(b * v, 4)
        c2w_flat = c2w.reshape(b * v, 4, 4)
        
        # Compute normalized coordinates for full image
        xy_norm = self._compute_normalized_coordinates(b, v, h, w, image.device)
        
        # Compute camera rays
        ray_o, ray_d, ray_d_cam = compute_camera_rays(
            fxfycxcy_flat, c2w_flat, h, w, image.device
        )
        
        # Process patches if patch_size is provided
        patch_data = self._process_patches(
            image_flat, fxfycxcy_flat, c2w_flat, patch_size, h, w, b, v, c
        ) if patch_size is not None else (None, None, None, None, None)
        
        # Reshape outputs
        ray_o = ray_o.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        ray_d = ray_d.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        ray_d_cam = ray_d_cam.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        
        return edict(
            image=image,
            ray_o=ray_o,
            ray_d=ray_d,
            ray_d_cam=ray_d_cam,
            fxfycxcy=fxfycxcy,
            c2w=c2w,
            index=index,
            xy_norm=xy_norm,
            ray_color_patch=patch_data[0],
            ray_o_patch=patch_data[1],
            ray_d_patch=patch_data[2],
            ray_xy_norm_patch=patch_data[3],
            proj_mat=patch_data[4],
        )
    
    def _validate_input(self, data_batch: edict) -> None:
        """Validate input data dimensions."""
        assert data_batch.image.dim() == 5, f"image dim should be 5, got {data_batch.image.dim()}"
        assert data_batch.fxfycxcy.dim() == 3, f"fxfycxcy dim should be 3, got {data_batch.fxfycxcy.dim()}"
        assert data_batch.c2w.dim() == 4, f"c2w dim should be 4, got {data_batch.c2w.dim()}"
    
    def _compute_normalized_coordinates(self, b: int, v: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Compute normalized coordinates for the full image."""
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        
        y_norm = (y + 0.5) / h * 2 - 1
        x_norm = (x + 0.5) / w * 2 - 1
        
        return torch.stack([x_norm, y_norm], dim=0)[None, None, :, :, :].expand(b, v, -1, -1, -1)
    
    def _process_patches(
        self, 
        image: torch.Tensor, 
        fxfycxcy: torch.Tensor, 
        c2w: torch.Tensor,
        patch_size: int, 
        h: int, 
        w: int, 
        b: int, 
        v: int, 
        c: int
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Process patch-based data if patch_size is provided."""
        ray_color, ray_o, ray_d, ray_xy_norm, proj_mat = sample_patch_rays(
            image.reshape(b * v, c, h, w), fxfycxcy, c2w, patch_size, h, w
        )
        
        n_patch = ray_color.size(1)
        
        return (
            ray_color.reshape(b, v, n_patch, c),
            ray_o.reshape(b, v, n_patch, 3),
            ray_d.reshape(b, v, n_patch, 3),
            ray_xy_norm.reshape(b, v, n_patch, 2),
            proj_mat.reshape(b, v, 3, 4),
        )
