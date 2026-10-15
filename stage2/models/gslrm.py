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

# gslrm.py is under the Adobe Research License. Copyright 2025 Adobe Inc.

"""
GSLRM (Gaussian Splatting Large Reconstruction Model)

This module implements a transformer-based model for generating 3D Gaussian splats
from multi-view images. The model uses a combination of image tokenization,
transformer processing, and Gaussian splatting for novel view synthesis.

Classes:
    Renderer: Handles Gaussian splatting rendering operations
    GaussiansUpsampler: Converts transformer tokens to Gaussian parameters
    GSLRM: Main model class that orchestrates the entire pipeline
"""

import copy
from typing import List, Tuple

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
from einops.layers.torch import Rearrange

# Local imports
from ..rendering import GaussianModel
from .transform_data import TransformInput
from .utils_transformer import (
    TransformerBlock,
    _init_weights,
)

class Renderer(nn.Module):
    """
    Holds the Gaussian model template used for reconstruction outputs.
    """
    
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        
        # Initialize Gaussian model with scaling modifier
        self.scaling_modifier = config.model.gaussians.get("scaling_modifier", None)
        self.gaussians_model = GaussianModel(
            config.model.gaussians.sh_degree, 
            self.scaling_modifier
        )
        
        print(f"Renderer initialized with scaling_modifier: {self.scaling_modifier}")


class GaussiansUpsampler(nn.Module):
    """
    Converts transformer output tokens to Gaussian splatting parameters.
    
    Takes high-dimensional transformer features and projects them to the
    concatenated Gaussian parameter space (xyz + features + scaling + rotation + opacity).
    """
    
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        
        # Layer normalization before final projection
        self.layernorm = nn.LayerNorm(config.model.transformer.d, bias=False)
        
        # Calculate output dimension for Gaussian parameters
        sh_dim = (config.model.gaussians.sh_degree + 1) ** 2 * 3
        gaussian_param_dim = 3 + sh_dim + 3 + 4 + 1  # xyz + features + scaling + rotation + opacity
        
        # Check upsampling factor (currently only supports 1x)
        upsample_factor = config.model.gaussians.upsampler.upsample_factor
        if upsample_factor > 1:
            raise NotImplementedError("GaussiansUpsampler only supports upsample_factor=1")
        
        # Linear projection to Gaussian parameters
        self.linear = nn.Linear(
            config.model.transformer.d,
            gaussian_param_dim,
            bias=False,
        )

    def forward(
        self, 
        gaussians: torch.Tensor,  # [b, n_gaussians, d]
        images: torch.Tensor      # [b, l, d] (unused but kept for interface compatibility)
    ) -> torch.Tensor:           # [b, n_gaussians, gaussian_param_dim]
        """
        Convert transformer tokens to Gaussian parameters.
        
        Args:
            gaussians: Transformer output tokens for Gaussians
            images: Image tokens (unused but kept for compatibility)
            
        Returns:
            Raw Gaussian parameters (before conversion to final format)
        """
        upsample_factor = self.config.model.gaussians.upsampler.upsample_factor
        if upsample_factor > 1:
            raise NotImplementedError("GaussiansUpsampler only supports upsample_factor=1")
        
        return self.linear(self.layernorm(gaussians))

    def to_gs(self, gaussians: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Convert raw Gaussian parameters to final format.
        
        Args:
            gaussians: Raw Gaussian parameters [b, n_gaussians, param_dim]
            
        Returns:
            Tuple of (xyz, features, scaling, rotation, opacity)
        """
        sh_dim = (self.config.model.gaussians.sh_degree + 1) ** 2 * 3
        
        # Split concatenated parameters
        xyz, features, scaling, rotation, opacity = gaussians.split(
            [3, sh_dim, 3, 4, 1], dim=2
        )
        
        # Reshape features to proper spherical harmonics format
        features = features.reshape(
            features.size(0),
            features.size(1),
            (self.config.model.gaussians.sh_degree + 1) ** 2,
            3,
        )
        
        # Apply activation functions with specific biases
        # Scaling: exp(x - 2.3) clamped to prevent too large values.
        #
        # The cap is configurable because it is not only a quality knob -- it sets a
        # hard ceiling on how many screen tiles one Gaussian can touch, and the
        # rasteriser's binning buffer is sized by the SUM of those tile counts. At
        # the default -1.20 (max scale 0.301) one scene produces ~218M
        # (Gaussian, tile) pairs, needing ~36.2 GiB in a SINGLE allocation on a
        # 47.5 GiB card that already holds ~11 GiB: a 20 MB margin. A run either
        # squeaks through or dies at a random step, and which one happens is not
        # reproducible -- see PROGRESS 85.
        #
        # Default unchanged, so every existing run path stays bit-identical.  A
        # YAML null disables the cap for controlled diagnostics; production runs
        # should only use that after measuring rasteriser peak memory.
        scale_clamp = self.config.model.gaussians.get('scaling_clamp_max', -1.20)
        scaling = scaling - 2.3
        if scale_clamp is not None:
            scaling = scaling.clamp(max=float(scale_clamp))
        
        # Opacity: sigmoid(x - 2.0) to get values in [0, 1]
        opacity = opacity - 2.0
        
        return xyz, features, scaling, rotation, opacity


class GSLRM(nn.Module):
    """
    Gaussian Splatting Large Reconstruction Model.
    
    A transformer-based model that generates 3D Gaussian splats from multi-view images.
    The model processes input images through tokenization, transformer layers, and
    generates Gaussian parameters for novel view synthesis.
    
    Architecture:
    1. Image tokenization with patch-based encoding
    2. Transformer processing with Gaussian positional embeddings
    3. Gaussian parameter generation and upsampling
    4. Gaussian reconstruction for rendering
    """
    
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        
        # Initialize data processing modules
        self._init_data_processors(config)
        
        # Initialize core model components
        self._init_tokenizer(config)
        self._init_positional_embeddings(config)
        self._init_transformer(config)
        self._init_gaussian_modules(config)
        self._init_rendering_modules(config)
        
    
    def _init_data_processors(self, config: edict) -> None:
        """Initialize input transformations."""
        self.input_transformer = TransformInput(config)
    
    def _init_tokenizer(self, config: edict) -> None:
        """Initialize image tokenization pipeline."""
        patch_size = config.model.image_tokenizer.patch_size
        input_channels = config.model.image_tokenizer.in_channels
        hidden_dim = config.model.transformer.d
        
        self.patch_embedder = nn.Sequential(
            Rearrange(
                "batch views channels (height patch_h) (width patch_w) -> (batch views) (height width) (patch_h patch_w channels)",
                patch_h=patch_size,
                patch_w=patch_size,
            ),
            nn.Linear(
                input_channels * (patch_size ** 2),
                hidden_dim,
                bias=False,
            ),
        )
        self.patch_embedder.apply(_init_weights)
    
    def _init_positional_embeddings(self, config: edict) -> None:
        """Initialize positional embeddings for reference/source markers and Gaussians."""
        hidden_dim = config.model.transformer.d
        
        # Optional reference/source view markers
        self.view_type_embeddings = None
        if config.model.get("add_refsrc_marker", False):
            self.view_type_embeddings = nn.Parameter(
                torch.randn(2, hidden_dim)  # [reference_marker, source_marker]
            )
            nn.init.trunc_normal_(self.view_type_embeddings, std=0.02)
        
        # Gaussian positional embeddings
        num_gaussians = config.model.gaussians.n_gaussians
        self.gaussian_position_embeddings = nn.Parameter(
            torch.randn(num_gaussians, hidden_dim)
        )
        nn.init.trunc_normal_(self.gaussian_position_embeddings, std=0.02)
    
    def _init_transformer(self, config: edict) -> None:
        """Initialize transformer architecture."""
        hidden_dim = config.model.transformer.d
        head_dim = config.model.transformer.d_head
        num_layers = config.model.transformer.n_layer
        
        self.input_layer_norm = nn.LayerNorm(hidden_dim, bias=False)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, head_dim)
            for _ in range(num_layers)
        ])
        self.transformer_layers.apply(_init_weights)
    
    def _init_gaussian_modules(self, config: edict) -> None:
        """Initialize Gaussian parameter generation modules."""
        hidden_dim = config.model.transformer.d
        patch_size = config.model.image_tokenizer.patch_size
        sh_degree = config.model.gaussians.sh_degree
        
        # Calculate output dimension for pixel-aligned Gaussians
        # Components: xyz(3) + sh_features((sh_degree+1)^2*3) + scaling(3) + rotation(4) + opacity(1)
        gaussian_param_dim = 3 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
        
        # Gaussian upsampler for transformer tokens
        self.gaussian_upsampler = GaussiansUpsampler(config)
        self.gaussian_upsampler.apply(_init_weights)
        
        # Pixel-aligned Gaussian decoder
        self.pixel_gaussian_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim, bias=False),
            nn.Linear(
                hidden_dim,
                (patch_size ** 2) * gaussian_param_dim,
                bias=False,
            ),
        )
        self.pixel_gaussian_decoder.apply(_init_weights)
    
    def _init_rendering_modules(self, config: edict) -> None:
        """Initialize the Gaussian model template."""
        self.gaussian_renderer = Renderer(config)


    def _create_posed_images_with_plucker(self, input_data: edict) -> torch.Tensor:
        """
        Create posed images by concatenating RGB with Plucker coordinates.
        
        Args:
            input_data: Input data containing images and ray information
            
        Returns:
            Posed images with Plucker coordinates [batch, views, channels, height, width]
        """
        # Normalize RGB to [-1, 1] range
        normalized_rgb = input_data.image[:, :, :3, :, :] * 2.0 - 1.0
        
        if self.config.model.get("use_custom_plucker", False):
            # Custom Plucker: RGB + ray_direction + nearest_points
            ray_origin_dot_direction = torch.sum(
                -input_data.ray_o * input_data.ray_d, dim=2, keepdim=True
            )
            nearest_points = input_data.ray_o + ray_origin_dot_direction * input_data.ray_d
            
            return torch.cat([
                normalized_rgb,
                input_data.ray_d,
                nearest_points,
            ], dim=2)
            
        elif self.config.model.get("use_aug_plucker", False):
            # Augmented Plucker: RGB + cross_product + ray_direction + nearest_points
            ray_cross_product = torch.cross(input_data.ray_o, input_data.ray_d, dim=2)
            ray_origin_dot_direction = torch.sum(
                -input_data.ray_o * input_data.ray_d, dim=2, keepdim=True
            )
            nearest_points = input_data.ray_o + ray_origin_dot_direction * input_data.ray_d
            
            return torch.cat([
                normalized_rgb,
                ray_cross_product,
                input_data.ray_d,
                nearest_points,
            ], dim=2)
            
        else:
            # Standard Plucker: RGB + cross_product + ray_direction
            ray_cross_product = torch.cross(input_data.ray_o, input_data.ray_d, dim=2)
            
            return torch.cat([
                normalized_rgb,
                ray_cross_product,
                input_data.ray_d,
            ], dim=2)
    
    def _add_view_type_embeddings(
        self, 
        image_tokens: torch.Tensor, 
        batch_size: int, 
        num_views: int, 
        num_patches: int, 
        hidden_dim: int
    ) -> torch.Tensor:
        """Add view type embeddings to distinguish reference vs source views."""
        image_tokens = image_tokens.reshape(batch_size, num_views, num_patches, hidden_dim)
        
        # Create view type markers: first view is reference, rest are source
        view_markers = [self.view_type_embeddings[0]] + [
            self.view_type_embeddings[1] for _ in range(1, num_views)
        ]
        view_markers = torch.stack(view_markers, dim=0)[None, :, None, :]  # [1, views, 1, hidden_dim]
        
        # Add markers to image tokens
        image_tokens = image_tokens + view_markers
        return image_tokens.reshape(batch_size, num_views * num_patches, hidden_dim)
    
    def _process_through_transformer(
        self, 
        gaussian_tokens: torch.Tensor, 
        image_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Process combined tokens through the transformer layers."""
        # Combine Gaussian and image tokens
        combined_tokens = torch.cat((gaussian_tokens, image_tokens), dim=1)
        combined_tokens = self.input_layer_norm(combined_tokens)
        
        for layer in self.transformer_layers:
            combined_tokens = layer(combined_tokens)
        
        return combined_tokens
    
    def _apply_hard_pixel_alignment(
        self, 
        pixel_aligned_xyz: torch.Tensor, 
        input_data: edict
    ) -> torch.Tensor:
        """Apply hard pixel alignment to ensure Gaussians align with ray directions."""
        depth_bias = self.config.model.get("depth_preact_bias", 0.0)
        
        # Apply sigmoid activation to depth values
        depth_values = torch.sigmoid(
            pixel_aligned_xyz.mean(dim=2, keepdim=True) + depth_bias
        )
        
        # Apply different depth computation strategies
        if (self.config.model.get("use_aug_plucker", False) or 
            self.config.model.get("use_custom_plucker", False)):
            # For Plucker coordinates: use dot product offset
            ray_origin_dot_direction = torch.sum(
                -input_data.ray_o * input_data.ray_d, dim=2, keepdim=True
            )
            depth_values = (2.0 * depth_values - 1.0) * 1.8 + ray_origin_dot_direction
            
        elif (self.config.model.get("depth_min", -1.0) > 0.0 and 
              self.config.model.get("depth_max", -1.0) > 0.0):
            # Use explicit depth range
            depth_min = self.config.model.depth_min
            depth_max = self.config.model.depth_max
            depth_values = depth_values * (depth_max - depth_min) + depth_min
            
        elif self.config.model.get("depth_reference_origin", False):
            # Reference from ray origin norm
            ray_origin_norm = input_data.ray_o.norm(dim=2, p=2, keepdim=True)
            depth_values = (2.0 * depth_values - 1.0) * 1.8 + ray_origin_norm
            
        else:
            # Default depth computation
            depth_values = (2.0 * depth_values - 1.0) * 1.5 + 2.7
        
        # Compute final 3D positions along rays
        aligned_positions = input_data.ray_o + depth_values * input_data.ray_d
        
        return aligned_positions
    
    
    def _create_gaussian_models_and_stats(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor, 
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        num_pixel_aligned: int,
        num_views: int,
        height: int,
        width: int,
        patch_size: int
    ) -> Tuple[List, torch.Tensor, List[float]]:
        """
        Create Gaussian models for each batch item and compute usage statistics.
        
        Returns:
            Tuple of (gaussian_models, pixel_aligned_positions, usage_statistics)
        """
        gaussian_models = []
        pixel_aligned_positions_list = []
        usage_statistics = []
        
        batch_size = xyz.size(0)
        opacity_threshold = 0.05
        
        for batch_idx in range(batch_size):
            # Create fresh Gaussian model for this batch item
            self.gaussian_renderer.gaussians_model.empty()
            gaussian_model = copy.deepcopy(self.gaussian_renderer.gaussians_model)
            
            # Set Gaussian data
            gaussian_model = gaussian_model.set_data(
                xyz[batch_idx].detach().float(),
                features[batch_idx].detach().float(),
                scaling[batch_idx].detach().float(),
                rotation[batch_idx].detach().float(),
                opacity[batch_idx].detach().float(),
            )
            gaussian_models.append(gaussian_model)

            # Compute usage statistics (fraction of Gaussians above opacity threshold)
            opacity_mask = gaussian_model.get_opacity > opacity_threshold
            usage_ratio = opacity_mask.sum() / opacity_mask.numel()
            if torch.is_tensor(usage_ratio):
                usage_ratio = usage_ratio.item()
            usage_statistics.append(usage_ratio)

            # Extract pixel-aligned positions and reshape
            pixel_xyz = gaussian_model.get_xyz[-num_pixel_aligned:, :]
            pixel_xyz_reshaped = rearrange(
                pixel_xyz,
                "(views height width patch_h patch_w) coords -> views coords (height patch_h) (width patch_w)",
                views=num_views,
                height=height // patch_size,
                width=width // patch_size,
                patch_h=patch_size,
                patch_w=patch_size,
            )
            pixel_aligned_positions_list.append(pixel_xyz_reshaped)
        
        # Stack pixel-aligned positions
        pixel_aligned_positions = torch.stack(pixel_aligned_positions_list, dim=0)
        
        return gaussian_models, pixel_aligned_positions, usage_statistics

    @torch.no_grad()
    def forward(
        self, 
        batch_data: edict, 
        create_visual: bool = False, 
        split_data: bool = False
    ) -> edict:
        """
        Forward pass of the GSLRM model.
        
        Args:
            batch_data: Input batch containing:
                - image: Multi-view images [batch, views, channels, height, width]
                - fxfycxcy: Camera intrinsics [batch, views, 4]
                - c2w: Camera-to-world matrices [batch, views, 4, 4]
            create_visual: Reserved; must be False.
            split_data: Reserved; must be False for five-view reconstruction.
            
        Returns:
            Dictionary containing reconstructed Gaussians and their input representations
        """
        if split_data or create_visual:
            raise ValueError("Inference expects split_data=False and create_visual=False")
        with torch.no_grad():
            input_data = self.input_transformer(batch_data)

            # Prepare posed images with Plucker coordinates [batch, views, channels, height, width]
            posed_images = self._create_posed_images_with_plucker(input_data)

        # Process images through tokenization and transformer
        batch_size, num_views, channels, height, width = posed_images.size()
        
        # Tokenize images into patches
        image_patch_tokens = self.patch_embedder(posed_images)  # [batch*views, num_patches, hidden_dim]
        _, num_patches, hidden_dim = image_patch_tokens.size()
        image_patch_tokens = image_patch_tokens.reshape(
            batch_size, num_views * num_patches, hidden_dim
        )  # [batch, views*patches, hidden_dim]

        # Add view type embeddings if enabled (reference vs source views)
        if self.view_type_embeddings is not None:
            image_patch_tokens = self._add_view_type_embeddings(
                image_patch_tokens, batch_size, num_views, num_patches, hidden_dim
            )

        # Prepare Gaussian tokens with positional embeddings
        gaussian_tokens = self.gaussian_position_embeddings.expand(batch_size, -1, -1)

        # Process through the transformer
        combined_tokens = self._process_through_transformer(
            gaussian_tokens, image_patch_tokens
        )

        # Split back into Gaussian and image tokens
        num_gaussians = self.config.model.gaussians.n_gaussians
        gaussian_tokens, image_patch_tokens = combined_tokens.split(
            [num_gaussians, num_views * num_patches], dim=1
        )
        
        # Generate Gaussian parameters from transformer outputs
        gaussian_params = self.gaussian_upsampler(gaussian_tokens, image_patch_tokens)

        # Generate pixel-aligned Gaussians from image tokens
        pixel_aligned_gaussian_params = self.pixel_gaussian_decoder(image_patch_tokens)
        
        # Calculate Gaussian parameter dimensions
        sh_degree = self.config.model.gaussians.sh_degree
        gaussian_param_dim = 3 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
        
        pixel_aligned_gaussian_params = pixel_aligned_gaussian_params.reshape(
            batch_size, -1, gaussian_param_dim
        )  # [batch, views*pixels, gaussian_params]
        num_pixel_aligned_gaussians = pixel_aligned_gaussian_params.size(1)

        # Combine all Gaussian parameters
        all_gaussian_params = torch.cat((gaussian_params, pixel_aligned_gaussian_params), dim=1)
        
        # Convert to final Gaussian format
        xyz, features, scaling, rotation, opacity = self.gaussian_upsampler.to_gs(all_gaussian_params)

        # Extract pixel-aligned Gaussian positions for processing
        pixel_aligned_xyz = xyz[:, -num_pixel_aligned_gaussians:, :]
        patch_size = self.config.model.image_tokenizer.patch_size
        
        pixel_aligned_xyz = rearrange(
            pixel_aligned_xyz,
            "batch (views height width patch_h patch_w) coords -> batch views coords (height patch_h) (width patch_w)",
            views=num_views,
            height=height // patch_size,
            width=width // patch_size,
            patch_h=patch_size,
            patch_w=patch_size,
        )

        # Apply hard pixel alignment if enabled
        if self.config.model.hard_pixelalign:
            pixel_aligned_xyz = self._apply_hard_pixel_alignment(
                pixel_aligned_xyz, input_data
            )
            
            # Reshape back to flat format and update xyz
            pixel_aligned_xyz_flat = rearrange(
                pixel_aligned_xyz,
                "batch views coords (height patch_h) (width patch_w) -> batch (views height width patch_h patch_w) coords",
                patch_h=patch_size,
                patch_w=patch_size,
            )
            
            # Replace pixel-aligned Gaussians in the full xyz tensor
            xyz = torch.cat(
                (xyz[:, :-num_pixel_aligned_gaussians, :], pixel_aligned_xyz_flat), 
                dim=1
            )

        # Create Gaussian splatting result structure
        gaussian_splat_result = edict(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        # Create Gaussian models for each batch item and compute usage statistics
        gaussian_models, pixel_aligned_positions, usage_statistics = self._create_gaussian_models_and_stats(
            xyz, features, scaling, rotation, opacity, 
            num_pixel_aligned_gaussians, num_views, height, width, patch_size
        )

        # Compile final results
        return edict(
            input=input_data,
            target=None,
            gaussians=gaussian_models,
            pixelalign_xyz=pixel_aligned_positions,
            img_tokens=image_patch_tokens,
            loss_metrics=None,
            render=None,
        )
