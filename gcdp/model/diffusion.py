"""This module contains the implementation of the diffusion model and the vision encoder."""

# The following code was directly taken from the work of Chi et al. (2021) and can be found on
# the repository of the original project: https://github.com/real-stanford/diffusion_policy


# ### **Network**
# Defines a 1D UNet architecture `ConditionalUnet1D`
# as the noies prediction network
# Components
# - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
# - `Downsample1d` Strided convolution to reduce temporal resolution
# - `Upsample1d` Transposed convolution to increase temporal resolution
# - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
# - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
# `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
# `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.


import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from typing import Tuple, Union, Callable


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding for 1D data."""

    def __init__(self, dim):
        """
        Initialize the positional encoding.

        Inputs:
        - dim: the dimension of the positional encoding
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # emb = x[:, None] * emb[None, :]
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """Downsample 1D data by strided convolution."""

    def __init__(self, dim):
        """
        Initialize the downsample layer.

        Inputs:
        - dim: the dimension of the input data
        """
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsample 1D data by transposed convolution."""

    def __init__(self, dim):
        """
        Initialize the upsample layer.

        Inputs:
        - dim: the dimension of the input data
        """
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d block with GroupNorm and Mish activation."""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        """
        Initialize the Conv1d block.

        Inputs:
        - inp_channels: the number of input channels
        - out_channels: the number of output channels
        - kernel_size: the size of the convolutional kernel
        - n_groups: the number of groups for GroupNorm
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Conditional residual block for 1D data."""

    def __init__(
        self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8
    ):
        """
        Initialize the conditional residual block.

        Inputs:
        - in_channels: the number of input channels
        - out_channels: the number of output channels
        - cond_dim: the dimension of the conditioning input
        - kernel_size: the size of the convolutional kernel
        - n_groups: the number of groups for GroupNorm
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels, out_channels, kernel_size, n_groups=n_groups
                ),
                Conv1dBlock(
                    out_channels, out_channels, kernel_size, n_groups=n_groups
                ),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        Forward pass of the conditional residual block.

        Inputs:
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]
        Outputs:
            out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """Conditional UNet for 1D data."""

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=128,
        down_dims=None,
        kernel_size=5,
        n_groups=8,
    ):
        """
        Initialize the Conditional UNet.

        Parameters:
            input_dim: Dim of actions.
            global_cond_dim: Dim of global conditioning applied with FiLM in addition to diffusion step embedding. This is usually obs_horizon * obs_dim diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
            down_dims: Channel size for each UNet level. The length of this array determines number of levels.
            kernel_size: Conv kernel size
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        if down_dims is None:
            down_dims = [512, 1024, 2048]
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        (
                            Downsample1d(dim_out)
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # print(
        #     "number of parameters: {:e}".format(
        #         sum(p.numel() for p in self.parameters())
        #     )
        # )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        Forward pass of the Conditional UNet.

        Parameters:
            x: (B,T,input_dim)
            timestep: (B,) or int, diffusion step
            global_cond: (B,global_cond_dim)
        Returns:
            output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for _, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for _, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


# ### **Vision Encoder**
# @markdown
# Defines helper functions:
# - `get_resnet` to initialize standard ResNet vision encoder
# - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """
    Initialize standard ResNet vision encoder.

    Parameters:
        name: resnet18, resnet34, resnet50
        weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with the output of func.

    Parameters:
        root_module: The module to search for submodules.
        predicate: Return true if the module is to be replaced.
        func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """Replace all BatchNorm layers with GroupNorm."""
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features,
        ),
    )
    return root_module


class DiffusionRgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(
        self,
        cfg: DictConfig,
        #  weights: str | None = None,
    ):
        """Initialize the vision encoder with configuration."""
        super().__init__()
        self.vision_backbone = cfg.model.vision_encoder.name
        self.pretrained_backbone_weights = cfg.model.vision_encoder.weights
        crop_shape = cfg.model.vision_encoder.crop_shape
        crop_is_random = cfg.model.vision_encoder.crop_is_random
        use_group_norm = cfg.model.vision_encoder.use_group_norm
        spatial_softmax_num_keypoints = (
            cfg.model.vision_encoder.spatial_softmax_num_keypoints
        )
        # Set up optional preprocessing.
        if crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    crop_shape
                )
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        # backbone_model = getattr(torchvision.models, self.vision_backbone)(
        #     # weights=self.pretrained_backbone_weights
        #     weights=weights
        # )
        func = getattr(torchvision.models, cfg.model.vision_encoder.name)
        backbone_model = func(weights=self.pretrained_backbone_weights)
        # Note: This assumes that the layer4 feature map is children()[-3]
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if use_group_norm:
            if self.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                ),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.input_shapes` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.input_shapes`.
        image_keys = [
            k
            for k in cfg.model.input_shapes
            if k.startswith("observation.image")
        ]
        # Note: we have a check in the config class to make sure all images have the same shape.
        image_key = image_keys[0]
        dummy_input_h_w = (
            crop_shape
            if crop_shape is not None
            else cfg.input_shapes[image_key][1:]
        )
        dummy_input = torch.zeros(
            size=(1, cfg.model.input_shapes[image_key][0], *dummy_input_h_w)
        )
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=spatial_softmax_num_keypoints
        )
        self.feature_dim = spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(
            spatial_softmax_num_keypoints * 2, self.feature_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vision encoder.

        Parameters:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with the output of func.

    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(
        predicate(m)
        for _, m in root_module.named_modules(remove_duplicate=True)
    )
    return root_module


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al (https://arxiv.org/pdf/1509.06113).

    A minimal port of the robomimic implementation.
    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Initialize the spatial softmax.

        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(
            pos_x.reshape(self._in_h * self._in_w, 1)
        ).float()
        pos_y = torch.from_numpy(
            pos_y.reshape(self._in_h * self._in_w, 1)
        ).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial softmax.

        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints
