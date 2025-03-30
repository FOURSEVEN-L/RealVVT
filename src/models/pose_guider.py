from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module



class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        guidance_input_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        attention_num_heads: int = 8,
    ):
        super().__init__()
        self.guidance_input_channels = guidance_input_channels
        self.conv_in = InflatedConv3d(
            guidance_input_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]

            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )

            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, condition):
        embedding = self.conv_in(condition)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
            
        embedding = self.conv_out(embedding)

        return embedding

