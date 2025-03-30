import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from typing import Union, Optional, Dict, Any, Tuple


class DiscriminatorHead(nn.Module):
    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, 1, 1),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(inplace=True), # use LeakyReLu instead of GELU shown in the paper to save memory
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, 1, 1),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(inplace=True), # use LeakyReLu instead of GELU shown in the paper to save memory
        )

        self.conv_out = nn.Conv2d(input_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        unet,
        num_h_per_head=4,
        adapter_channel_dims=[320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320],
    ):
        super().__init__()
        self.unet = unet
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DiscriminatorHead(adapter_channel)
                        for _ in range(self.num_h_per_head)
                    ]
                )
                for adapter_channel in adapter_channel_dims
            ]
        )

    def _forward(self, sample, timestep, encoder_hidden_states, added_time_ids, pose_cond_fea):
        with torch.no_grad():
            features = self.unet(
                sample, 
                timestep, 
                encoder_hidden_states,
                added_time_ids,
                pose_cond_fea=pose_cond_fea,
                return_feat=True)
        
        assert self.head_num == len(features)
        outputs = []
        for feature, head in zip(features, self.heads):
            for h in head:
                outputs.append(h(feature))
        return outputs

    def forward(self, flag, *args):
        if flag == "d_loss":
            return self.d_loss(*args)
        elif flag == "g_loss":
            return self.g_loss(*args)
        else:
            assert 0, "not supported"

    def d_loss(
        self, 
        sample_fake, 
        sample_real, 
        timestep, 
        encoder_hidden_states, 
        added_time_ids, 
        pose_cond_fea, 
        weight,
    ):
        loss = 0.0
        fake_outputs = self._forward(
            sample_fake.detach(), timestep, encoder_hidden_states, added_time_ids, pose_cond_fea
        )
        real_outputs = self._forward(
            sample_real.detach(), timestep, encoder_hidden_states, added_time_ids, pose_cond_fea
        )
        for fake_output, real_output in zip(fake_outputs, real_outputs):
            loss += (
                torch.mean(weight * torch.relu(fake_output.float() + 1))
                + torch.mean(weight * torch.relu(1 - real_output.float()))
            ) / (self.head_num * self.num_h_per_head)
        return loss

    def g_loss(
        self, 
        sample_fake, 
        timestep, 
        encoder_hidden_states, 
        added_time_ids, 
        pose_cond_fea, 
        weight,
    ):
        loss = 0.0
        fake_outputs = self._forward(sample_fake, timestep, encoder_hidden_states, added_time_ids, pose_cond_fea)
        for fake_output in fake_outputs:
            loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / (
                self.head_num * self.num_h_per_head
            )
        return loss

    def feature_loss(
        self, 
        sample_fake, 
        sample_real, 
        timestep, 
        encoder_hidden_states, 
        added_time_ids, 
        pose_cond_fea, 
        weight
    ):
        loss = 0.0
        
        features_fake = self.unet(
            sample_fake, timestep, encoder_hidden_states, added_time_ids,
            pose_cond_fea=pose_cond_fea,
            return_feat=True
        )
        features_real = self.unet(
            sample_real.detach(), timestep, encoder_hidden_states, added_time_ids,
            pose_cond_fea=pose_cond_fea,
            return_feat=True
        )
        for feature_fake, feature_real in zip(features_fake, features_real):
            loss += torch.mean((feature_fake - feature_real) ** 2) / (self.head_num)
        return loss
