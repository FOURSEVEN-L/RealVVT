from typing import Any, Dict, Optional

import torch
from einops import rearrange

from src.diffusers.utils import logging
from src.diffusers.models.attention import BasicTransformerBlock


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def torch_dfs(model: torch.nn.Module):
    """
    Perform a depth-first search (DFS) traversal on a PyTorch model's neural network architecture.

    This function recursively traverses all the children modules of a given PyTorch model and returns a list
    containing all the modules in the model's architecture. The DFS approach starts with the input model and
    explores its children modules depth-wise before backtracking and exploring other branches.

    Args:
        model (torch.nn.Module): The root module of the neural network to traverse.

    Returns:
        list: A list of all the modules in the model's architecture.
    """
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            batch_size,
            fusion_blocks=fusion_blocks,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        logger.info(f"mode is {MODE} and do_classifier_free_guidance {do_classifier_free_guidance}")
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt

        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states if self.only_cross_attention else None
                    ),
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=(
                            encoder_hidden_states if self.only_cross_attention else None
                        ),
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    if len(self.bank) > 0:
                        video_length = norm_hidden_states.shape[0] // self.bank[0].shape[0]
                        bank_fea = [
                            rearrange(
                                d.unsqueeze(1).repeat(1, video_length, 1, 1),
                                "b t l c -> (b t) l c",
                            )
                            for d in self.bank
                        ]
                    else:
                        bank_fea = []
                    shuffled_indices = torch.randperm(16)
                    rearrange_hidden_states = rearrange(norm_hidden_states, "(b t) l c -> b t l c", t = 16)
                    shuffled_hidden_states = rearrange_hidden_states[:, shuffled_indices, :, :]
                    other_hidden_states = rearrange(shuffled_hidden_states, "b t l c -> (b t) l c")
                    modify_norm_hidden_states = torch.cat(
                        [norm_hidden_states] + [other_hidden_states] + bank_fea, dim=1
                    )
                    # modify_norm_hidden_states = torch.cat(
                    #     [norm_hidden_states] + bank_fea, dim=1
                    # )

                    hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )
                    if do_classifier_free_guidance:
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )

                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states
        
        if self.fusion_blocks == "midup":
            attn_modules = [
                module
                for module in (
                    torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                )
                if isinstance(module, BasicTransformerBlock)
            ]
        elif self.fusion_blocks == "full":
            attn_modules = [
                module
                for module in torch_dfs(self.unet)
                if isinstance(module, BasicTransformerBlock)
            ]
        attn_modules = sorted(
            attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
        )
        logger.info(f"{MODE} attn_modules hacker nums {len(attn_modules)}")
        
        for i, module in enumerate(attn_modules):
            module._original_inner_forward = module.forward
            if isinstance(module, BasicTransformerBlock):
                module.forward = hacked_basic_transformer_inner_forward.__get__(
                    module, BasicTransformerBlock
                )

            module.bank = []
            module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer):
        if self.fusion_blocks == "midup":
            reader_attn_modules = [
                module
                for module in (
                    torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                )
                if isinstance(module, BasicTransformerBlock)
            ]
            writer_attn_modules = [
                module
                for module in (
                    torch_dfs(writer.unet.mid_block)
                    + torch_dfs(writer.unet.up_blocks)
                )
                if isinstance(module, BasicTransformerBlock)
            ]
        elif self.fusion_blocks == "full":
            reader_attn_modules = [
                module
                for module in torch_dfs(self.unet)
                if isinstance(module, BasicTransformerBlock)
            ]
            writer_attn_modules = [
                module
                for module in torch_dfs(writer.unet)
                if isinstance(module, BasicTransformerBlock)
            ]
        reader_attn_modules = sorted(
            reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
        )
        writer_attn_modules = sorted(
            writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
        )
        for r, w in zip(reader_attn_modules, writer_attn_modules):
            r.bank = [v.clone() for v in w.bank]
            # w.bank.clear()

    def clear(self):
        if self.fusion_blocks == "midup":
            reader_attn_modules = [
                module
                for module in (
                    torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                )
                if isinstance(module, BasicTransformerBlock)
            ]
        elif self.fusion_blocks == "full":
            reader_attn_modules = [
                module
                for module in torch_dfs(self.unet)
                if isinstance(module, BasicTransformerBlock)
            ]
        reader_attn_modules = sorted(
            reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
        )
        for r in reader_attn_modules:
            r.bank.clear()
