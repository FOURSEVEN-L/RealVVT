import abc
import math
import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List
from einops import rearrange

from src.diffusers.models.attention_processor import Attention

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img

def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs['enabled'] = False

    return torch.cuda.amp.autocast(*args, **kwargs)

class AttendExciteCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet


    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                # map_ = map_.view(map_.size(0), h, w)
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height* width)

        return maps.permute(1, 2, 0).contiguous()  # shape: (heads, tokens, height* width)

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        
        
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        attention_probs = attn.batch_to_head_dim(attention_probs)


        self.attnstore(attention_probs, is_cross, self.place_in_unet)
        attention_probs = attn.head_to_batch_dim(attention_probs)

        hidden_states = attn.head_to_batch_dim(hidden_states)
        hidden_states = torch.bmm(attention_probs, value) 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states

class AttnStoreProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attnstore, place_in_unet):
        self.attention_probs=None
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        is_cross = encoder_hidden_states is not None
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        

        key1 = attn.head_to_batch_dim(key)
        query1 = attn.head_to_batch_dim(query)
        if query1.shape[1] == 192 and key1.shape[1]>100:
            self.attention_probs = attn.get_attention_scores(query1, key1, attention_mask)
        else:
            self.attention_probs = hidden_states

        self.attnstore(self.attention_probs, is_cross, self.place_in_unet)
        del key1
        del query1
        
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states




def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0

    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        # attn_procs[name] = AttendExciteCrossAttnProcessor(
        #     attnstore=controller, place_in_unet=place_in_unet
        # )
        attn_procs[name] = AttnStoreProcessor2_0(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

def register_unt_attention_control(unet, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn2.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
     
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        if attn.shape[1] == 192 and attn.shape[2] == 576:  # avoid memory overhead

            self.step_store[key].append(attn)   
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0

import torch.nn.functional as F
from collections import defaultdict

# def aggregate_attention(attention_store: AttentionStore,
#                         res: int,
#                         from_where: List[str],
#                         is_cross: bool,
#                         select: int) -> torch.Tensor:
#     """ Aggregates the attention across the different layers and heads at the specified resolution. """
#     out = defaultdict(list)
#     attention_maps = attention_store.get_average_attention()
#     num_pixels = res ** 2
#
#
#     for location in from_where:
#         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
#             # if item.shape[1] == num_pixels:
#             res_tmp = int(item.shape[1]**0.5)
#             if res_tmp in [16, 32, 64]:
#                 cross_maps = item.reshape(2, -1, res_tmp, res_tmp, item.shape[-1])[select].contiguous()
#
#                 # if res != res_tmp:
#                 #     cross_maps = cross_maps.permute(0, 3, 1, 2)
#                 #     cross_maps = F.interpolate(cross_maps, size=(res, res), mode='bilinear')
#                 #     cross_maps = cross_maps.permute(0, 2, 3, 1)
#
#                 out[res_tmp].append(cross_maps)
#
#     for key, value in out.items():
#         out[key] = torch.cat(out[key], dim=0).contiguous()
#
#     # out = out.sum(0) / out.shape[0]
#
#     return out

def aggregate_attention(attention_store: AttentionStore,
                        from_where: List[str],
                        is_cross: bool,
                        select: int):
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = defaultdict(list)
    out_mix = defaultdict(list)
    
    attention_maps = attention_store.get_average_attention()

    for location in from_where:
        
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:

            res_tmp_h = 16
            res_tmp_w = 12
             
            map = rearrange(item, "(b head) (res_tmp_h res_tmp_w) c -> head res_tmp_h res_tmp_w b c", res_tmp_h = res_tmp_h, head=20)

            out[res_tmp_h].append(map.contiguous())      

    for key, value in out.items():
       
        out[key] = torch.cat(out[key], dim=0).contiguous()

        
        
        out[key] = out[key].sum(0) / out[key].shape[0]

        out_key = out[key]

    return out

def aggregate_attention_single(attention_store: AttentionStore,
                        from_where: List[str],
                        is_cross: bool,
                        select: int):
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = defaultdict(list)
    out_mix = defaultdict(list)
    attention_maps = attention_store.get_average_attention()

    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:

            res_tmp = int(item.shape[1]**0.5)
            if res_tmp in [16, 32, 64]:
                map = item.reshape(1, -1, res_tmp, res_tmp, item.shape[-1])
                out[res_tmp].append(map[0].contiguous())

    for key, value in out.items():
        out[key] = torch.cat(out[key], dim=0).contiguous()

        out[key] = out[key].sum(0) / out[key].shape[0]

    return out

