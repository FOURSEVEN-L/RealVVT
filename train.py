import os
import numpy as np
import math
import random
import time
import logging
import copy
import inspect
import argparse
from datetime import datetime
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict
from einops import repeat
from einops import rearrange
from PIL import Image

import diffusers
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.utils import load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.image_processor import VaeImageProcessor

from src.models.unet_2d_condition import UNet2DConditionModel
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from src.data.dataset_vivid_vitonhd import StableVideoAnimationViViD
from src.models.utils import (
    add_noise, 
    pixel2latent, 
    encode_image, 
    scaling_with_edm_noise,
    rand_log_normal,
    update_ema,
)
from src.utils.ptp_utils import AttentionStore, aggregate_attention, register_attention_control

from src.models.pose_guider import PoseGuider
from src.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from src.models.mutual_self_attention_svd import ReferenceAttentionControl
from src.pipelines.pipeline_svd_pose_animation import StableVideoDiffusionPipeline
from src.utils.utils import (
    Color, 
    NoColor, 
    seed_everything,
    save_videos_from_pil,
    save_videos_grid,
    read_frames,
    get_fps,
)


logger = get_logger(__name__, log_level="INFO")



class AnimateNet(nn.Module):
    def __init__(
        self,
        unet: UNetSpatioTemporalConditionModel,
        reference_net: UNet2DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.unet = unet
        self.reference_net = reference_net
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    
    def forward(
        self,
        input_latents,
        ref_input_latents,
        timesteps,
        clip_image_embeddings,
        mask,
        masked_image_latents,
        added_time_ids,
        pose_images,
        uncond_fwd: bool = False,
    ):  
        pose_images = pose_images.transpose(1, 2)  # (bs, c, f, H, W)

        pose_fea = self.pose_guider(pose_images).transpose(1, 2) # (bs, f, c, H, W)

        #mask concat
        latent_model_input = torch.cat([input_latents, masked_image_latents, mask], dim=2)
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)

            self.reference_net(
                ref_input_latents,
                ref_timesteps,
                clip_image_embeddings,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
    
        # Predict the noise residual
        model_pred = self.unet(
            latent_model_input, 
            timesteps, 
            clip_image_embeddings,
            added_time_ids=added_time_ids,
            pose_cond_fea=pose_fea,
        ).sample
        return model_pred



def main(cfg):  
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    logging_dir = f"{save_dir}/tensorboard"
    os.makedirs(logging_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=cfg.solver.mixed_precision,
        log_with="tensorboard",
        project_dir=save_dir, 
    )
    color = Color
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    generator = torch.Generator(
        device=accelerator.device).manual_seed(cfg.seed)
    #todo
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="unet2",
        variant="fp16",
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=True,
    )
    
    reference_net = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_sd_model_name_or_path,
        subfolder="unet", 
        variant="fp16",
    ).to(device="cuda")
    

    

    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda")
    


    # target network
    target_unet = copy.deepcopy(unet)
    target_reference_net = copy.deepcopy(reference_net)
    target_pose_guider = copy.deepcopy(pose_guider)


    feature_extractor = CLIPImageProcessor.from_pretrained(
        cfg.pretrained_model_name_or_path, 
        subfolder="feature_extractor",
        variant="fp16")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        cfg.pretrained_model_name_or_path, 
        subfolder="image_encoder",
        variant="fp16")

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        cfg.pretrained_model_name_or_path, 
        subfolder="vae",
        variant="fp16")
        

    if cfg.unet_checkpoint_path:
        logger.info("Loading existing unet weights")
        # load pretrained weights
        unet.load_state_dict(
            torch.load(cfg.unet_checkpoint_path, map_location="cpu"),
            strict=True,
        )
        
        target_unet.load_state_dict(
            torch.load(cfg.unet_checkpoint_path, map_location="cpu"),
            strict=True,
        )
    
    if cfg.pose_guider_checkpoint_path:
        logger.info("Loading existing pose_guider weights")
        # load pretrained weights
        pose_guider.load_state_dict(
            torch.load(cfg.pose_guider_checkpoint_path, map_location="cpu"),
            strict=True,
        )
        
        target_pose_guider.load_state_dict(
            torch.load(cfg.pose_guider_checkpoint_path, map_location="cpu"),
            strict=True,
        )
 
    
    if cfg.reference_net_checkpoint_path:
        logger.info("Loading existing unet weights")
        # load pretrained weights
        reference_net.load_state_dict(
            torch.load(cfg.reference_net_checkpoint_path, map_location="cpu"),
            strict=True,
        )
        
        target_reference_net.load_state_dict(
            torch.load(cfg.reference_net_checkpoint_path, map_location="cpu"),
            strict=True,
        )

    unet_params = [
        p.numel() for n, p in unet.named_parameters()
    ]
    logger.info(f"unet {sum(unet_params) / 1e6}M-parameter")

    reference_net_params = [
        p.numel() for n, p in reference_net.named_parameters()
    ]
    logger.info(f"reference_net {sum(reference_net_params) / 1e6}M-parameter")

    pose_guider_params = [
        p.numel() for n, p in pose_guider.named_parameters()
    ]
    logger.info(f"pose_guider {sum(pose_guider_params) / 1e6}M-parameter")
    

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    target_unet.requires_grad_(False)
    target_reference_net.requires_grad_(False)
    target_pose_guider.requires_grad_(False)
    
    unet.requires_grad_(True)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_net.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    pose_guider.requires_grad_(True)


    reference_control_writer = ReferenceAttentionControl(
        reference_net,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )
    
    target_reference_control_writer = ReferenceAttentionControl(
        target_reference_net,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    target_reference_control_reader = ReferenceAttentionControl(
        target_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = AnimateNet(
        unet,
        reference_net,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )
    
    controller = AttentionStore()
    register_attention_control(net, controller)

    target_net = AnimateNet(
        target_unet,
        target_reference_net,
        target_pose_guider,
        target_reference_control_writer,
        target_reference_control_reader,
    )
    
    weight_dtype = torch.float32
    if cfg.solver.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.solver.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    target_unet.to(accelerator.device, dtype=weight_dtype)
    target_reference_net.to(accelerator.device, dtype=weight_dtype)
    target_pose_guider.to(accelerator.device, dtype=weight_dtype)



    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            reference_net.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        reference_net.enable_gradient_checkpointing()

    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate
    
    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        logger.info("use_8bit_adam")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = StableVideoAnimationViViD(
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
        data_meta_paths=cfg.data.meta_paths,
    )
    
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        shuffle=True,
        seed=cfg.seed
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Prepare everything with our `accelerator`.
    net, train_dataloader, lr_scheduler, optimizer = accelerator.prepare(
        net,
        train_dataloader,
        lr_scheduler,
        optimizer)

    logger.info(net)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) /  cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            "tensorboard",
        )
    
    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes 
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {num_train_epochs}")
    logger.info(f"Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        reference_net.train()
        pose_guider.train()


        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            if global_step == 0:
                if torch.distributed.get_rank() == 0:
                    torch.cuda.memory._record_memory_history()

            t_data = time.time() - t_data_start

            
            pixel_values = batch["pixel_values"].to(weight_dtype).to(
                accelerator.device, non_blocking=True
            )

            pixel_values_pose = batch["pixel_values_pose"].to(weight_dtype).to(
                accelerator.device, non_blocking=True
            )

            pixel_values_video_mask = batch["pixel_values_video_mask"].to(weight_dtype).to(
                accelerator.device, non_blocking=True
            )

            ref_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype).to(
                accelerator.device, non_blocking=True
            )

            # ref_mask_pixel_values = batch["pixel_values_ref_mask"].to(weight_dtype).to(
            #     accelerator.device, non_blocking=True
            # )
            
            # CLIP image embeddings
            latents = pixel2latent(pixel_values, vae)

            encoder_hidden_states = encode_image(
                ref_pixel_values.float(), 
                feature_extractor, 
                image_encoder,
            )
            P_mean = cfg.noise_scheduler.P_mean
            P_std = cfg.noise_scheduler.P_mean
            sigma_data = cfg.noise_scheduler.sigma_data

            # Predict the noise residual and compute loss
            noisy_latents, sigma = add_noise(latents, P_mean, P_std)

            c_skip, c_out, c_in, c_noise = scaling_with_edm_noise(sigma)

            scaled_inputs = noisy_latents * c_in
        
            bsz = latents.shape[0]
        
            noise_aug_strength = rand_log_normal(
                shape=[bsz, 1, 1, 1, 1], 
                loc=-3.0, 
                scale=0.5,
                device=latents.device,
                dtype=latents.dtype)
            
            noisy_condition = ref_pixel_values.unsqueeze(dim=1)
            
            rnd_normal = torch.randn(
                [bsz, 1, 1, 1, 1], 
                device=latents.device,
                dtype=latents.dtype)

            noisy_condition = noisy_condition + noise_aug_strength * rnd_normal

            noisy_condition_latents = pixel2latent(noisy_condition, vae) / vae.config.scaling_factor
        
            ref_conditional_latents = vae.encode(ref_pixel_values).latent_dist.mode() * 0.18215
            
            # mask latent
            
            width=cfg.data.train_width  
            height=cfg.data.train_height    
            mask=pixel_values_video_mask   
           
            
            if pixel_values.shape[2] == 4:
                # if images are in latent space, we can't mask it
                masked_image = None
                
            else:
                masked_image = pixel_values * (mask < 0.5)   
            
            # 7. Prepare mask latent variables
            
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            
            mask = torch.nn.functional.interpolate(
                mask, size=(mask.shape[2], height // vae_scale_factor, width // vae_scale_factor)
            )
            
            mask = mask.to(device=latents.device, dtype=latents.dtype) 
            
            masked_image_latents = pixel2latent(masked_image, vae)  
             
            
            # classifier-free guidance
            random_null_ratio = cfg.random_null_ratio   
       
            if random_null_ratio > 0.0:
                p = random.random()
                uncond_fwd = p < 2 * random_null_ratio
            
                encoder_hidden_states = torch.zeros_like(encoder_hidden_states) if uncond_fwd else encoder_hidden_states
                
                uncond_latents_fwd = p > random_null_ratio and p < 3 * random_null_ratio
                noisy_condition_latents = torch.zeros_like(noisy_condition_latents) if uncond_latents_fwd else noisy_condition_latents

            # Repeat the condition latents for each frame so we can concatenate them with the noise
            noisy_condition_latents = noisy_condition_latents.repeat(1, latents.shape[1], 1, 1, 1)

            motion_bucket_id = cfg.data.motion_bucket_id
            fps = cfg.data.fps
            
            motion_score = torch.tensor([motion_bucket_id]).repeat(bsz).to(latents.device)
            fps = torch.tensor([fps]).repeat(bsz).to(latents.device)
            added_time_ids = torch.stack([fps, motion_score, noise_aug_strength.reshape(bsz)], dim=1)

            
    
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            
           
            with torch.cuda.amp.autocast(enabled=True):
                model_pred = net(
                    scaled_inputs, 
                    ref_conditional_latents,
                    c_noise, 
                    encoder_hidden_states, 
                    mask=mask,
                    masked_image_latents=masked_image_latents,
                    added_time_ids=added_time_ids,
                    pose_images=pixel_values_pose,
                    uncond_fwd=uncond_fwd)
            
                pred = model_pred * c_out + c_skip * noisy_latents
                loss = torch.mean((weight.float() * (pred.float() - latents.float()) ** 2))

            loss = loss / cfg.solver.gradient_accumulation_steps
            # Gather the losses across all processes for logging (if we use distributed training).
            # todo
            
            loss_idx = random.choices([0, 1], weights=[1, 1])[0]    
         
            # Agnostic Mask-Guided Attention for Clothing Consistency
            if loss_idx == 1:
                attention_map_dict = aggregate_attention(
                    attention_store=controller,
                    from_where=("up", "down", "mid"),
                    is_cross=True,
                    select=0
                )
                
                len_attn_dict = len(attention_map_dict)
                if len_attn_dict>0:
                    for key, value in attention_map_dict.items():
                   
                        attention_map_dict[key] *= 100      
                        
                        attention_flat = attention_map_dict[key]
                       
                        softmax_attention = torch.nn.functional.softmax(attention_flat, dim=-1) 
                        attention_map_dict[key] = rearrange(softmax_attention, "h w b c -> b h w c")

                    bg_mask = rearrange(mask, "b f c h w -> (b f) c h w")
                    bg_mask_16 = F.interpolate(bg_mask, size=(16,12),
                                            mode='bilinear', antialias=True) 
                    bg_mask_16 = bg_mask_16.squeeze() 
                    
                    ob_pos = -1
                
                  
                    
                    att_mask_loss_ob_neg_16 = attention_map_dict[16][bg_mask_16 == 0][:, ob_pos]  
                  
                    att_mask_loss_ob_neg = att_mask_loss_ob_neg_16.mean()    
                  
                    
                    
                    att_mask_loss_ob_pos_16 = max(0, 1. - attention_map_dict[16][bg_mask_16 > 0][:, ob_pos].max())

                    att_mask_loss_ob_pos = att_mask_loss_ob_pos_16
                    
                    att_loss = cfg.data.obpos_loss_weight * att_mask_loss_ob_pos + cfg.data.obneg_loss_weight * att_mask_loss_ob_neg
                    
                    loss = loss + att_loss
                
            
            avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
            train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
 
            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    trainable_params,
                    cfg.solver.max_grad_norm,
                )

            if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                
                target_reference_control_reader.clear()
                target_reference_control_writer.clear()
                
                # ema
                update_ema(target_unet.parameters(), unet.parameters(), cfg.target_ema_decay)
                update_ema(target_reference_net.parameters(), reference_net.parameters(), cfg.target_ema_decay)
                update_ema(target_pose_guider.parameters(), pose_guider.parameters(), cfg.target_ema_decay)

                
                progress_bar.update(1)
                if global_step == 0:
                    if torch.distributed.get_rank() == 0:
                        torch.cuda.memory._dump_snapshot("gpu_mem.pickle")
                        torch.cuda.memory._record_memory_history(enabled=None)

                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    # save models
                    logger.info("save model")
                    if accelerator.is_main_process:
                        save_models(accelerator, net, save_dir, global_step, cfg, prefix='model')
                        save_models(accelerator, target_net, save_dir, global_step, cfg, prefix='model_ema')

                # Validation
                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    logger.info(
                        f"Running validation... \n Generating {len(cfg.val_data.ref_images)} videos."
                    )
                    
                    # def is_compiled_module(module):
                    #     if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
                    #         return False
                    #     return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)
                    
                    # def my_unwrap_model(model):
                    #     is_compiled = is_compiled_module(model)
                    #     if is_compiled:
                    #         compiled_model = model
                    #         model = model._origmod
                    #     if isinstance(model, UNetSpatioTemporalConditionModel):
                    #     # if is_UNetSpatioTemporalConditionModel_available() and isinstance(model, UNetSpatioTemporalConditionModel):
                    #         model = model.base_model.model
                    ori_net = accelerator.unwrap_model(target_net)
                    # ori_net = my_unwrap_model(target_net)
                    ori_net = copy.deepcopy(ori_net)
                    del controller
                    controller = AttentionStore()
                    register_attention_control(net, controller)
    
                    # The models need unwrapping because for compatibility in distributed training mode.
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(
                        cfg.pretrained_model_name_or_path,
                        vae=vae,
                        image_encoder=accelerator.unwrap_model(image_encoder),
                        unet=ori_net.unet,
                        pose_guider=ori_net.pose_guider,
                        reference_net=ori_net.reference_net,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=False)

                    # run inference
                    val_save_dir = os.path.join(
                        save_dir, "validation_images")

                    if not os.path.exists(val_save_dir):
                        os.makedirs(val_save_dir)
                        
                    generator = torch.manual_seed(cfg.seed) 
                                         
                    with torch.autocast(
                        str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                    ):

                        for val_img_idx in range(len(cfg.val_data.ref_images)):
                            img_path = cfg.val_data.ref_images[val_img_idx]
                            pose_path = cfg.val_data.drive_poses[val_img_idx]
                            video_path = cfg.val_data.pixel_values[val_img_idx]
                            pixel_values_video_mask = cfg.val_data.pixel_values_video_mask[val_img_idx]
                            pixel_values_agnostic = cfg.val_data.pixel_agnostic[val_img_idx]
                            
                            fps = 15
                            pose_images = read_frames(pose_path) 
                            
                            video_images = read_frames(video_path)
                            video_mask_images = read_frames(pixel_values_video_mask)
                            agnostic_images = read_frames(pixel_values_agnostic)
                            logger.info(f"inference on {img_path}, pose is {pose_path}"
                            )

                            
                            ref_image_pil = load_image(img_path)
                            nframes = 30
                            start_frames = 10
                            infer_width, infer_height = cfg.val_data.infer_width, cfg.val_data.infer_height

                            logger.info(f"infer width {infer_width}, infer height {infer_height}")

                            simple_transform = transforms.ToTensor()
                            pose_transform = transforms.Compose(
                                [transforms.Resize((infer_height, infer_width)), transforms.ToTensor()]
                            )
                            ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
                            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
                            ref_image_tensor = repeat(
                                ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=nframes
                            )
                            
                            pose_images_list = []
                            pose_tensor_list = []
                            for pose_img in pose_images[start_frames:(start_frames+nframes)]:
                                
                                pose_tensor_list.append(pose_transform(pose_img))
                               
                                pose_images_list.append(pose_img)
                                
                            # video_mask
                            agnostic_list=[]
                            for agnostic_image_pil in agnostic_images[start_frames:(start_frames+nframes)]:
                                agnostic_list.append(agnostic_image_pil)
                            
                            video_images_list = []
                            video_tensor_list = []
                            
                            for video_img in video_images[start_frames:(start_frames+nframes)]:
                                video_tensor_list.append(pose_transform(video_img))
                                
        
                            video_tensor = torch.stack(video_tensor_list, dim=0)
                            video_tensor = video_tensor.transpose(0,1)
                          
                            video_tensor = video_tensor.unsqueeze(0)
                            
                            video_mask_images_list = []
                            video_mask_tensor_list = []
                            
                            for video_mask_img in video_mask_images[start_frames:(start_frames+nframes)]:
                                video_mask = pose_transform(video_mask_img) 
                                video_mask_slice = video_mask[:1]   
                              
                                video_mask_tensor_list.append(video_mask_slice)
                                video_mask_images_list.append(video_mask_img)

                            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
                            pose_tensor = pose_tensor.transpose(0, 1)
                            pose_tensor = pose_tensor.unsqueeze(0)

                            video_tensor = torch.stack(video_tensor_list, dim=0)  # (f, c, h, w)
                            video_tensor = video_tensor.transpose(0, 1)
                            video_tensor = video_tensor.unsqueeze(0)
                            
                            
                            # 8 steps
                            video_frames = pipeline(
                                ref_image_pil.resize((infer_width, infer_height)),
                                pose_images_list,
                                agnostic_list,
                                video_images_list,
                                video_mask_images_list,
                                height=infer_height,
                                width=infer_width,
                                num_frames=nframes,
                                decode_chunk_size=cfg.val_data.decode_chunk_size,
                                motion_bucket_id=cfg.val_data.motion_bucket_id,
                                fps=cfg.val_data.fps,
                                noise_aug_strength=cfg.val_data.noise_aug_strength,
                                generator=generator,
                                num_inference_steps=cfg.val_data.num_inference_steps,
                                min_guidance_scale=cfg.val_data.min_guidance_scale,
                                max_guidance_scale=cfg.val_data.max_guidance_scale,
                                tile_overlap=cfg.val_data.tile_overlap,
                                tile_size=cfg.val_data.tile_size,
                            ).frames[0]
                            video_np = np.stack([np.asarray(frame) / 255.0 for frame in video_frames])
                            video = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0)
        
                            video_shape = video.shape
                            
        
                            video_result = F.interpolate(
                                video, 
                                size=(video_shape[2], infer_height, infer_width), 
                                mode='trilinear', 
                                align_corners=False).cpu()

                            video = torch.cat([ref_image_tensor, video_tensor, video_result], dim=0)

                            out_file_path = os.path.join(
                                val_save_dir,
                                f"step_{global_step}_val_img_{val_img_idx}_step-{cfg.val_data.num_inference_steps}.mp4",
                            )

                            save_videos_grid(
                                video, 
                                out_file_path,
                                n_rows=3, 
                                fps=fps)

                    del pipeline
                    del ori_net
    
            logs = {    
                f"{color.cyan}step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
        
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
                
        # save model after each epoch
        if epoch + 1 % cfg.save_model_epoch_interval == 0:
            if accelerator.is_main_process:
                save_models(accelerator, net, save_dir, global_step, cfg, prefix='model')
                save_models(accelerator, target_net, save_dir, global_step, cfg, prefix='model_ema')
            
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()



def save_models(accelerator, net, save_dir, global_step, cfg, prefix):
    unwarp_net = accelerator.unwrap_model(net)
    save_checkpoint(
        unwarp_net.pose_guider,
        save_dir,
        global_step,
        name=f"{prefix}_pose_guider",
        total_limit=cfg.total_limit,
    )
    save_checkpoint(
        unwarp_net.reference_net,
        save_dir,
        global_step,
        name=f"{prefix}_reference_net",
        total_limit=cfg.total_limit,
    )
    save_checkpoint(
        unwarp_net.unet,
        save_dir,
        global_step,
        name=f"{prefix}_unet",
        total_limit=cfg.total_limit,
    )
    

def save_checkpoint(model, save_dir, ckpt_num, name="appearance_net", total_limit=None):
    save_path = os.path.join(save_dir, f"{name}-{ckpt_num}.pth")
    logger.info(f"save models on steps {ckpt_num} for {name} on {save_path}")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.endswith(".pth")]
        checkpoints = [d for d in checkpoints if name in d]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    # accelerator.save(model, save_path)
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    main(config)
