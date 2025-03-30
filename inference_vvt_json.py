import os
import cv2
import json
import random
import time
from datetime import datetime
import torch.nn.functional as F

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
from torchvision import transforms

import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    AutoencoderKL
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from src.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

from src.pipelines.pipeline_svd_pose_animation import StableVideoDiffusionPipeline
from src.models.pose_guider import PoseGuider
from src.schudulers.scheduling_euler_discrete import EulerDiscreteScheduler
from src.schudulers.scheduling_edm_euler import EDMEulerScheduler

# from src.dwpose import DWposeDetector, draw_pose

from src.utils.utils import (
    get_fps, 
    read_frames, 
    save_videos_grid,
    save_video_frames,
)



class AnimateController:
    def __init__(
        self,
        config_path="./configs/inference/pose_animation2.yaml",
        weight_dtype=torch.float16,
    ):
        # Read pretrained weights path from config
        self.cfg = OmegaConf.load(config_path)
        self.pipeline = None
        self.weight_dtype = weight_dtype
        # self.dwpose_processor = DWposeDetector()
   
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="vae")
        sd_vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_sd_model_name_or_path, 
            subfolder="vae")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="image_encoder")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="unet",
            variant="fp16",
            low_cpu_mem_usage=False, 
            ignore_mismatched_sizes=True,
        )
        
        reference_net = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_sd_model_name_or_path,
            subfolder="unet",
        ).to(device="cuda")

        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, 
            block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")

        # load pretrained weights
        pose_guider.load_state_dict(
            torch.load(self.cfg.pose_guider_checkpoint_path, map_location="cpu"),
            strict=True,
        )

        # load pretrained weights
        reference_net.load_state_dict(
            torch.load(self.cfg.reference_net_checkpoint_path, map_location="cpu"),
            strict=False,
        )

        unet.load_state_dict(
            torch.load(self.cfg.unet_checkpoint_path, map_location="cpu"),
            strict=True,
        )

        self.checkpoint_step = self.cfg.unet_checkpoint_path.split("/")[-1].split(".")[0].split("-")[1]

        image_encoder.to(self.weight_dtype)
        vae.to(self.weight_dtype)
        sd_vae.to(self.weight_dtype)
        pose_guider.to(self.weight_dtype)
        reference_net.to(self.weight_dtype)
        unet.to(self.weight_dtype)

        # noise_scheduler = EDMEulerScheduler(
        #     sigma_min=0.002,
        #     sigma_max=80,
        #     sigma_data=1.0,
        #     rho=7.0,
        # )
        val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="scheduler")
        
        val_noise_scheduler = EDMEulerScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="scheduler")


        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            unet=unet,
            reference_net=reference_net,
            image_encoder=image_encoder,
            vae=vae,
            pose_guider=pose_guider,
        )





        pipe = pipe.to("cuda", dtype=self.weight_dtype)
        
        self.pipeline = pipe


 
    def animate(
        self,
        ref_img_path, 
        densepose_video_path,
        video_mask_path,
        video_agnostic_path,
        video_path,
        height=512,
        width=384,
        num_inference_steps=25,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        overlap=2,
        fps=6,
        noise_aug_strength=0.02,
        frames_per_batch=16,
        motion_bucket_id=40,
        decode_chunk_size=8,
        seed=123,
    ):
        generator = torch.manual_seed(seed) 
        
        checkpoint_step = self.checkpoint_step
        print("pipeline done")
        tgt_h = height
        tgt_w = width

        ref_name = Path(ref_img_path).stem
        pose_name = Path(densepose_video_path).stem.replace("_detail", "")

        ref_image_pil = Image.open(ref_img_path).convert("RGB")
        ref_w_ori, ref_h_ori = ref_image_pil.size
        save_dir_before = f"../animateoutput/inf_1107_81/"
        # save_dir = f"../animate/inf_output/{ref_name}_{pose_name}_{checkpoint_step}"
        if not os.path.exists(save_dir_before):
            os.makedirs(save_dir_before, exist_ok=True)

        
        
        

        
        pose_tensor_list = []
        pose_images = read_frames(densepose_video_path) # pose
        src_fps = get_fps(densepose_video_path)
        
        video_images = read_frames(video_path)
        video_mask_images = read_frames(video_mask_path)
        agnostic_images = read_frames(video_agnostic_path)
    

        

        simple_transform = transforms.ToTensor()
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
            
        nframes = frames_per_batch
        start_frame = random.randint(1,60)
        # infer_width, infer_height = width, height
        print(f"nframes = {nframes}, with {src_fps} fps")
        pose_images_list = []
        pose_tensor_list = []
        for pose_img in pose_images[start_frame:(start_frame+nframes)]:
   
            pose_tensor_list.append(pose_transform(pose_img))
  
            pose_images_list.append(pose_img)

        
                

                    
        # video_mask
        agnostic_list=[]
        agnostic_tensor_list=[]
        for agnostic_image_pil in agnostic_images[start_frame:(start_frame+nframes)]:
            agnostic_list.append(agnostic_image_pil)
            agnostic_tensor_list.append(pose_transform(agnostic_image_pil))
        
        video_images_list = []
        video_tensor_list = []
        
        for video_img in video_images[start_frame:(start_frame+nframes)]:
            video_tensor_list.append(pose_transform(video_img))

        video_tensor = torch.stack(video_tensor_list, dim=0)
        video_tensor = video_tensor.transpose(0,1)

        video_tensor = video_tensor.unsqueeze(0)
        
        video_mask_images_list = []
        video_mask_tensor_list = []
        
        for video_mask_img in video_mask_images[start_frame:(start_frame+nframes)]:
            video_mask = pose_transform(video_mask_img) # [3, 960, 640] 他还是那个0011的东西
            video_mask_slice = video_mask[:1]   # [1, 960, 640]
            video_mask_tensor_list.append(video_mask)
            video_mask_images_list.append(video_mask_img)
        mask_tensor = torch.stack(video_mask_tensor_list, dim=0)  # (f, c, h, w)
        mask_tensor = mask_tensor.transpose(0, 1)
        mask_tensor = mask_tensor.unsqueeze(0)
        
        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=nframes
        )

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)
        
        agnostic_tensor = torch.stack(agnostic_tensor_list, dim=0)  # (f, c, h, w)
        agnostic_tensor = agnostic_tensor.transpose(0, 1)
        agnostic_tensor = agnostic_tensor.unsqueeze(0)
        
        start_time = time.time()
        with torch.cuda.amp.autocast(enabled=True):
            frames = self.pipeline(
                image=ref_image_pil.resize((width, height)),
                pose_images=pose_images_list,
                agnostic_list=agnostic_list,
                video_images_list=video_images_list,
                video_mask_image_list=video_mask_images_list,
                height=height,
                width=width,
                num_frames=nframes,
                decode_chunk_size=decode_chunk_size,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                noise_aug_strength=noise_aug_strength,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                tile_overlap=overlap,
                num_inference_steps=num_inference_steps,
                tile_size=frames_per_batch,
            ).frames[0]
            print("svd pose2vid ellapsed: ", (time.time() - start_time) * 1000)

        video_np = np.stack([np.asarray(frame) / 255.0 for frame in frames])
        video = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0)
        
        video_shape = video.shape
        print('video shape:', video_shape)
        print('pose_tensor', pose_tensor.min(), pose_tensor.max())
        print('video', video.min(), video.max())
        
        video_result = F.interpolate(
            video, 
            size=(video_shape[2], height, width), 
            mode='trilinear', 
            align_corners=False).cpu()

        video = torch.cat([ref_image_tensor, agnostic_tensor, video_result], dim=0)
       
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")

        save_videos_grid(
            video, 
            f"{save_dir_before}/contrast_result/{pose_name}.mp4", 
            n_rows=3, 
            fps=src_fps)
        
        save_videos_grid(
            video_tensor, 
            f"{save_dir_before}/video_ori/{pose_name}.mp4", 
            n_rows=1, 
            fps=src_fps)
        
        save_videos_grid(
            agnostic_tensor, 
            f"{save_dir_before}/agnostic/{pose_name}.mp4", 
            n_rows=1, 
            fps=src_fps)
        
        save_videos_grid(
            mask_tensor, 
            f"{save_dir_before}/mask/{pose_name}.mp4", 
            n_rows=1, 
            fps=src_fps)
        
        save_videos_grid(
            video_result, 
            f"{save_dir_before}/video_result/{pose_name}.mp4", 
            n_rows=1, 
            fps=src_fps)

        save_video_frames(
            video_tensor,
            f"{save_dir_before}/video_ori_frames/",
            pose_name)
        
        save_video_frames(
            video_result,
            f"{save_dir_before}/video_result_frames/",
            pose_name)

controller = AnimateController()


if __name__ == "__main__":
    json_file_path = '/nfs/hw-data/ms/AIGC/lisiqi/animate/vvt-dataset/test/vvt_test.json'  # JSON文件路径
    with open(json_file_path, 'r') as f:
        data = json.load(f)  # 加载JSON为Python对象

    # 解析JSON内容，逐个获取ref_img_path, densepose_video_path, video_mask_path
    for item in data:
        ref_img_path = item["cloth_path"]
        densepose_video_path = item["kps_path"]
        video_mask_path = item["video_mask_path"]
        video_agnostic_path = item["video_agnostic_path"]
        video_path = item["video_path"]
        controller.animate(ref_img_path, densepose_video_path, video_mask_path, video_agnostic_path, video_path)
