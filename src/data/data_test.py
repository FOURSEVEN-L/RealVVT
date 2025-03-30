import random
import json
from PIL import Image
from typing import List
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torchvision.utils import save_image

from decord import VideoReader
from utils import getMaxBox_from_pose, extend_rectangle


class StableVideoAnimationDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        random_frame_stride=True,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
       
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.img_size = (height, width)
        self.mask_size = (height // 8, width // 8)
 
        self.random_frame_stride = random_frame_stride

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)

      

        self.simple_transform = transforms.ToTensor()

        self.resize_transform = transforms.Resize(self.img_size, antialias=True)

        self.resize_norm_transform = transforms.Compose([
            transforms.Resize(self.img_size, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    

    def get_sample(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path, fault_tol=1)
        kps_reader = VideoReader(kps_path, fault_tol=1)
        
        # random sample rate
        if self.random_frame_stride:
            get_random_rate = random.randint(1, self.sample_rate)
        else:
            get_random_rate = self.sample_rate
        
        fps = video_reader.get_avg_fps()

        # fps reset
        fps = float(fps / get_random_rate)

        video_length = min(len(video_reader), len(kps_reader))

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * get_random_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
           
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img_pil = Image.fromarray(video_reader[ref_img_idx].asnumpy())
        ref_pose_pil = Image.fromarray(kps_reader[ref_img_idx].asnumpy())

        # transform
        pixel_values_vid = [
            self.simple_transform(img) for img in vid_pil_image_list
        ]
        pixel_values_vid = torch.stack(pixel_values_vid, dim=0)

        pixel_values_pose = [
            self.simple_transform(img) for img in pose_pil_image_list
        ]
        pixel_values_pose = torch.stack(pixel_values_pose, dim=0)
        
        pixel_values_ref_img = self.simple_transform(ref_img_pil)
        pixel_values_ref_pose = self.simple_transform(ref_pose_pil)

        sample = dict(
            fps=fps,
            video_dir=video_path,
            pixel_values=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            pixel_values_ref_pose=pixel_values_ref_pose,
        )

        return sample


    def __getitem__(self, idx):
        while True:
            try:
                sample = self.get_sample(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)
                print("__getitem__[Error:%s]" % (self.vid_meta[idx]['video_path']), str(e)," reroll:", idx)            

        (top, left), (bottom, right) = getMaxBox_from_pose(
            sample['pixel_values_pose'], 
            sample['pixel_values_ref_pose'],
            extend_rect=True,
            scale_ratio=0.2,
        )
        
        img_c, img_h, img_w = sample['pixel_values_ref_img'].shape
        crop_top = random.randint(0, top)
        crop_bottom = random.randint(bottom, img_h)
        if img_h < img_w:
            crop_left = left
            crop_right = right
        else:
            crop_left = random.randint(0, left)
            crop_right = random.randint(right, img_w)

        
        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        tgt_h, tgt_w = self.img_size 

        if (crop_w / crop_h) <= (tgt_w / tgt_h):
            pad_width = int(crop_h * tgt_w / tgt_h) - crop_w
            pad_left = random.randint(0, pad_width + 1)
            pad_right = pad_width - pad_left
            pad_top = pad_bottom = 0
            
            real_crop_left = max(0, crop_left - pad_left)
            real_pad_left = max(0, pad_left - crop_left)
            real_crop_right = min(crop_right + pad_right, img_w)
            real_pad_right = max(0, crop_right + pad_right - img_w)

            real_crop_top = crop_top
            real_pad_top = 0
            real_crop_bottom = crop_bottom
            real_pad_bottom = 0
        else:    
            pad_height = int(crop_w * tgt_h / tgt_w) - crop_h
            pad_top = random.randint(0, pad_height + 1)
            pad_bottom = pad_height - pad_top
            pad_left = pad_right = 0
            
            real_crop_top = max(0, crop_top - pad_top)
            real_pad_top = max(0, pad_top - crop_top)
            real_crop_bottom = min(crop_bottom + pad_bottom, img_h)
            real_pad_bottom = max(0, crop_bottom + pad_bottom - img_h)

            real_crop_left = crop_left
            real_pad_left = 0
            real_crop_right = crop_right
            real_pad_right = 0

        # do crop & pad 
        pixel_values_pose = F.pad(
            sample['pixel_values_pose'][:, :, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            0)
        pixel_values = F.pad(
            sample['pixel_values'][:, :, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            1)
        pixel_values_ref_img = F.pad(
            sample['pixel_values_ref_img'][:, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            1)
        
        pixel_values_ref_pose = F.pad(
            sample['pixel_values_ref_pose'][:, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            0)
        
        # do tranform now 
        pixel_values = self.resize_norm_transform(pixel_values)
        pixel_values_pose = self.resize_transform(pixel_values_pose)
        
        pixel_values_ref_img = self.resize_norm_transform(pixel_values_ref_img)
        pixel_values_ref_pose = self.resize_transform(pixel_values_ref_pose)

        debug = False
        # if debug:
        #     # mask generator
        #     _, _, = getMaxBox_from_pose(
        #         pixel_values_pose, 
        #         pixel_values_ref_pose,
        #         extend_rect=True,
        #         scale_ratio=0.2)
        #     save_image(pixel_values, "exp_outputs" + '/' + f'img.png')
        #     save_image(pixel_values_pose, "exp_outputs" + '/' + f'pose.png')

        sample['pixel_values'] = pixel_values
        sample['pixel_values_pose'] = pixel_values_pose
        sample['pixel_values_ref_img'] = pixel_values_ref_img
        sample['pixel_values_ref_pose'] = pixel_values_ref_pose
        
        return sample
    

    def __len__(self):
        return len(self.vid_meta)
    

class StableVideoAnimationViViD(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        random_frame_stride=True,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
       
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.img_size = (height, width)
        self.mask_size = (height // 8, width // 8)
 
        self.random_frame_stride = random_frame_stride

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)

        # print(f"We have {self.length} videos")

        self.simple_transform = transforms.ToTensor()

        self.resize_transform = transforms.Resize(self.img_size, antialias=True)

        self.resize_norm_transform = transforms.Compose([
            transforms.Resize(self.img_size, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    

    def get_sample(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        video_mask_path = video_meta["video_mask_path"]
        video_agnostic_path = video_meta["video_agnostic_path"]
        cloth_path = video_meta["cloth_path"]
        cloth_mask_path = video_meta["cloth_mask_path"]

        video_reader = VideoReader(video_path, fault_tol=1)
        kps_reader = VideoReader(kps_path, fault_tol=1)
        video_mask_reader = VideoReader(video_mask_path, fault_tol=1)
        # video_agnostic_reader = VideoReader(video_agnostic_path, fault_tol=1)

        
        # random sample rate
        if self.random_frame_stride:
            get_random_rate = random.randint(1, self.sample_rate)
        else:
            get_random_rate = self.sample_rate
        
        fps = video_reader.get_avg_fps()

        # fps reset
        fps = float(fps / get_random_rate)

        video_length = min(len(video_reader), len(kps_reader), len(video_mask_reader))

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * get_random_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        video_mask_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = video_mask_reader[index]
            video_mask_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
           
        
        ref_img_pil = Image.open(cloth_path)
        ref_mask_pil = Image.open(cloth_mask_path)

        # transform
        pixel_values_vid = [
            self.simple_transform(img) for img in vid_pil_image_list
        ]
        pixel_values_vid = torch.stack(pixel_values_vid, dim=0)

        pixel_values_pose = [
            self.simple_transform(img) for img in pose_pil_image_list
        ]
        pixel_values_pose = torch.stack(pixel_values_pose, dim=0)

        pixel_values_video_mask = []
        for img in video_mask_image_list:
            img = self.simple_transform(img)
            mask = img[:1]
            pixel_values_video_mask.append(mask)
        pixel_values_video_mask = torch.stack(pixel_values_video_mask, dim=0)
        
        
        pixel_values_ref_img = self.simple_transform(ref_img_pil)
        pixel_values_ref_mask = self.simple_transform(ref_mask_pil)

        sample = dict(
            fps=fps,
            video_dir=video_path,
            pixel_values=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_video_mask=pixel_values_video_mask,
            pixel_values_ref_img=pixel_values_ref_img,
            pixel_values_ref_mask=pixel_values_ref_mask,
        )

        return sample


    def __getitem__(self, idx):
        # while True:
        #     try:
        #         sample = self.get_sample(idx)
        #         break
        #     except Exception as e:
        #         idx = random.randint(0, self.length - 1)
        #         print("__getitem__[Error:%s]" % (self.vid_meta[idx]['video_path']), str(e)," reroll:", idx)            
        sample = self.get_sample(idx)
        (top, left), (bottom, right) = getMaxBox_from_pose(
            sample['pixel_values_pose'], 
            sample['pixel_values_ref_mask'],
            extend_rect=True,
            scale_ratio=0.2,
        )
        
        img_c, img_h, img_w = sample['pixel_values_ref_img'].shape
        crop_top = random.randint(0, top)
        crop_bottom = random.randint(bottom, img_h)
        if img_h < img_w:
            crop_left = left
            crop_right = right
        else:
            crop_left = random.randint(0, left)
            crop_right = random.randint(right, img_w)

        
        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        tgt_h, tgt_w = self.img_size 

        if (crop_w / crop_h) <= (tgt_w / tgt_h):
            pad_width = int(crop_h * tgt_w / tgt_h) - crop_w
            pad_left = random.randint(0, pad_width + 1)
            pad_right = pad_width - pad_left
            pad_top = pad_bottom = 0
            
            real_crop_left = max(0, crop_left - pad_left)
            real_pad_left = max(0, pad_left - crop_left)
            real_crop_right = min(crop_right + pad_right, img_w)
            real_pad_right = max(0, crop_right + pad_right - img_w)

            real_crop_top = crop_top
            real_pad_top = 0
            real_crop_bottom = crop_bottom
            real_pad_bottom = 0
        else:    
            pad_height = int(crop_w * tgt_h / tgt_w) - crop_h
            pad_top = random.randint(0, pad_height + 1)
            pad_bottom = pad_height - pad_top
            pad_left = pad_right = 0
            
            real_crop_top = max(0, crop_top - pad_top)
            real_pad_top = max(0, pad_top - crop_top)
            real_crop_bottom = min(crop_bottom + pad_bottom, img_h)
            real_pad_bottom = max(0, crop_bottom + pad_bottom - img_h)

            real_crop_left = crop_left
            real_pad_left = 0
            real_crop_right = crop_right
            real_pad_right = 0

        # do crop & pad 
        pixel_values_pose = F.pad(
            sample['pixel_values_pose'][:, :, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            0)
        pixel_values = F.pad(
            sample['pixel_values'][:, :, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            1)
        pixel_values_mask = F.pad(
            sample['pixel_values_video_mask'][:, :, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            0)
        pixel_values_ref_img = F.pad(
            sample['pixel_values_ref_img'][:, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            1)
        
        pixel_values_ref_mask = F.pad(
            sample['pixel_values_ref_mask'][:, real_crop_top:real_crop_bottom, real_crop_left:real_crop_right],
            (real_pad_left, real_pad_right, real_pad_top, real_pad_bottom),
            "constant",
            0)
        
        # do tranform now 
        pixel_values = self.resize_norm_transform(pixel_values)
        pixel_values_pose = self.resize_transform(pixel_values_pose)
        pixel_values_mask = self.resize_transform(pixel_values_mask)
        
        pixel_values_ref_img = self.resize_norm_transform(pixel_values_ref_img)
        pixel_values_ref_mask = self.resize_transform(pixel_values_ref_mask)

        # debug = False
        # if debug:
        #     # mask generator
        #     _, _, = getMaxBox_from_pose(
        #         pixel_values_pose, 
        #         pixel_values_ref_mask,
        #         extend_rect=True,
        #         scale_ratio=0.2)
        #     save_image(pixel_values, "exp_outputs" + '/' + f'img.png')
        #     save_image(pixel_values_pose, "exp_outputs" + '/' + f'pose.png')

        sample['pixel_values'] = pixel_values
        sample['pixel_values_pose'] = pixel_values_pose
        sample['pixel_values_video_mask'] = pixel_values_mask 
        sample['pixel_values_ref_img'] = pixel_values_ref_img
        sample['pixel_values_ref_mask'] = pixel_values_ref_mask
        
        return sample
    

    def __len__(self):
        return len(self.vid_meta)


if __name__=="__main__":    
    json_data_path = [
        "/nfs/hw-data/ms/AIGC/lisiqi/animate/VideoData/dresses/dresses_train.json",
        "/nfs/hw-data/ms/AIGC/lisiqi/animate/VideoData/lower_body/lower_body_train.json",
        "/nfs/hw-data/ms/AIGC/lisiqi/animate/VideoData/upper_body/upper_body_train.json",
    ]
    train_dataset = StableVideoAnimationViViD(
        width=448,
        height=672,
        n_sample_frames=14,
        sample_rate=4,
        data_meta_paths=json_data_path,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    count_iter  = 0 
    # print("dataloader length: ", len(train_dataloader), flush=True)
    
    for step, batch in enumerate(train_dataloader):

        count_iter += 1
        print(
            "count_iter: ", count_iter,
            batch["pixel_values"][0].shape,
            batch["pixel_values_pose"][0].shape,
            batch["pixel_values_ref_img"][0].shape,
        )
    print("---------------- all epoch over ------------")
