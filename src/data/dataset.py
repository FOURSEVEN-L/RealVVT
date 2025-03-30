import os, csv, random, json
import numpy as np
from decord import VideoReader
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class RandomHorizontalFlipVideo(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        if torch.rand(1) < self.p:
            return torch.flip(clip, [3])
        return clip


class WebVid10M(Dataset):
    def __init__(
        self,
        data_meta_paths, 
        video_folder,
        width=1024,
        height=576,
        fps=6, 
        sample_n_frames=16):

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        print(f"We have {self.length} videos")

        self.video_folder = video_folder
        self.fps = fps
        self.sample_n_frames = sample_n_frames

        sample_size = (height, width)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
            transforms.CenterCrop(sample_size),
            RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])


    def get_batch(self, idx):
        video_dict = self.vid_meta[idx]
        videoid = video_dict['img_path']
        text_prompt = video_dict['caption_content']
        video_path = os.path.join(self.video_folder, f"{videoid}")
        video_reader = VideoReader(video_path)

        fps = video_reader.get_avg_fps()
        sample_stride = round(fps/self.fps)

        # sample sample_n_frames frames from videos with stride sample_stride
        video_length = len(video_reader)
        clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

        
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255. # [T, C, H, W] with range [0, 1]
        del video_reader

        return pixel_values, self.fps, videoid, text_prompt


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, fps, videoid, text_prompt= self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values) #[T, C, H, W] with range [-1, 1]
        sample = dict(pixel_values=pixel_values, fps=fps, id=videoid, text_prompt=text_prompt)
        return sample


class ImageDataset(Dataset):
    def __init__(self, data_path):
        filenames = sorted(os.listdir(data_path))
        self.length = len(filenames)
        self.data_path = data_path
        self.filenames = filenames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.data_path, filename)
        sample = dict(path=path, name=filename)
        return sample


class MultiImageDataset(Dataset):
    def __init__(self, data_paths):
        self.paths = []
        for data_path in data_paths:
            filenames = sorted(os.listdir(data_path))
            for filename in filenames:
                path = os.path.join(data_path, filename)
                self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        dataset_name = path.split('/')[-2]
        filename = path.split('/')[-1]
        sample = dict(path=path, dataset_name=dataset_name, name=filename)
        return sample


if __name__ == "__main__":
    meta_paths = [
        "/youtu_xuanyuan_shuzhiren_2906355_cq10/private/xiaobinhu/portrait_video_split_7290_26780.json"
    ]
    video_folder = "/youtu_xuanyuan_shuzhiren_2906355_cq10/private/xiaobinhu/portrait_video_split_7290_26780/"

    train_width = 1024
    train_height = 576
    fps = 6
    n_sample_frames = 14

    train_dataset = WebVid10M(
        data_meta_paths=meta_paths,
        video_folder=video_folder,
        width=train_width,
        height=train_height,
        fps=fps,
        sample_n_frames=n_sample_frames,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    for step, batch in enumerate(train_dataloader):
        print(batch["pixel_values"].shape)
        break