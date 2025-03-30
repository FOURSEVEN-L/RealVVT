import os
import sys
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import av
import av.filter
import torch
import torchvision
from einops import rearrange
from PIL import Image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
    

def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height
        stream.options = {
            "crf": "23",
            "pix_fmt": "yuv420p",
        }

        for pil_image in pil_images:
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path, fps=10):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    skip_frames = round(video_stream.average_rate / fps)
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            if frame.pts % skip_frames == 0:
                image = Image.frombytes(
                    "RGB",
                    (frame.width, frame.height),
                    frame.to_rgb().to_ndarray(),
                )
                frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

def save_video_frames(video_tensor, base_dir, posename):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    video_tensor = video_tensor.cpu()
    tensor = video_tensor.squeeze(0)
    tensor = rearrange(tensor, "c b h w -> b h w c")
    batchsize  = tensor.size(0)
    for i in range(batchsize):
        frame = tensor[i]
        frame = frame.numpy()
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(frame)
        image.save(os.path.join(base_dir,f'{posename}_{i:04d}.png'))

def load_images_from_folder(folder):
    images = []
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
    }  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        parts = filename.split('_')
        if len(parts) > 1:
            try:
                return int(parts[-1].split('.')[0])  # Extracting the number part
            except ValueError:
                return float('inf')  # In case of non-integer part, place this file at the end
        return float('inf')  # Non-frame files are placed at the end

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)
    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)

    return images


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""