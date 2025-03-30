import os
import torch
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image


def getMaxBox_from_pose(
    pixel_values_pose: torch.Tensor,
    ref_image_pose: torch.Tensor,
    extend_rect: bool = False,
    scale_ratio: float = 0.2,
    pose_threshold: float = 0.1,
):
    img_shape = ref_image_pose.shape[-2:]

    ref_gray_pose = ref_image_pose.mean(dim=0)
    ref_mask = (ref_gray_pose > pose_threshold).float()
    nonzero_coords_all = [torch.nonzero(ref_mask)]

    for pixel_values_pose in pixel_values_pose:
        gray_pose = pixel_values_pose.mean(dim=0)
        mask = (gray_pose > pose_threshold).float()
        nonzero_coords_all.append(torch.nonzero(mask))

    nonzero_coords_all = torch.cat(nonzero_coords_all, dim=0)
    if nonzero_coords_all.shape[0] > 0:
        min_coords = nonzero_coords_all.min(dim=0)[0]
        max_coords = nonzero_coords_all.max(dim=0)[0]
        top, left = min_coords.tolist()
        bottom, right = max_coords.tolist()

        top_left = (top, left)
        bottom_right = (bottom, right)

        if extend_rect:
            top_left, bottom_right = extend_rectangle(
                top_left, bottom_right, img_shape, scale_ratio=scale_ratio)
       
        return top_left, bottom_right
    else:
        top_left = (0, 0)
        bottom_right = (img_shape[0], img_shape[1])
        return top_left, bottom_right
        

def get_bounding_box(nonzero_coords):
    h_top = torch.min(nonzero_coords[:, 1])
    h_bottom = torch.max(nonzero_coords[:, 1])
    w_left = torch.min(nonzero_coords[:, 2])
    w_right = torch.max(nonzero_coords[:, 2])
    
    return (h_top, w_left), (h_bottom, w_right)


def extend_rectangle(top_left, bottom_right, img_shape, scale_ratio=0.05):
    """
    top_left: (y1, x1)
    bottom_right: (y2, x2)
    img_shape: (h, w)
    """

    top, left = top_left
    bottom, right = bottom_right

    human_h = bottom - top
    human_w = right - left

    new_top = int(top - scale_ratio / 2.0 * human_h)
    new_left = int(left - scale_ratio / 2.0 * human_w)

    new_bottom = int(bottom + scale_ratio / 2.0 * human_h)
    new_right = int(right + scale_ratio / 2.0 * human_w)

    
    new_top, new_left = max(0, new_top), max(0, new_left)

    new_bottom, new_right = min(img_shape[0], new_bottom), min(img_shape[1], new_right)

    return (new_top, new_left), (new_bottom, new_right)
