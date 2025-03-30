# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


def draw_pose_kpts24(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose_kpts24(canvas, candidate, subset)

    canvas = util.draw_handpose_kpts24(canvas, hands)

    canvas = util.draw_facepose_kpts24(canvas, faces)

    return canvas
    

class DWposeDetector:
    def __init__(self):        
        self.pose_estimation = Wholebody("cuda:0")

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            detected_map = draw_pose(pose, H, W)
            return detected_map, pose
    

    def estimate(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            return candidate, subset


    def estimate_kpts24(self, oriImg, single_person=True):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation.get_kpts24(oriImg, single_person)
            return candidate, subset


    def smooth_poses(self, candidate_list, subset_list, avg=10):

        weight_norm = {}
        weight_norm[5] = np.array(
            [0.05, 0.25, 0.4, 0.25, 0.05])
        weight_norm[10] = np.array(
            [0.02, 0.05, 0.09, 0.15, 0.19, 0.19, 0.15, 0.09, 0.05, 0.02])

        nframes = len(candidate_list)

        new_candidate_list = []
        for idx in range(nframes):
            left_idx = max(0, idx-5)
            right_idx = min(nframes-1, idx+4)
            if right_idx - (idx+4) < 0:
                weight = weight_norm[avg][left_idx - (idx-5) : right_idx - (idx+4)]
                scores = subset_list[left_idx - (idx-5) : right_idx - (idx+4)]
            else:
                weight = weight_norm[avg][left_idx - (idx-5):]
                scores = subset_list[left_idx - (idx-5):]
            weight /= np.sum(weight)
            new_candidate = np.sum(candidate_list[left_idx: right_idx+1] * weight[:,np.newaxis, np.newaxis], axis=0)
            #print(new_candidate.shape)
            new_candidate_list.append(new_candidate)
        new_candidate_list = np.array(new_candidate_list)
        
        return new_candidate_list, subset_list
    

    def draw_kpts24(self, candidate, subset, H, W, single_person=True):

        with torch.no_grad():
            # candidate, subset = self.pose_estimation.get_kpts24(oriImg, single_person)
            if single_person:
                candidate = candidate[0:1]
                subset = subset[0:1]
            nums, keys, locs = candidate.shape #1,133,3
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:24].copy()      #add foot
            body = body.reshape(nums*24, locs)
            score = subset[:,:24]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(24*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            # foot = candidate[:,18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:,113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces, size=(W,H))

            return draw_pose_kpts24(pose, H, W), pose

    
    def det_and_draw_kpts24(self, oriImg, single_person=True):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation.get_kpts24(oriImg, single_person)
            nums, keys, locs = candidate.shape #1,133,3
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:24].copy()      #add foot
            body = body.reshape(nums*24, locs)
            score = subset[:,:24]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(24*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            #foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces, size=(W,H))

            return draw_pose_kpts24(pose, H, W), pose