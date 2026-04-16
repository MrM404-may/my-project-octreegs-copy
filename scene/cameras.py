#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution_scale = resolution_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Store image data on CPU to save GPU memory
        self._image_data_cpu = image.cpu().clone()
        self._gt_alpha_mask_cpu = gt_alpha_mask.cpu().clone() if gt_alpha_mask is not None else None
        
        # Store dimensions
        self.image_width = self._image_data_cpu.shape[2]
        self.image_height = self._image_data_cpu.shape[1]

        # Initialize image on GPU
        self._load_image_to_gpu()

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def _load_image_to_gpu(self):
        """Load image from CPU to GPU"""
        self.original_image = self._image_data_cpu.clamp(0.0, 1.0).to(self.data_device)
        
        if self._gt_alpha_mask_cpu is not None:
            self.original_image *= self._gt_alpha_mask_cpu.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

    def release_image_from_gpu(self):
        """Release GPU memory for this camera's image"""
        if hasattr(self, 'original_image'):
            del self.original_image
            torch.cuda.empty_cache()
            return True
        return False

    def reload_image(self):
        """Reload the image from CPU to GPU"""
        self._load_image_to_gpu()
        return self.original_image
    
    def is_image_loaded(self):
        """Check if image is loaded on GPU"""
        return hasattr(self, 'original_image')

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

