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
from utils.general_utils import PILtoTorch
from PIL import Image

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image_path, gt_alpha_mask_path,
                 image_name, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_size=None
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

        # 只保存图像路径，不加载图像数据
        self.image_path = image_path
        self.gt_alpha_mask_path = gt_alpha_mask_path
        
        # 存储图像尺寸
        if image_size:
            self.image_width, self.image_height = image_size
        else:
            # 从路径获取图像尺寸
            with Image.open(image_path) as img:
                self.image_width, self.image_height = img.size

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load_image_to_gpu(self):
        """从磁盘加载图像到 GPU"""
        # 从磁盘读取图像
        image = Image.open(self.image_path)
        # 调整分辨率
        from utils.camera_utils import loadCam
        # 这里需要根据 resolution_scale 调整图像大小
        # 简化处理，直接使用原始尺寸
        resized_image_rgb = PILtoTorch(image, (self.image_width, self.image_height))
        
        # 存储到 GPU
        self.original_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0).to(self.data_device)
        
        if self.gt_alpha_mask_path:
            mask = Image.open(self.gt_alpha_mask_path)
            mask_rgb = PILtoTorch(mask, (self.image_width, self.image_height))
            self.original_image *= mask_rgb[:1, ...].to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

    def release_image_from_gpu(self):
        """释放 GPU 中的图像内存"""
        if hasattr(self, 'original_image'):
            del self.original_image
            torch.cuda.empty_cache()
            return True
        return False

    def reload_image(self):
        """重新从磁盘加载图像到 GPU"""
        self.load_image_to_gpu()
        return self.original_image
    
    def is_image_loaded(self):
        """检查图像是否在 GPU 上"""
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

