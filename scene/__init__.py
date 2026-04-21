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

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, logger=None, batch_size=None):
        """
        :param path: Path to colmap scene main folder.
        :param batch_size: 批量加载相机的批次大小（可选，主要用于支持区域训练）
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
        self.batch_size = batch_size  # 保存批次大小参数

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.camera_uid_map = {}  # 新增：相机 uid 到相机对象的映射，方便快速查找

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.ds)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.random_background, args.white_background,  args.eval, ply_path=ply_path)
        else:
            scene_info = sceneLoadTypeCallbacks["City"](args.source_path, args.random_background, args.white_background, args.eval, args.ds, undistorted=args.undistorted)

        self.gaussians.set_appearance(len(scene_info.train_cameras))
        
        if not self.loaded_iter:
            points = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        # 新增：建立相机 uid 到相机对象的映射（只处理训练相机，用于区域训练）
        for resolution_scale in self.resolution_scales:
            for cam in self.train_cameras[resolution_scale]:
                self.camera_uid_map[cam.uid] = cam

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            print("Load Voxel Size: ", self.gaussians.voxel_size)
            print("Load Standard Dist: ", self.gaussians.standard_dist)
        else:
            if args.random_background:
                logger.info("Using random background")
            elif args.white_background:
                logger.info("Using white background")
            else:
                logger.info("Using black background")
            points = torch.unique(points, dim=0)
            self.gaussians.set_level(points, self.train_cameras, self.resolution_scales, args.dist_ratio, args.init_level, args.levels)
            self.gaussians.create_from_pcd(points, self.cameras_extent, logger)

    def save_ply(self, pcd, ratio, path):
        points = torch.tensor(pcd.points[::ratio]).float().cuda()
        colors = torch.tensor(pcd.colors[::ratio]).float().cuda()
        storePly(path, points.cpu().numpy(), colors.cpu().numpy())
        return points

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self):
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.train_cameras[scale])
        return all_cams

    def getTestCameras(self):
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.test_cameras[scale])
        return all_cams

    def releaseCameraMemory(self, camera_id):
        """Release GPU memory for a specific camera by ID"""
        # Release from train cameras
        for scale in self.resolution_scales:
            if scale in self.train_cameras:
                for cam in self.train_cameras[scale]:
                    if cam.uid == camera_id:
                        if hasattr(cam, 'release_image_from_gpu'):
                            cam.release_image_from_gpu()
                            return True
        # Release from test cameras
        for scale in self.resolution_scales:
            if scale in self.test_cameras:
                for cam in self.test_cameras[scale]:
                    if cam.uid == camera_id:
                        if hasattr(cam, 'release_image_from_gpu'):
                            cam.release_image_from_gpu()
                            return True
        return False

    def releaseAllTrainCamerasMemory(self):
        """Release GPU memory for all training cameras"""
        released_count = 0
        for scale in self.resolution_scales:
            if scale in self.train_cameras:
                for cam in self.train_cameras[scale]:
                    if hasattr(cam, 'release_image_from_gpu'):
                        if cam.release_image_from_gpu():
                            released_count += 1
        # Only clear cache once after all releases
        if released_count > 0:
            torch.cuda.empty_cache()
        return released_count

    def reloadCameraImage(self, camera_id):
        """Reload image to GPU for a specific camera by ID"""
        # Reload from train cameras
        for scale in self.resolution_scales:
            if scale in self.train_cameras:
                for cam in self.train_cameras[scale]:
                    if cam.uid == camera_id:
                        if hasattr(cam, 'reload_image'):
                            cam.reload_image()
                            return True
        # Reload from test cameras
        for scale in self.resolution_scales:
            if scale in self.test_cameras:
                for cam in self.test_cameras[scale]:
                    if cam.uid == camera_id:
                        if hasattr(cam, 'reload_image'):
                            cam.reload_image()
                            return True
        return False
    
    def ensureCameraLoaded(self, camera_id):
        """Ensure a camera's image is loaded on GPU, loading it if necessary"""
        # Check train cameras
        for scale in self.resolution_scales:
            if scale in self.train_cameras:
                for cam in self.train_cameras[scale]:
                    if cam.uid == camera_id:
                        if hasattr(cam, 'is_image_loaded') and not cam.is_image_loaded():
                            if hasattr(cam, 'reload_image'):
                                cam.reload_image()
                        return True
        # Check test cameras
        for scale in self.resolution_scales:
            if scale in self.test_cameras:
                for cam in self.test_cameras[scale]:
                    if cam.uid == camera_id:
                        if hasattr(cam, 'is_image_loaded') and not cam.is_image_loaded():
                            if hasattr(cam, 'reload_image'):
                                cam.reload_image()
                        return True
        return False

    def reloadMultipleCameras(self, camera_ids):
        """Reload images for multiple cameras by ID list"""
        loaded_count = 0
        for camera_id in camera_ids:
            if self.reloadCameraImage(camera_id):
                loaded_count += 1
        return loaded_count

    def releaseMultipleCameras(self, camera_ids):
        """Release GPU memory for multiple cameras by ID list"""
        released_count = 0
        for camera_id in camera_ids:
            if self.releaseCameraMemory(camera_id):
                released_count += 1
        # Only clear cache once after all releases
        if released_count > 0:
            torch.cuda.empty_cache()
        return released_count
    
    # ====================== 新增：区域训练相关的内存管理方法 ======================
    def load_cameras_by_ids(self, camera_ids):
        """
        加载指定 ID 列表的相机图像到 GPU
        :param camera_ids: 要加载的相机 ID 列表
        :return: 加载成功的相机对象列表
        """
        loaded_cameras = []
        for camera_id in camera_ids:
            if camera_id in self.camera_uid_map:
                cam = self.camera_uid_map[camera_id]
                if not cam.is_image_loaded():
                    cam.reload_image()
                loaded_cameras.append(cam)
        return loaded_cameras
    
    def release_images(self, camera_ids=None):
        """
        释放图像内存
        :param camera_ids: 要释放的相机 ID 列表，如果为 None 则释放所有当前已加载的训练相机
        """
        released_count = 0
        if camera_ids is not None:
            # 释放指定 ID 列表的相机
            for camera_id in camera_ids:
                if camera_id in self.camera_uid_map:
                    cam = self.camera_uid_map[camera_id]
                    if cam.release_image_from_gpu():
                        released_count += 1
        else:
            # 释放所有训练相机的图像内存
            for resolution_scale in self.resolution_scales:
                if resolution_scale in self.train_cameras:
                    for cam in self.train_cameras[resolution_scale]:
                        if cam.release_image_from_gpu():
                            released_count += 1
        # 清理一次缓存
        if released_count > 0:
            torch.cuda.empty_cache()
        return released_count
    
    def get_num_batches(self):
        """
        获取当前总批次数（用于渲染阶段）
        :return: 总批次数
        """
        total_cameras = len(self.getTrainCameras())
        if self.batch_size and self.batch_size > 0:
            return (total_cameras + self.batch_size - 1) // self.batch_size
        return 1
    
    def getTrainCameras(self, batch_idx=None):
        """
        获取训练相机列表，支持按批次获取
        :param batch_idx: 批次索引，如果为 None 则获取全部
        :return: 相机对象列表
        """
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.train_cameras[scale])
        
        if batch_idx is not None and self.batch_size and self.batch_size > 0:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_cams))
            return all_cams[start_idx:end_idx]
        
        return all_cams