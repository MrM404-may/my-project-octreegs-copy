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
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, loadCam

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, logger=None):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 存储相机信息而不是相机对象，实现延迟加载
        self.train_camera_infos = {}
        self.test_camera_infos = {}
        # 存储创建的相机对象
        self.train_cameras = {}
        self.test_cameras = {}

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
        else:
            # 加载迭代时不需要点云处理
            points = None

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 存储相机信息，不立即创建相机对象
        for resolution_scale in self.resolution_scales:
            print("Storing Training Camera Infos")
            self.train_camera_infos[resolution_scale] = scene_info.train_cameras
            print("Storing Test Camera Infos")
            self.test_camera_infos[resolution_scale] = scene_info.test_cameras
            # 初始化相机对象字典
            self.train_cameras[resolution_scale] = {}
            self.test_cameras[resolution_scale] = {}

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
            # 传递相机信息而不是相机对象
            self.gaussians.set_level(points, self, self.resolution_scales, args.dist_ratio, args.init_level, args.levels)
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

    def getTrainCameras(self, scale=None):
        """获取训练相机，支持按分辨率缩放获取"""
        all_cams = []   
        if scale is not None:
            if scale in self.resolution_scales:
                # 确保所有相机都已创建
                self._ensure_cameras_created(scale, 'train')
                all_cams.extend(self.train_cameras[scale].values())
        else:
            for s in self.resolution_scales:
                # 确保所有相机都已创建
                self._ensure_cameras_created(s, 'train')
                all_cams.extend(self.train_cameras[s].values())
        return all_cams

    def getTestCameras(self):
        """获取测试相机"""
        all_cams = []   
        for scale in self.resolution_scales:
            # 确保所有相机都已创建
            self._ensure_cameras_created(scale, 'test')
            all_cams.extend(self.test_cameras[scale].values())
        return all_cams
    
    def _ensure_cameras_created(self, resolution_scale, cam_type):
        """确保指定分辨率和类型的相机已创建"""
        if cam_type == 'train':
            cam_infos = self.train_camera_infos.get(resolution_scale, [])
            cam_dict = self.train_cameras[resolution_scale]
        else:
            cam_infos = self.test_camera_infos.get(resolution_scale, [])
            cam_dict = self.test_cameras[resolution_scale]
        
        # 只创建未创建的相机
        for id, cam_info in enumerate(cam_infos):
            if id not in cam_dict:
                # 创建相机对象，默认不加载图像
                cam = loadCam(self.args, id, cam_info, resolution_scale, load_image=False)
                cam_dict[id] = cam

    def releaseCameraMemory(self, camera_id):
        """Release GPU memory for a specific camera by ID"""
        # 确保相机已创建
        cam = self.get_camera_by_id(camera_id)
        if cam and hasattr(cam, 'release_image_from_gpu'):
            cam.release_image_from_gpu()
            return True
        return False

    def releaseAllTrainCamerasMemory(self):
        """Release GPU memory for all training cameras"""
        released_count = 0
        for scale in self.resolution_scales:
            if scale in self.train_cameras:
                # 确保相机已创建
                self._ensure_cameras_created(scale, 'train')
                for cam in self.train_cameras[scale].values():
                    if hasattr(cam, 'release_image_from_gpu'):
                        if cam.release_image_from_gpu():
                            released_count += 1
        # Only clear cache once after all releases
        if released_count > 0:
            torch.cuda.empty_cache()
        return released_count

    def reloadCameraImage(self, camera_id):
        """Reload image to GPU for a specific camera by ID"""
        # 确保相机已创建
        cam = self.get_camera_by_id(camera_id)
        if cam and hasattr(cam, 'reload_image'):
            cam.reload_image()
            return True
        return False
    
    def ensureCameraLoaded(self, camera_id):
        """Ensure a camera's image is loaded on GPU, loading it if necessary"""
        # 确保相机已创建
        cam = self.get_camera_by_id(camera_id)
        if cam:
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
    
    def get_camera_by_id(self, camera_id):
        """根据相机ID获取相机对象，支持延迟加载"""
        # 检查训练相机
        for scale in self.resolution_scales:
            if scale in self.train_cameras:
                # 确保相机已创建
                self._ensure_cameras_created(scale, 'train')
                for cam in self.train_cameras[scale].values():
                    if cam.uid == camera_id:
                        return cam
        # 检查测试相机
        for scale in self.resolution_scales:
            if scale in self.test_cameras:
                # 确保相机已创建
                self._ensure_cameras_created(scale, 'test')
                for cam in self.test_cameras[scale].values():
                    if cam.uid == camera_id:
                        return cam
        return None
    
    def load_cameras_by_ids(self, camera_ids):
        """加载多个相机到GPU"""
        loaded_cameras = []
        for camera_id in camera_ids:
            cam = self.get_camera_by_id(camera_id)
            if cam:
                if hasattr(cam, 'is_image_loaded') and not cam.is_image_loaded():
                    if hasattr(cam, 'reload_image'):
                        cam.reload_image()
                loaded_cameras.append(cam)
        return loaded_cameras
    
    def release_images(self):
        """释放所有相机的GPU内存"""
        self.releaseAllTrainCamerasMemory()
        # 也释放测试相机的内存
        for scale in self.resolution_scales:
            if scale in self.test_cameras:
                for cam in self.test_cameras[scale]:
                    if hasattr(cam, 'release_image_from_gpu'):
                        cam.release_image_from_gpu()
        torch.cuda.empty_cache()