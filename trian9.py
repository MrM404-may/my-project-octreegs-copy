
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, 
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
# ====================== 新增：引入分区训练依赖 ======================
from shapely.geometry import Polygon
# =====================================================================
from region import get_camera_ids_by_regions  # 保持原有导入
from send_email import send_mail  # 用于训练完成后发送通知邮件
# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")
# ====================== 全局变量：存储多区域相机池 ======================
REGION_CAMERA_POOLS = {}  # {区域索引: [相机对象列表]}
REGIONS_CONFIG = []       # 存储加载的多区域配置
REGION_ITER_BOUNDS = []   # 存储每个区域的迭代边界 [(start_iter, end_iter), ...]
REGION_LOCAL_ITERS = {}   # 【新增】每个区域的本地迭代计数器 {区域索引: 当前本地迭代数}
TOTAL_TRAIN_ITER = 0      # 总训练迭代数
# ====================== 新增：加载多区域配置函数 ======================
def save_masked_gt(gt_image, mask, camera_filename, region_idx, region_name):
    save_dir = os.path.join(MASK_VIS_ROOT_DIR, f"region_{region_idx}_{region_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    gt_np = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask_expanded = mask.unsqueeze(0).repeat(3, 1, 1)
    masked_gt = gt_image * mask_expanded
    masked_gt_np = (masked_gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    save_path = os.path.join(save_dir, f"{os.path.splitext(camera_filename)[0]}_gt_masked.png")
    cv2.imwrite(save_path, masked_gt_np)
    
    SAVED_CAMERAS[region_idx].add(camera_filename)

# ====================== 核心：ID→文件名→掩码映射 ======================
def load_camera_id_mapping(json_path):
    id_to_filename = {}
    filename_to_id = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            camera_data = json.load(f)
            if isinstance(camera_data, list):
                for cam in camera_data:
                    if 'id' in cam and 'img_name' in cam:
                        cam_id = str(cam['id'])
                        cam_filename = cam['img_name']
                        id_to_filename[cam_id] = cam_filename
                        filename_to_id[cam_filename] = cam_id
    except Exception as e:
        pass
    return id_to_filename, filename_to_id

def load_mask_for_camera(camera_obj, mask_dir, json_path, current_region_idx):
    global CACHED_ID_TO_FILENAME, CACHED_FILENAME_TO_ID, CACHED_MASKS
    cam_filename = camera_obj.image_name
    
    if CACHED_FILENAME_TO_ID is None:
        CACHED_ID_TO_FILENAME, CACHED_FILENAME_TO_ID = load_camera_id_mapping(json_path)
    
    if cam_filename in CACHED_MASKS:
        return CACHED_MASKS[cam_filename]

    cam_id = CACHED_FILENAME_TO_ID.get(cam_filename, os.path.splitext(cam_filename)[0])
    filename_prefix = os.path.splitext(cam_filename)[0]
    mask_filename = f"{filename_prefix}.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    
    if not os.path.exists(mask_path):
        CACHED_MASKS[cam_filename] = None
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        CACHED_MASKS[cam_filename] = None
        return None
    
    mask_tensor = torch.from_numpy(mask).float().cuda() / 255.0
    if len(mask_tensor.shape) > 2:
        mask_tensor = mask_tensor.squeeze()
    
    CACHED_MASKS[cam_filename] = mask_tensor
    return mask_tensor
CACHED_ID_TO_FILENAME = None
CACHED_FILENAME_TO_ID = None
CACHED_MASKS = {}

MASK_VIS_ROOT_DIR = "./mask_visualization"
SAVED_CAMERAS = {}  # 改为动态字典，适配多区域 {区域索引: {文件名集合}}

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
def load_regions_config(regions_json_path):
    """
    加载多区域配置JSON文件
    参数:
        regions_json_path: 区域配置JSON路径
    返回:
        regions_config: 列表，每个元素包含{"name": 区域名, "vertices": 顶点列表, "iterations": 训练迭代数}
    """
    try:
        with open(regions_json_path, 'r', encoding='utf-8') as f:
            regions_config = json.load(f)
            # 验证配置格式
            for i, region in enumerate(regions_config):
                if not all(key in region for key in ['name', 'vertices', 'iterations']):
                    raise ValueError(f"第{i+1}个区域配置缺少必要字段（name/vertices/iterations）")
                if len(region['vertices']) < 3:
                    raise ValueError(f"第{i+1}个区域{region['name']}顶点数不足3个")
                # 打印每个区域的iterations字段值
                print(f"区域 {i+1} ({region['name']}) 的迭代次数: {region['iterations']}")
            return regions_config
    except Exception as e:
        print(f"加载区域配置失败: {e}")
        sys.exit(1)
# =====================================================================

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def merge_region_anchors(model_path):
    """
    拼接所有区域的锚点文件，生成最终的点云文件和MLP文件
    参数:
        model_path: 模型保存路径
    """
    import os
    import numpy as np
    from plyfile import PlyData, PlyElement
    
    # 1. 收集所有区域的锚点文件
    region_anchors_dir = os.path.join(model_path, "region_anchors")
    ply_files = []
    
    for region_dir in os.listdir(region_anchors_dir):
        if region_dir.startswith("region_"):
            ply_file = os.path.join(region_anchors_dir, region_dir, "anchors.ply")
            if os.path.exists(ply_file):
                ply_files.append(ply_file)
    
    if not ply_files:
        print("❌ 没有找到锚点文件")
        return
    
    # 2. 读取并拼接所有锚点文件
    all_anchors = []
    all_feats = []
    all_scalings = []
    all_rotations = []
    all_opacities = []
    all_regions = []
    
    for ply_file in ply_files:
        print(f"📖 读取锚点文件: {ply_file}")
        ply_data = PlyData.read(ply_file)
        vertex_data = ply_data['vertex'].data
        
        # 提取顶点数据
        anchors = np.array([(v['x'], v['y'], v['z']) for v in vertex_data], dtype=np.float32)
        
        # 提取特征值，从f_anchor_feat_0到f_anchor_feat_31中选择前3个作为f0、f1、f2
        feats = np.array([(v['f_anchor_feat_0'], v['f_anchor_feat_1'], v['f_anchor_feat_2']) for v in vertex_data], dtype=np.float32)
        
        # 提取缩放值，使用scale_0、scale_1、scale_2
        scalings = np.array([(v['scale_0'], v['scale_1'], v['scale_2']) for v in vertex_data], dtype=np.float32)
        
        # 提取旋转值，使用rot_0、rot_1、rot_2、rot_3
        rotations = np.array([(v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']) for v in vertex_data], dtype=np.float32)
        
        opacities = np.array([(v['opacity'],) for v in vertex_data], dtype=np.float32)
        regions = np.array([(v['region'],) for v in vertex_data], dtype=np.float32)
        
        all_anchors.append(anchors)
        all_feats.append(feats)
        all_scalings.append(scalings)
        all_rotations.append(rotations)
        all_opacities.append(opacities)
        all_regions.append(regions)
    
    # 拼接所有数据
    all_anchors = np.vstack(all_anchors)
    all_feats = np.vstack(all_feats)
    all_scalings = np.vstack(all_scalings)
    all_rotations = np.vstack(all_rotations)
    all_opacities = np.vstack(all_opacities)
    all_regions = np.vstack(all_regions)
    
    print(f"✅ 拼接完成，总锚点数: {all_anchors.shape[0]}")
    
    # 3. 保存拼接后的点云文件
    output_ply = os.path.join(model_path, "merged_anchors.ply")
    print(f"💾 保存拼接后的点云文件: {output_ply}")
    
    # 创建顶点元素
    vertices = np.array(
        list(zip(
            all_anchors[:, 0], all_anchors[:, 1], all_anchors[:, 2],
            all_feats[:, 0], all_feats[:, 1], all_feats[:, 2],
            all_scalings[:, 0], all_scalings[:, 1], all_scalings[:, 2],
            all_rotations[:, 0], all_rotations[:, 1], all_rotations[:, 2], all_rotations[:, 3],
            all_opacities[:, 0],
            all_regions[:, 0]
        )),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f0', 'f4'), ('f1', 'f4'), ('f2', 'f4'),
            ('s0', 'f4'), ('s1', 'f4'), ('s2', 'f4'),
            ('r0', 'f4'), ('r1', 'f4'), ('r2', 'f4'), ('r3', 'f4'),
            ('opacity', 'f4'),
            ('region', 'f4')
        ]
    )
    
    # 创建PLY数据
    ply_element = PlyElement.describe(vertices, 'vertex')
    ply_data = PlyData([ply_element])
    ply_data.write(output_ply)
    
    print(f"✅ 点云文件保存完成: {output_ply}")
    
    # 4. 保存MLP为PT文件
    print("💾 保存MLP为PT文件...")
    # 加载训练好的GaussianModel
    from scene.gaussian_model import GaussianModel
    from arguments import ModelParams
    import torch
    
    # 创建一个临时的GaussianModel实例
    # 这里需要根据实际的参数来创建
    # 假设我们使用默认参数
    class DummyArgs:
        def __init__(self):
            self.feat_dim = 32
            self.n_offsets = 8
            self.fork = 2
            self.use_feat_bank = True
            self.appearance_dim = 0
            self.add_opacity_dist = True
            self.add_cov_dist = True
            self.add_color_dist = True
            self.add_level = True
            self.visible_threshold = 0.01
            self.dist2level = "progressive"
            self.base_layer = 0
            self.progressive = True
            self.extend = 1.1
            self.num_regions = len(ply_files)  # 区域数量
    
    dummy_args = DummyArgs()
    gaussians = GaussianModel(
        dummy_args.feat_dim, dummy_args.n_offsets, dummy_args.fork, dummy_args.use_feat_bank, dummy_args.appearance_dim,
        dummy_args.add_opacity_dist, dummy_args.add_cov_dist, dummy_args.add_color_dist, dummy_args.add_level,
        dummy_args.visible_threshold, dummy_args.dist2level, dummy_args.base_layer, dummy_args.progressive, dummy_args.extend
    )
    
    # 加载训练好的MLP参数
    # 假设我们从最后一个迭代的检查点加载
    import os
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if os.path.exists(point_cloud_dir):
        # 查找最大的迭代次数
        import re
        iterations = []
        for fname in os.listdir(point_cloud_dir):
            match = re.match(r"iteration_(+)", fname)
            if match:
                iterations.append(int(match.group(1)))
        if iterations:
            max_iter = max(iterations)
            mlp_checkpoint_path = os.path.join(point_cloud_dir, f"iteration_{max_iter}")
            print(f"Loading MLP from: {mlp_checkpoint_path}")
            gaussians.load_mlp_checkpoints(mlp_checkpoint_path)
    
    # 保存MLP为PT文件
    mlp_pt_path = os.path.join(model_path, "mlp_checkpoints.pt")
    print(f"Saving MLP to: {mlp_pt_path}")
    
    # 保存MLP参数
    mlp_state_dict = {
        "mlp_opacity": [mlp.state_dict() for mlp in gaussians.mlp_opacity],
        "mlp_cov": [mlp.state_dict() for mlp in gaussians.mlp_cov],
        "mlp_color": [mlp.state_dict() for mlp in gaussians.mlp_color]
    }
    
    # 如果使用了特征银行，也保存相关参数
    if hasattr(gaussians, 'mlp_feature_bank') and gaussians.use_feat_bank:
        mlp_state_dict["mlp_feature_bank"] = gaussians.mlp_feature_bank.state_dict()
    
    # 如果使用了外观编码，也保存相关参数
    if hasattr(gaussians, 'embedding_appearance') and gaussians.appearance_dim > 0:
        mlp_state_dict["embedding_appearance"] = gaussians.embedding_appearance.state_dict()
    
    torch.save(mlp_state_dict, mlp_pt_path)
    print(f"✅ MLP PT文件保存完成: {mlp_pt_path}")
    
    # 提示用户如何使用这些文件进行渲染
    print("\n📋 使用说明:")
    print(f"   1. 拼接后的点云文件: {output_ply}")
    print("   2. MLP文件: 需要根据模型结构保存")
    print("   3. 渲染时，使用这些文件作为输入")
def monitor_mlp_gradients(gaussians, current_region, iteration, logger=None):
    """监控MLP的梯度"""
    if iteration % 100 == 0:  # 每100次迭代监控一次
        print(f"\n[梯度监控] 迭代 {iteration}，区域 {current_region}")
        
        # 监控当前区域的专家MLP
        for mlp_name, mlp_list in [
            ("opacity_mlp", gaussians.mlp_opacity),
            ("cov_mlp", gaussians.mlp_cov),
            ("color_mlp", gaussians.mlp_color)
        ]:
            if current_region < len(mlp_list):
                mlp = mlp_list[current_region]
                total_norm = 0.0
                for param in mlp.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"  {mlp_name}_{current_region} 梯度范数: {total_norm:.6f}")
                
                # 可选：监控其他区域的MLP梯度（应该接近0）
                for i in range(len(mlp_list)):
                    if i != current_region:
                        other_mlp = mlp_list[i]
                        other_total_norm = 0.0
                        for param in other_mlp.parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2)
                                other_total_norm += param_norm.item() ** 2
                        other_total_norm = other_total_norm ** 0.5
                        if other_total_norm > 1e-6:  # 只有当梯度大于阈值时才打印
                            print(f"  {mlp_name}_{i} 梯度范数: {other_total_norm:.6f} (非当前区域)")
def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    global CACHED_ID_TO_FILENAME, CACHED_FILENAME_TO_ID, CACHED_MASKS, REGION_CAMERA_POOLS, REGIONS_CONFIG, REGION_ITER_BOUNDS, REGION_LOCAL_ITERS, TOTAL_TRAIN_ITER
    first_iter = 1
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
    # 添加batch_size参数，实现批量加载相机（100张/批）
    batch_size = 1
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales, batch_size=batch_size)
    gaussians.training_setup(opt)
    gaussians.set_coarse_interval(opt.coarse_iter, opt.coarse_factor)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

# ====================== 多区域配置加载 (核心改造) ======================
    # 配置路径（可根据需要改为命令行参数）
    JSON_PATH = r"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/cameras.json"
    DEPTH_DIR = r"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/images_metric3"
    MASK_DIR = r"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/images_mark2"
    REGIONS_JSON_PATH = r"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/regions_config.json"  # 多区域配置文件
    CACHE_JSON_PAH = r"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/cache.json"
    MAX_CAMERA_ID = 3000
    MY_CAMERA_INTRINSICS = (
        867.3526294468791,  # fx
        861.2148494110500,  # fy
        650.3807935145040,  # cx
        364.2087458587477   # cy
    )
    iteration_interval = 20000  # 保留原间隔参数，但不再用于计算
    
    # 1. 加载多区域配置
    REGIONS_CONFIG = load_regions_config(REGIONS_JSON_PATH)
    num_regions = len(REGIONS_CONFIG)
    print(f"\n📌 加载到 {num_regions} 个训练区域：")
    for i, region in enumerate(REGIONS_CONFIG):
        print(f"   区域{i+1}: {region['name']} | 训练迭代数: {region['iterations']} | 顶点数: {len(region['vertices'])}")

    # 2. 计算每个区域的迭代边界 + 【初始化本地迭代计数器】
    current_iter = 0
    for i, region in enumerate(REGIONS_CONFIG):
        start_iter = current_iter + 1
        end_iter = current_iter + region['iterations']
        REGION_ITER_BOUNDS.append((start_iter, end_iter))
        REGION_LOCAL_ITERS[i] = 0  # 每个区域初始本地迭代数为0
        current_iter = end_iter
    TOTAL_TRAIN_ITER = current_iter
    print(f"   总训练迭代数: {TOTAL_TRAIN_ITER}")

    # 3. 初始化SAVED_CAMERAS（适配多区域）
    for i in range(num_regions):
        SAVED_CAMERAS[i] = set()

    # ====================== 修复：4. 加载每个区域的相机ID并构建相机池 ======================
    id_to_filename, filename_to_id = load_camera_id_mapping(JSON_PATH)
    
    # 【核心逻辑】
    print("\n正在分析相机所属区域 (基于 JSON 配置)...")
    all_region_ids, _ = get_camera_ids_by_regions(
        json_path=JSON_PATH,
        regions_config=REGIONS_CONFIG,
        depth_images_dir=None,        
        camera_intrinsics=MY_CAMERA_INTRINSICS,
        mask_ratio_threshold=0.3,           
        depth_img_ext='.png',               
        is_camera_raw=False,                
        verbose=True,
        cache_json_path=CACHE_JSON_PAH  
    )
    
    # 构建 Shapely 多边形列表
    REGION_POLYGONS_SHAPELY = []
    for region in REGIONS_CONFIG:
        REGION_POLYGONS_SHAPELY.append(Polygon(region['vertices']))

    # 构建相机池
    print("\n正在构建区域相机池...")
    # 直接使用region.py返回的相机ID列表
    # 确保一个相机ID只能属于一个区域（按区域顺序首次分配）
    assigned_camera_ids = set()
    print("\n区域相机ID列表构建完成...")
    for region_idx, region in enumerate(REGIONS_CONFIG):
        target_ids = set(all_region_ids[region_idx])
        target_ids = {int(rid) for rid in target_ids if int(rid) <= region.get('max_id', MAX_CAMERA_ID)}
        # 排除已经分配给前面区域的相机ID，确保每个相机ID只属于一个区域
        unique_target_ids = target_ids - assigned_camera_ids
        duplicate_count = len(target_ids) - len(unique_target_ids)
        assigned_camera_ids.update(unique_target_ids)
        print(f"   区域{region_idx+1} ({region['name']}) 目标相机ID数量: {len(unique_target_ids)} (去重: {duplicate_count}个)")
        
        # 存储相机ID列表，而不是相机对象
        REGION_CAMERA_POOLS[region_idx] = list(unique_target_ids)
        print(f"   ✅ 区域{region_idx+1} ({region['name']}) 相机ID列表构建完成")
    
    # 相机ID到相机对象的映射将在需要时动态构建

    # 5. 初始化第一个区域
    current_region_id
