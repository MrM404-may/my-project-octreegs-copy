
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
    # 初始化场景
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
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
    current_region_idx = 0
    current_region = REGIONS_CONFIG[current_region_idx]
    current_camera_pool = REGION_CAMERA_POOLS[current_region_idx]
    current_region_polygon = Polygon(current_region['vertices'])
    REGION_LOCAL_ITERS[current_region_idx] = 0  # 初始化第一个区域的本地迭代
    opt.iterations = TOTAL_TRAIN_ITER
    
    # 4. 初始化第一个区域
    current_region_idx = 0
    current_region = REGIONS_CONFIG[current_region_idx]
    current_polygon = Polygon(current_region['vertices'])
    REGION_LOCAL_ITERS[current_region_idx] = 0  # 确保第一个区域本地迭代为0
    print(f"\n🔄 [Init] 激活第一个区域: {current_region['name']} ...")
    gaussians.enter_region(current_polygon)
    # gaussians.restore_initial_state()
    gaussians.set_region(current_region_idx)  
    
    # 生成随机训练序列
    def generate_random_sequence(camera_pool):
        """生成随机训练序列"""
        # camera_pool现在是相机ID列表，直接打乱它
        sequence = camera_pool.copy()
        random.shuffle(sequence)
        return sequence
    
    # 加载当前区域的相机池
    current_camera_pool = REGION_CAMERA_POOLS[current_region_idx]
    # 生成随机训练序列
    camera_sequence = generate_random_sequence(current_camera_pool)
    
    # ===============================================================================

    # 初始化进度条，使用当前区域的迭代次数作为总长度
    current_region_total_iters = current_region['iterations']
    progress_bar = tqdm(total=current_region_total_iters, desc=f"Training [Region {current_region_idx+1}: {current_region['name']}]", leave=True)
    progress_bar.set_postfix({"Region": f"{current_region_idx+1}({current_region['name']})", "CamPool": len(current_camera_pool), "LocalIter": f"0/{current_region_total_iters}"})
    
    # 初始化时加载第一个区域的所有照片到GPU
    print(f"\n🔄 [Init] 加载区域 {current_region_idx+1} ({current_region['name']}) 的所有相机到GPU...")
    scene.release_images()
    torch.cuda.empty_cache()
    # 一次性加载当前区域的所有相机到GPU
    scene.load_cameras_by_ids(current_camera_pool)
    print(f"✅ 区域 {current_region_idx+1} 所有相机已加载到GPU")
    # 训练循环
    for iteration in range(first_iter, TOTAL_TRAIN_ITER + 1):        
        # network gui not available in octree-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # ====================== 动态切换训练区域 (核心改造) ======================
        # 检查是否需要切换区域
        for region_idx in range(num_regions):
            start_iter, end_iter = REGION_ITER_BOUNDS[region_idx]
            if iteration == start_iter and region_idx != current_region_idx:
                prev_region_idx = current_region_idx
                prev_region_config = REGIONS_CONFIG[prev_region_idx]

                print({"Region": f"{prev_region_idx+1}→{current_region_idx+1}", "Status": "Switching"})

                # 1. 为当前区域的所有高斯赋值region属性
                gaussians.set_region(prev_region_idx)
                
                # 2. 存储当前区域训练好的锚点
                gaussians.save_region_anchors(prev_region_idx, os.path.join(dataset.model_path, "region_anchors"))
                
                # 3. 退出当前区域
                gaussians.clean_out_region(current_polygon, f"Region_{current_region_idx}")
                gaussians.clean_in_region()
                gaussians.exit_and_cleanup()
                
                # 4. 释放上个区域的照片内存
                print(f"🔄 [Switch] 释放区域 {prev_region_idx+1} 的照片内存...")
                scene.release_images()
                torch.cuda.empty_cache()
                print(f"✅ 区域 {prev_region_idx+1} 照片内存已释放")
        
                # 5. 切换变量
                current_region_idx = region_idx
                current_region = REGIONS_CONFIG[current_region_idx]
                current_polygon = Polygon(current_region['vertices'])
                
                # 6. 【核心】重置当前区域的本地迭代计数器
                REGION_LOCAL_ITERS[current_region_idx] = 0
                
                # 7. 进入新区域
                gaussians.enter_region(current_polygon)
                gaussians.set_region(current_region_idx)
                
                # 8. 加载新区域的所有照片到GPU
                current_camera_pool = REGION_CAMERA_POOLS[current_region_idx]
                camera_sequence = generate_random_sequence(current_camera_pool)
                print(f"🔄 [Switch] 加载区域 {current_region_idx+1} ({current_region['name']}) 的所有相机到GPU...")
                scene.load_cameras_by_ids(current_camera_pool)
                print(f"✅ 区域 {current_region_idx+1} 所有相机已加载到GPU")

                progress_bar.set_description(f"Training [Region {current_region_idx+1}: {current_region['name']}]")
                progress_bar.set_postfix({"Region": f"{current_region_idx+1}({current_region['name']})", "CamPool": len(current_camera_pool), "LocalIter": "0/{current_region['iterations']}", "Status": "Ready"})
                
                progress_bar.set_description(f"Training [Region {current_region_idx+1}: {current_region['name']}]")
                send_mail(f"switching from region {prev_region_idx} to {current_region_idx} at iteration {iteration}")
                break
        # ===============================================================================

        # ====================== 【核心】本地迭代计数器自增 ======================
        REGION_LOCAL_ITERS[current_region_idx] += 1
        current_local_iter = REGION_LOCAL_ITERS[current_region_idx]  # 获取当前区域的本地步数
        current_region_total_iters = current_region['iterations']    # 获取当前区域的总步数
        # ===========================================================================

        iter_start.record()

        gaussians.update_learning_rate(current_local_iter) # 注意：学习率通常还是随全局迭代衰减，这里保持不变

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 随机选择当前区域中的一个相机
        # 直接从当前区域的相机池中随机选择一个相机ID
        if not camera_sequence:
            # 当序列用完时，生成新的随机序列
            camera_sequence = generate_random_sequence(current_camera_pool)
        camera_id = camera_sequence.pop()
        # 根据相机ID获取相机对象
        viewpoint_cam = scene.get_camera_by_id(camera_id)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # 注意：set_anchor_mask 里的 iteration 通常用于控制 LOD  progressive，建议保留全局 iteration
        gaussians.set_anchor_mask(viewpoint_cam.camera_center, current_local_iter, viewpoint_cam.resolution_scale)
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        
        # 【注意】retain_grad 控制是否需要回传梯度来 densify，这里改为基于本地迭代判断
        # retain_grad = (iteration < opt.update_until and iteration >= 0) 
        # 改为：
        retain_grad = (current_local_iter < opt.update_until and current_local_iter >= 0)
        
        # 传递当前区域索引作为camera_region参数
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, camera_region=current_region_idx)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        ssim_loss = (1.0 - ssim(image, gt_image))
        if scaling.shape[0] > 0:
            scaling_reg = scaling.prod(dim=1).mean()
        else:
            scaling_reg = torch.tensor(0.0, device="cuda")
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        loss.backward()
        # 监控MLP梯度
        # monitor_mlp_gradients(gaussians, current_region_idx, iteration)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 1000 == 0:
                # 【修改】进度条增加本地迭代显示
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Region": f"{current_region_idx+1}({current_region['name']})",
                    "LocalIter": f"{current_local_iter}/{current_region_total_iters}"
                })
                progress_bar.update(1000)
            if iteration == TOTAL_TRAIN_ITER:
                # 存储最后一个区域的训练好的锚点
                gaussians.set_region(current_region_idx)
                gaussians.save_region_anchors(current_region_idx, os.path.join(dataset.model_path, "region_anchors"))
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            
            
            # ====================== 【核心】密度优化逻辑：全部替换为本地迭代 ======================
            # if iteration < opt.update_until and iteration > opt.start_stat:
            # 改为：
            if current_local_iter < opt.update_until and current_local_iter > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification: 基于本地迭代
                # if opt.update_anchor and iteration > opt.update_from and iteration % opt.update_interval == 0:
                # 改为：
                if opt.update_anchor and current_local_iter > opt.update_from and current_local_iter % opt.update_interval == 0:
                    gaussians.adjust_anchor(
                        iteration=current_local_iter, # 传入本地迭代，虽然内部可能没用来做判断，但传进去更保险
                        check_interval=opt.update_interval, 
                        success_threshold=opt.success_threshold,
                        grad_threshold=opt.densify_grad_threshold, 
                        update_ratio=dataset.update_ratio,
                        extra_ratio=dataset.extra_ratio,
                        extra_up=dataset.extra_up,
                        min_opacity=opt.min_opacity
                    )
            # elif iteration == opt.update_until:
            # 改为：
            elif current_local_iter == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
            # =====================================================================================
                    
            # Optimizer step
            if iteration < TOTAL_TRAIN_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
            # 更新进度条
            current_region_total_iters = current_region['iterations']
            progress_bar.set_postfix({"Region": f"{current_region_idx+1}({current_region['name']})", "CamPool": len(current_camera_pool), "LocalIter": f"{current_local_iter}/{current_region_total_iters}", "Loss": f"{loss.item():.7f}"})
            # 确保进度条的当前值不超过总迭代次数
            if current_local_iter <= current_region_total_iters:
                progress_bar.n = current_local_iter
                progress_bar.refresh()
            
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            # if (iteration in saving_iterations):
            #     logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
            #     # 为当前区域的所有高斯赋值region属性
            #     gaussians.set_region(current_region_idx)
            #     scene.save(iteration)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                gaussians.set_region(current_region_idx)
                scene.save(iteration)
    send_mail(f"The training process has completed after {TOTAL_TRAIN_ITER} iterations.")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    scene.gaussians.set_anchor_mask(viewpoint.camera_center, iteration, viewpoint.resolution_scale)
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales, logger=logger)
        gaussians.eval()

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            # 直接获取所有训练相机
            train_cameras = scene.getTrainCameras()
            if train_cameras:
                t_train_list, visible_count = render_set(dataset.model_path, "train", scene.loaded_iter, train_cameras, gaussians, pipeline, background)
                if t_train_list:
                    train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
                    logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
                    if wandb is not None:
                        wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            # 渲染测试相机（如果有的话）
            test_cameras = scene.getTestCameras()
            if test_cameras:
                t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, test_cameras, gaussians, pipeline, background)
                test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
                logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
                if wandb is not None:
                    wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 30000000 for i in range(1, 32 + 1)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i * 50000 for i in range(1, 14 + 1)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    send_mail(f"begin training {args.model_path}.")
    # enable logging
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    # if args.test_iterations[0] == -1:
    #     args.test_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    # if len(args.test_iterations) == 0 or args.test_iterations[-1] != args.iterations:
    #     args.test_iterations.append(args.iterations)
    # print(args.test_iterations)

    # if args.save_iterations[0] == -1:
    #     args.save_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    # if len(args.save_iterations) == 0 or args.save_iterations[-1] != args.iterations:
    #     args.save_iterations.append(args.iterations)
    # print(args.save_iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Octree-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # 拼接所有区域的锚点文件
    logger.info("\n🔗 拼接所有区域的锚点文件...")
    merge_region_anchors(args.model_path)

    # 保存MLP为PT文件
    logger.info("\n💾 保存MLP为PT文件...")
    # 这里需要加载训练好的MLP参数并保存为PT文件
    # 例如，可以使用torch.save保存模型的状态字典
    # 具体实现需要根据模型的结构来确定

    # rendering
    logger.info(f'\nStarting Rendering~')
    if args.eval:
        visible_count = render_sets(lp.extract(args), -1, pp.extract(args), skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    else:
        visible_count = render_sets(lp.extract(args), -1, pp.extract(args), skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    eval_name = 'test' if args.eval else 'train'
    evaluate(args.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
