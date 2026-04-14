import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取相机参数
def load_camera_params(camera_json_path):
    with open(camera_json_path, 'r') as f:
        cameras = json.load(f)
    return cameras

# 读取逆深度图
def load_inv_depth_map(inv_depth_path):
    return np.load(inv_depth_path)

# 从逆深度计算深度
def inv_depth_to_depth(inv_depth_map):
    # 避免除以零
    depth_map = 1.0 / (inv_depth_map + 1e-8)
    
    # 第一步：限制深度范围，避免过大的值
    depth_map = np.clip(depth_map, 0, 30.0)
    
    # 第二步：野点过滤：使用中值滤波
    from scipy.ndimage import median_filter
    depth_map = median_filter(depth_map, size=7)  # 增大滤波窗口
    
    # 第三步：再次限制范围，确保滤波后没有异常值
    depth_map = np.clip(depth_map, 0, 25.0)
    
    # 第四步：计算深度图的统计信息，进一步过滤异常值
    mean_depth = np.mean(depth_map[depth_map > 0])
    std_depth = np.std(depth_map[depth_map > 0])
    if std_depth > 0:
        # 过滤掉远离均值的异常值
        depth_map = np.where(np.abs(depth_map - mean_depth) > 3 * std_depth, 0, depth_map)
    
    return depth_map

# 像素坐标转相机坐标
def pixel_to_camera(pixel_x, pixel_y, depth, fx, fy, cx, cy):
    # 相机内参：fx, fy是焦距，cx, cy是光心
    camera_x = (pixel_x - cx) * depth / fx
    camera_y = (pixel_y - cy) * depth / fy
    camera_z = depth
    return np.array([camera_x, camera_y, camera_z])

# 相机坐标转世界坐标
def camera_to_world(camera_point, camera_position, camera_rotation):
    # 相机旋转矩阵是相机坐标系到世界坐标系的变换
    # 注意：这里假设rotation是行优先的
    rotation_matrix = np.array(camera_rotation)
    world_point = rotation_matrix @ camera_point + np.array(camera_position)
    return world_point

# 生成二维代价地图（xy平面）
def generate_2d_cost_map(depth_maps, cameras, grid_size=(200, 200), z_height=1.0):
    # 确定xy范围
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    
    # 首先计算所有相机的位置范围
    for camera in cameras:
        pos = camera['position']
        min_x = min(min_x, pos[0] - 4)
        max_x = max(max_x, pos[0] + 4)
        min_y = min(min_y, pos[1] - 4)
        max_y = max(max_y, pos[1] + 4)
    
    # 扩展范围以确保覆盖所有点
    x_range = max_x - min_x
    y_range = max_y - min_y
    min_x -= x_range * 0.1
    max_x += x_range * 0.1
    min_y -= y_range * 0.1
    max_y += y_range * 0.1
    
    # 创建代价地图
    cost_map = np.zeros(grid_size)
    x_step = (max_x - min_x) / grid_size[0]
    y_step = (max_y - min_y) / grid_size[1]
    
    # 对每个相机的深度图进行处理
    for depth_map, camera in zip(depth_maps, cameras):
        height, width = depth_map.shape
        fx = camera['fx']
        fy = camera['fy']
        cx = width / 2
        cy = height / 2
        
        # 采样像素点 - 增加采样密度
        step = max(1, int(min(width, height) / 40))
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                if depth > 0.5 and depth < 15.0:  # 更严格的深度限制
                    # 像素转相机坐标
                    camera_point = pixel_to_camera(x, y, depth, fx, fy, cx, cy)
                    # 相机转世界坐标
                    world_point = camera_to_world(camera_point, camera['position'], camera['rotation'])
                    
                    # 检查是否在z_height附近
                    if abs(world_point[2] - z_height) < 0.2:  # 更严格的高度范围
                        # 计算在代价地图中的位置
                        map_x = int((world_point[0] - min_x) / x_step)
                        map_y = int((world_point[1] - min_y) / y_step)
                        
                        if 0 <= map_x < grid_size[0] and 0 <= map_y < grid_size[1]:
                            # 增加代价（距离越近代价越高，同时考虑深度因素）
                            height_distance = abs(world_point[2] - z_height)
                            depth_factor = 1.0 / (1.0 + depth * 0.1)  # 近处的点权重更高
                            cost_map[map_y, map_x] += depth_factor / (1.0 + height_distance * 5)  # 更严格的高度惩罚
    
    # 归一化代价地图
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    
    # 平滑代价地图，减少野点
    from scipy.ndimage import gaussian_filter
    cost_map = gaussian_filter(cost_map, sigma=2.5)  # 增大平滑程度
    
    # 二次归一化
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    
    # 应用阈值，进一步减少野点
    cost_map[cost_map < 0.15] = 0
    
    # 再次平滑，确保结果更清晰
    cost_map = gaussian_filter(cost_map, sigma=1.5)
    
    # 最终归一化
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    
    return cost_map, (min_x, max_x, min_y, max_y)

# 生成三维代价地图
def generate_3d_cost_map(depth_maps, cameras, grid_size=(40, 40, 30)):
    # 首先收集所有世界点，计算实际范围
    all_world_points = []
    
    for depth_map, camera in zip(depth_maps, cameras):
        height, width = depth_map.shape
        fx = camera['fx']
        fy = camera['fy']
        cx = width / 2
        cy = height / 2
        
        # 采样像素点
        step = max(1, int(min(width, height) / 50))
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                if depth > 0.5 and depth < 15.0:
                    camera_point = pixel_to_camera(x, y, depth, fx, fy, cx, cy)
                    world_point = camera_to_world(camera_point, camera['position'], camera['rotation'])
                    all_world_points.append(world_point)
    
    # 计算世界点的范围
    if all_world_points:
        all_world_points = np.array(all_world_points)
        min_x = np.min(all_world_points[:, 0])
        max_x = np.max(all_world_points[:, 0])
        min_y = np.min(all_world_points[:, 1])
        max_y = np.max(all_world_points[:, 1])
        min_z = np.min(all_world_points[:, 2])
        max_z = np.max(all_world_points[:, 2])
        
        # 扩展范围
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z
        
        # 确保范围不为零
        if x_range == 0:
            x_range = 1.0
            min_x -= 0.5
            max_x += 0.5
        if y_range == 0:
            y_range = 1.0
            min_y -= 0.5
            max_y += 0.5
        if z_range == 0:
            z_range = 1.0
            min_z -= 0.5
            max_z += 0.5
        
        # 扩展范围以确保覆盖所有点
        min_x -= x_range * 0.1
        max_x += x_range * 0.1
        min_y -= y_range * 0.1
        max_y += y_range * 0.1
        min_z -= z_range * 0.1
        max_z += z_range * 0.1
        
        print(f"World point range - X: [{min_x:.2f}, {max_x:.2f}], Y: [{min_y:.2f}, {max_y:.2f}], Z: [{min_z:.2f}, {max_z:.2f}]")
    else:
        # 如果没有世界点，使用相机位置范围
        min_x, max_x = float('inf'), -float('inf')
        min_y, max_y = float('inf'), -float('inf')
        min_z, max_z = float('inf'), -float('inf')
        
        for camera in cameras:
            pos = camera['position']
            min_x = min(min_x, pos[0] - 5)
            max_x = max(max_x, pos[0] + 5)
            min_y = min(min_y, pos[1] - 5)
            max_y = max(max_y, pos[1] + 5)
            min_z = min(min_z, pos[2] - 5)
            max_z = max(max_z, pos[2] + 5)
    
    # 创建三维代价地图
    cost_map = np.zeros(grid_size)
    x_step = (max_x - min_x) / grid_size[2]  # x对应grid_size[2]
    y_step = (max_y - min_y) / grid_size[1]  # y对应grid_size[1]
    z_step = (max_z - min_z) / grid_size[0]  # z对应grid_size[0]
    
    # 对每个相机的深度图进行处理
    for camera_idx, (depth_map, camera) in enumerate(zip(depth_maps, cameras)):
        height, width = depth_map.shape
        fx = camera['fx']
        fy = camera['fy']
        cx = width / 2
        cy = height / 2
        
        # 增加采样密度
        step = max(1, int(min(width, height) / 50))
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                if depth > 0.5 and depth < 15.0:  # 更严格的深度限制
                    camera_point = pixel_to_camera(x, y, depth, fx, fy, cx, cy)
                    world_point = camera_to_world(camera_point, camera['position'], camera['rotation'])
                    
                    # 计算在代价地图中的位置
                    map_x = int((world_point[0] - min_x) / x_step)
                    map_y = int((world_point[1] - min_y) / y_step)
                    map_z = int((world_point[2] - min_z) / z_step)
                    
                    # 边界检查
                    if (0 <= map_x < grid_size[0] and 
                        0 <= map_y < grid_size[1] and 
                        0 <= map_z < grid_size[2]):
                        # 增加代价，考虑深度因素
                        depth_factor = 1.0 / (1.0 + depth * 0.1)  # 近处的点权重更高
                        cost_map[map_z, map_y, map_x] += depth_factor
    
    # 归一化
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    
    # 平滑代价地图，减少野点
    from scipy.ndimage import gaussian_filter
    cost_map = gaussian_filter(cost_map, sigma=1.8)  # 增大平滑程度
    
    # 二次归一化
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    
    # 应用阈值，进一步减少野点
    cost_map[cost_map < 0.02] = 0
    
    # 再次平滑，确保结果更清晰
    cost_map = gaussian_filter(cost_map, sigma=1.0)
    
    # 最终归一化
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    else:
        print("3D cost map is all zero!")
    
    return cost_map, (min_x, max_x, min_y, max_y, min_z, max_z)

# 可视化二维代价地图
def visualize_2d_cost_map(cost_map, bounds, save_path=None):
    min_x, max_x, min_y, max_y = bounds
    plt.figure(figsize=(10, 8))
    plt.imshow(cost_map, cmap='jet', origin='lower', 
               extent=[min_x, max_x, min_y, max_y])
    plt.colorbar(label='Cost')
    plt.title('2D Cost Map (XY Plane)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

# 可视化三维代价地图
def visualize_3d_cost_map(cost_map, bounds, save_path=None):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    grid_size = cost_map.shape
    
    # 生成网格点
    x = np.linspace(min_x, max_x, grid_size[2])
    y = np.linspace(min_y, max_y, grid_size[1])
    z = np.linspace(min_z, max_z, grid_size[0])
    
    # 创建点云数据
    points = []
    values = []
    
    # 降低阈值，显示更多点
    threshold = 0.02
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                if cost_map[i, j, k] > threshold:  # 进一步降低阈值
                    points.append([x[k], y[j], z[i]])
                    values.append(cost_map[i, j, k])
    
    points = np.array(points)
    values = np.array(values)
    
    print(f"3D cost map points: {len(points)}")  # 打印点的数量
    
    if len(points) == 0:
        print("No points to display! Try lowering the threshold.")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用较大的点大小
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=values, cmap='jet', alpha=0.6, s=20)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Cost Map')
    plt.colorbar(scatter, label='Cost')
    
    # 设置坐标轴范围
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

# 主函数
def main():
    # 路径设置
    camera_json_path = '/workspace/inv_depth/camera.json'
    inv_depth_dir = '/workspace/inv_depth'
    output_dir = '/workspace/output'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取相机参数
    print("Loading camera parameters...")
    cameras = load_camera_params(camera_json_path)
    print(f"Loaded {len(cameras)} cameras")
    
    # 读取逆深度图
    print("Loading inverse depth maps...")
    inv_depth_files = sorted([f for f in os.listdir(inv_depth_dir) if f.endswith('.npy')])
    print(f"Found {len(inv_depth_files)} depth maps")
    
    depth_maps = []
    valid_cameras = []
    
    # 只处理前N个深度图和对应的相机
    max_depth_maps = min(len(inv_depth_files), len(cameras))
    print(f"Processing {max_depth_maps} depth maps and cameras")
    
    for i in range(max_depth_maps):
        file = inv_depth_files[i]
        inv_depth_path = os.path.join(inv_depth_dir, file)
        inv_depth_map = load_inv_depth_map(inv_depth_path)
        depth_map = inv_depth_to_depth(inv_depth_map)
        depth_maps.append(depth_map)
        valid_cameras.append(cameras[i])
        print(f"Loaded {file}, shape: {depth_map.shape}")
    
    # 生成二维代价地图
    print("Generating 2D cost map...")
    cost_map_2d, bounds_2d = generate_2d_cost_map(depth_maps, valid_cameras)
    visualize_2d_cost_map(cost_map_2d, bounds_2d, 
                         save_path=os.path.join(output_dir, '2d_cost_map.png'))
    
    # 保存二维代价地图数据
    np.save(os.path.join(output_dir, '2d_cost_map.npy'), cost_map_2d)
    with open(os.path.join(output_dir, '2d_bounds.txt'), 'w') as f:
        f.write(f"min_x: {bounds_2d[0]}, max_x: {bounds_2d[1]}\n")
        f.write(f"min_y: {bounds_2d[2]}, max_y: {bounds_2d[3]}")
    
    # 生成三维代价地图
    print("Generating 3D cost map...")
    # 使用更新后的网格大小
    cost_map_3d, bounds_3d = generate_3d_cost_map(depth_maps, valid_cameras)
    visualize_3d_cost_map(cost_map_3d, bounds_3d, 
                         save_path=os.path.join(output_dir, '3d_cost_map.png'))
    
    # 保存三维代价地图数据
    np.save(os.path.join(output_dir, '3d_cost_map.npy'), cost_map_3d)
    with open(os.path.join(output_dir, '3d_bounds.txt'), 'w') as f:
        f.write(f"min_x: {bounds_3d[0]}, max_x: {bounds_3d[1]}\n")
        f.write(f"min_y: {bounds_3d[2]}, max_y: {bounds_3d[3]}\n")
        f.write(f"min_z: {bounds_3d[4]}, max_z: {bounds_3d[5]}")
    
    print("Processing completed!")
    print(f"Output files saved to {output_dir}")

if __name__ == "__main__":
    main()
