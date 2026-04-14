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
def generate_2d_cost_map(depth_maps, cameras, grid_size=(100, 100), z_height=1.0):
    # 确定xy范围
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    
    for i, (depth_map, camera) in enumerate(zip(depth_maps, cameras)):
        # 遍历图像边缘像素，计算世界坐标
        height, width = depth_map.shape
        pixels = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
        
        for (x, y) in pixels:
            depth = depth_map[y, x]
            if depth > 0:  # 有效深度
                # 像素转相机坐标
                fx = camera['fx']
                fy = camera['fy']
                cx = width / 2  # 假设光心在图像中心
                cy = height / 2
                camera_point = pixel_to_camera(x, y, depth, fx, fy, cx, cy)
                
                # 相机转世界坐标
                world_point = camera_to_world(camera_point, camera['position'], camera['rotation'])
                min_x = min(min_x, world_point[0])
                max_x = max(max_x, world_point[0])
                min_y = min(min_y, world_point[1])
                max_y = max(max_y, world_point[1])
    
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
        
        # 采样像素点
        step = max(1, int(min(width, height) / 50))  # 每隔step个像素采样一次
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                if depth > 0:
                    # 像素转相机坐标
                    camera_point = pixel_to_camera(x, y, depth, fx, fy, cx, cy)
                    # 相机转世界坐标
                    world_point = camera_to_world(camera_point, camera['position'], camera['rotation'])
                    
                    # 检查是否在z_height附近
                    if abs(world_point[2] - z_height) < 0.5:
                        # 计算在代价地图中的位置
                        map_x = int((world_point[0] - min_x) / x_step)
                        map_y = int((world_point[1] - min_y) / y_step)
                        
                        if 0 <= map_x < grid_size[0] and 0 <= map_y < grid_size[1]:
                            # 增加代价（距离越近代价越低）
                            distance = abs(world_point[2] - z_height)
                            cost_map[map_y, map_x] += 1.0 / (1.0 + distance)
    
    # 归一化代价地图
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    
    return cost_map, (min_x, max_x, min_y, max_y)

# 生成三维代价地图
def generate_3d_cost_map(depth_maps, cameras, grid_size=(50, 50, 50)):
    # 确定三维范围
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    min_z, max_z = float('inf'), -float('inf')
    
    for depth_map, camera in zip(depth_maps, cameras):
        height, width = depth_map.shape
        # 采样像素点
        step = max(1, int(min(width, height) / 30))
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                if depth > 0:
                    fx = camera['fx']
                    fy = camera['fy']
                    cx = width / 2
                    cy = height / 2
                    camera_point = pixel_to_camera(x, y, depth, fx, fy, cx, cy)
                    world_point = camera_to_world(camera_point, camera['position'], camera['rotation'])
                    
                    min_x = min(min_x, world_point[0])
                    max_x = max(max_x, world_point[0])
                    min_y = min(min_y, world_point[1])
                    max_y = max(max_y, world_point[1])
                    min_z = min(min_z, world_point[2])
                    max_z = max(max_z, world_point[2])
    
    # 扩展范围
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    min_x -= x_range * 0.1
    max_x += x_range * 0.1
    min_y -= y_range * 0.1
    max_y += y_range * 0.1
    min_z -= z_range * 0.1
    max_z += z_range * 0.1
    
    # 创建三维代价地图
    cost_map = np.zeros(grid_size)
    x_step = (max_x - min_x) / grid_size[0]
    y_step = (max_y - min_y) / grid_size[1]
    z_step = (max_z - min_z) / grid_size[2]
    
    # 对每个相机的深度图进行处理
    for depth_map, camera in zip(depth_maps, cameras):
        height, width = depth_map.shape
        fx = camera['fx']
        fy = camera['fy']
        cx = width / 2
        cy = height / 2
        
        step = max(1, int(min(width, height) / 50))
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                if depth > 0:
                    camera_point = pixel_to_camera(x, y, depth, fx, fy, cx, cy)
                    world_point = camera_to_world(camera_point, camera['position'], camera['rotation'])
                    
                    # 计算在代价地图中的位置
                    map_x = int((world_point[0] - min_x) / x_step)
                    map_y = int((world_point[1] - min_y) / y_step)
                    map_z = int((world_point[2] - min_z) / z_step)
                    
                    if (0 <= map_x < grid_size[0] and 
                        0 <= map_y < grid_size[1] and 
                        0 <= map_z < grid_size[2]):
                        # 增加代价
                        cost_map[map_z, map_y, map_x] += 1.0
    
    # 归一化
    if np.max(cost_map) > 0:
        cost_map = cost_map / np.max(cost_map)
    
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
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                if cost_map[i, j, k] > 0.1:  # 只显示代价大于0.1的点
                    points.append([x[k], y[j], z[i]])
                    values.append(cost_map[i, j, k])
    
    points = np.array(points)
    values = np.array(values)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=values, cmap='jet', alpha=0.5, s=10)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Cost Map')
    plt.colorbar(scatter, label='Cost')
    
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
    
    # 读取逆深度图
    print("Loading inverse depth maps...")
    inv_depth_files = sorted([f for f in os.listdir(inv_depth_dir) if f.endswith('.npy')])
    depth_maps = []
    
    for file in inv_depth_files:
        inv_depth_path = os.path.join(inv_depth_dir, file)
        inv_depth_map = load_inv_depth_map(inv_depth_path)
        depth_map = inv_depth_to_depth(inv_depth_map)
        depth_maps.append(depth_map)
        print(f"Loaded {file}, shape: {depth_map.shape}")
    
    # 确保深度图数量与相机数量匹配
    if len(depth_maps) != len(cameras):
        print(f"Warning: Number of depth maps ({len(depth_maps)}) doesn't match number of cameras ({len(cameras)})")
        # 只使用前N个相机，其中N是深度图数量
        cameras = cameras[:len(depth_maps)]
    
    # 生成二维代价地图
    print("Generating 2D cost map...")
    cost_map_2d, bounds_2d = generate_2d_cost_map(depth_maps, cameras)
    visualize_2d_cost_map(cost_map_2d, bounds_2d, 
                         save_path=os.path.join(output_dir, '2d_cost_map.png'))
    
    # 保存二维代价地图数据
    np.save(os.path.join(output_dir, '2d_cost_map.npy'), cost_map_2d)
    with open(os.path.join(output_dir, '2d_bounds.txt'), 'w') as f:
        f.write(f"min_x: {bounds_2d[0]}, max_x: {bounds_2d[1]}\n")
        f.write(f"min_y: {bounds_2d[2]}, max_y: {bounds_2d[3]}")
    
    # 生成三维代价地图
    print("Generating 3D cost map...")
    cost_map_3d, bounds_3d = generate_3d_cost_map(depth_maps, cameras)
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
