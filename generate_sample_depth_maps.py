import numpy as np
import os

# 创建输出目录
output_dir = '/workspace/inv_depth'
os.makedirs(output_dir, exist_ok=True)

# 生成5个示例逆深度图
for i in range(5):
    # 创建随机逆深度图 (720x1280)
    # 逆深度值通常在0到1之间，值越大表示距离越近
    inv_depth = np.random.rand(720, 1280) * 0.1
    
    # 保存为npy文件
    output_path = os.path.join(output_dir, f'{i}.inv_depth.npy')
    np.save(output_path, inv_depth)
    print(f'Created {output_path}')

print('Sample inverse depth maps generated successfully!')
