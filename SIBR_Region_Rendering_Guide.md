# SIBR 区域渲染功能实现指南

## 1. 功能概述

本文档介绍了如何在 SIBR 查看器中实现区域渲染功能，包括：

- 区域MLP模型的加载和管理
- 相机区域信息JSON解析
- 新视角区域判断（基于位置和方向相似性）
- 区域渲染GUI控制
- 保持与原版MLP接口的完全兼容

## 2. 核心文件修改

### 2.1 GaussianView.hpp

**修改内容：**
- 添加 `RegionCameraInfo` 结构体，用于存储区域相机信息
- 添加 `RegionMLP` 结构体，用于管理区域MLP模型
- 添加区域相关的成员变量
- 扩展构造函数，新增区域MLP参数
- 声明区域渲染相关的函数

**关键新增结构体：**

```cpp
struct RegionCameraInfo {
    int camera_id;
    sibr::Vector3f position;
    sibr::Matrix3f rotation;
    std::string img_name;
    std::map<std::string, bool> in_regions;
};

struct RegionMLP {
    torch::jit::script::Module opacity_mlp_module;
    torch::jit::script::Module color_mlp_module;
    torch::jit::script::Module cov_mlp_module;
    torch::jit::script::Module appearance_module;
    bool has_appearance = false;
};
```

### 2.2 GaussianView.cpp

**修改内容：**
- 修改 `AnchorPoint` 结构体，添加区域字段
- 更新 `loadPly` 函数，支持加载区域信息
- 实现构造函数中的区域MLP初始化分支
- 实现区域MLP加载函数 `loadRegionMLPs`
- 实现区域相机JSON解析函数 `loadRegionCameraInfo`
- 实现新视角区域判断函数 `determineRegionForViewpoint`
- 实现激活区域MLP函数 `activateRegionMLP`
- 修改渲染流程，支持区域MLP使用
- 添加GUI界面选项，包括自动/手动区域切换、区域选择等

**关键新增函数：**

```cpp
// 加载区域MLP模型
void loadRegionMLPs(const std::string& basePath, int numRegions);

// 加载区域相机信息JSON
void loadRegionCameraInfo(const std::string& jsonPath);

// 判断新视角所属的区域
int determineRegionForViewpoint(const sibr::Camera& eye);

// 激活指定区域的MLP
void activateRegionMLP(int regionId);
```

### 2.3 Config.hpp

**修改内容：**
- 添加 `appearance_id` 命令行参数
- 添加 `useRegionMLP` 命令行参数
- 添加 `regionCameraJsonPath` 命令行参数
- 添加 `numRegions` 命令行参数

### 2.4 main.cpp (gaussianViewer)

**修改内容：**
- 更新 GaussianView 构造函数调用，传递新的区域参数

## 3. 区域MLP文件结构

您需要为每个区域准备以下文件：

```
<model_dir>/point_cloud/iteration_XXXX/
├── opacity_mlp_0.pt
├── color_mlp_0.pt
├── cov_mlp_0.pt
├── embedding_appearance_0.pt
├── opacity_mlp_1.pt
├── color_mlp_1.pt
├── cov_mlp_1.pt
├── embedding_appearance_1.pt
...
└── point_cloud.ply (包含anchor的区域信息)
```

## 4. 区域相机信息JSON格式

JSON文件应包含每个训练相机的信息，包括：
- 相机ID
- 位置
- 旋转矩阵
- 图像名称
- 区域归属信息

**示例：**

```json
{
  "0": {
    "id": 0,
    "position": [
      -0.055249775341301224,
      -0.1455984459477679,
      1.7622681171025578
    ],
    "rotation": [
      [
        0.9998546616603204,
        -0.005087245087889541,
        -0.016271923473946532
      ],
      [
        0.005917917532707269,
        0.9986598106666802,
        0.05141556973588323
      ],
      [
        0.015988552411094144,
        -0.051504392983561506,
        0.998544773004792
      ]
    ],
    "img_name": "000011.png",
    "in_regions": {
      "Region1": false,
      "Region2": false,
      "Region3": false,
      "Region4": false,
      "Region5": false,
      "Region6": false,
      "Region7": false,
      "Region8": false,
      "Region9": false,
      "Region10": false,
      "Region11": true
    },
    "mask_ratios": {
      "Region1": 0.0,
      "Region2": 0.0,
      "Region3": 0.0,
      "Region4": 0.0,
      "Region5": 0.0,
      "Region6": 0.0,
      "Region7": 0.0,
      "Region8": 0.0,
      "Region9": 0.0,
      "Region10": 0.0,
      "Region11": 0.0
    }
  },
  ...
}
```

## 5. 使用方法

### 5.1 编译项目

按照 SIBR 原始文档的编译步骤，需要：

1. 安装必要的依赖：
   - OpenGL 开发库
   - GLEW
   - ASSIMP
   - FFMPEG
   - embree3
   - eigen3
   - Boost
   - OpenMP
   - OpenCV
   - GLFW
   - libtorch (1.10+)

2. 配置 CMake：
   - 设置 libtorch 路径
   - 设置 Python 路径

3. 编译项目

### 5.2 运行查看器

使用以下命令运行带有区域渲染功能的查看器：

```bash
SIBR_gaussianViewer_app.exe \
  -s <path_to_data> \
  -m <path_to_ckpt> \
  --use_region_mlp true \
  --region_camera_json <path_to_camera_region_json> \
  --num_regions <number_of_regions>
```

### 5.3 GUI 控制

运行后，可以通过 GUI 界面控制区域渲染功能：

- **Current Region**：显示当前使用的区域
- **Auto Determine Region**：启用/禁用自动区域判断
- **Select Region**：手动选择区域（当自动判断禁用时）

## 6. 技术实现细节

### 6.1 新视角区域判断算法

区域判断基于以下步骤：

1. 计算新视角与每个训练相机的位置距离
2. 计算新视角与每个训练相机的方向相似性（点积）
3. 组合得分：`score = 方向相似性 / (位置距离 + 0.01)`
4. 将得分累加到相机所属的区域
5. 选择得分最高的区域

### 6.2 区域MLP管理

- 所有区域的MLP模型在初始化时加载到内存
- 根据当前视角选择对应的区域MLP
- 保持与原版MLP接口的完全兼容

### 6.3 向后兼容性

- 通过 `use_region_mlp` 参数控制是否使用区域渲染
- 当 `use_region_mlp` 为 false 时，使用原版的全局MLP
- 所有命令行参数都有默认值，确保向后兼容

## 7. 注意事项

1. **性能考虑**：区域MLP模型会增加内存使用，因为需要加载多个MLP模型
2. **区域数量**：建议根据实际情况设置合理的区域数量，避免过多区域导致内存不足
3. **JSON文件格式**：确保区域相机信息JSON文件格式正确，特别是区域名称和相机归属信息
4. **PLY文件**：确保PLY文件包含anchor的区域信息

## 8. 故障排除

### 8.1 Git 依赖问题

如果遇到 Git 依赖问题，可以修改 `cmake/windows/git_describe.cmake` 文件，让它在找不到 Git 时使用默认值：

```cmake
find_package(Git)
if(Git_FOUND)
  message(STATUS "Git found: ${GIT_EXECUTABLE}")
else()
  message(STATUS "Git not found. Using default values.")
  # Set default values when Git is not found
  if(GIT_DESCRIBE_GIT_BRANCH)
    set(${GIT_DESCRIBE_GIT_BRANCH} "unknown")
  endif()
  if(GIT_DESCRIBE_GIT_COMMIT_HASH)
    set(${GIT_DESCRIBE_GIT_COMMIT_HASH} "unknown")
  endif()
  if(GIT_DESCRIBE_GIT_TAG)
    set(${GIT_DESCRIBE_GIT_TAG} "unknown")
  endif()
  if(GIT_DESCRIBE_GIT_VERSION)
    set(${GIT_DESCRIBE_GIT_VERSION} "unknown")
  endif()
  return()
endif()
```

### 8.2 依赖安装问题

如果遇到依赖安装问题，可以参考 SIBR 原始文档中的依赖配置步骤，或者使用预编译的二进制文件。

## 9. 总结

本实现提供了一种在 SIBR 查看器中实现区域渲染的方法，通过加载和管理区域MLP模型，根据新视角与训练相机的相似性自动选择合适的区域MLP，从而实现更好的渲染效果。同时，保持了与原版MLP接口的完全兼容，确保了向后兼容性。