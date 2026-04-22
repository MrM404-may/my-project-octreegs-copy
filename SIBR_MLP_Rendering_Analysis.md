# SIBR 中 MLP 使用与新视角渲染分析

## 1. 项目结构概览

SIBR (Simple IBR) 是一个用于图像基渲染的框架，在我们的项目中主要用于高斯点云的可视化和新视角渲染。核心组件位于 `SIBR_viewers` 目录下：

| 路径 | 用途 |
|------|------|
| `/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp` | 高斯点云渲染的核心实现 |
| `/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.hpp` | 高斯点云渲染的头文件 |
| `/workspace/SIBR_viewers/src/projects/gaussianviewer/apps/gaussianViewer/main.cpp` | 高斯点云查看器应用 |

## 2. MLP 模型的加载与使用

### 2.1 MLP 模型文件路径

在 `GaussianView` 类的构造函数中，MLP 模型从以下路径加载：

```cpp
std::string ply_path = plyPath + "point_cloud.ply";
std::string opacity_mlp_path = plyPath + "opacity_mlp.pt";
std::string cov_mlp_path = plyPath + "cov_mlp.pt";
std::string color_mlp_path = plyPath + "color_mlp.pt";
std::string appearance_path = plyPath + "embedding_appearance.pt";
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

### 2.2 MLP 模型加载

MLP 模型使用 PyTorch 的 C++ API 加载：

```cpp
opacity_mlp_module = torch::jit::load(opacity_mlp_path, _libtorch_device);
color_mlp_module = torch::jit::load(color_mlp_path, _libtorch_device);
cov_mlp_module = torch::jit::load(cov_mlp_path, _libtorch_device);
if (isFileExists_fopen(appearance_path))
{
    appearance_module = torch::jit::load(appearance_path, _libtorch_device);
    SIBR_LOG << "appearance code id : " << _appearance_id << std::endl;
    _add_appearance = true;
}
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

### 2.3 MLP 模型的输入

MLP 模型的输入由以下部分组成：

1. **锚点特征**：从 PLY 文件加载的锚点特征向量
2. **视角方向**：相机位置与锚点位置的归一化向量
3. **距离信息**：相机与锚点的距离（可选）
4. **外观编码**：如果使用外观模型，则添加外观编码

```cpp
torch::Tensor cat_local_view = torch::cat({ ak_feat, ob_view, ob_dist }, 1).to(_libtorch_device);
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

### 2.4 MLP 模型的使用

#### 不透明度 MLP

```cpp
torch::Tensor neural_opacity;
if (_add_opacity_dist)
    neural_opacity = opacity_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
else
    neural_opacity = opacity_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

#### 协方差 MLP

```cpp
torch::Tensor scale_rot;
if(_add_cov_dist)
    scale_rot = cov_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
else
    scale_rot = cov_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

#### 颜色 MLP

```cpp
if (_add_appearance)
{
    torch::Tensor camera_indices = torch::ones({ M }, torch::kLong).to(_libtorch_device);
    camera_indices = camera_indices * _appearance_id;
    torch::Tensor appearance = appearance_module.forward({ camera_indices }).toTensor().to(_libtorch_device);
    if (_add_color_dist)
    {
        cat_local_view = torch::cat({ cat_local_view, appearance }, 1).to(_libtorch_device);
        gs_color = color_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
    }
    else
    {
        cat_local_view_wodist = torch::cat({ cat_local_view_wodist, appearance }, 1).to(_libtorch_device);
        gs_color = color_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
    }
}
else
{
    if (_add_color_dist)
    {
        gs_color = color_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
    }
    else
    {
        gs_color = color_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
    }
}
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

## 3. 新视角渲染流程

### 3.1 渲染入口函数

新视角渲染的入口函数是 `onRenderIBR`：

```cpp
void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget& dst, sibr::Camera& eye)
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

### 3.2 渲染流程步骤

1. **相机参数转换**：将相机的视图矩阵和投影矩阵转换为目标坐标系
2. **LOD 计算**：根据相机位置计算每个锚点的LOD级别
3. **可见性过滤**：过滤掉不可见的锚点
4. **MLP 推理**：使用MLP模型计算高斯点的属性
5. **光栅化**：使用CUDA光栅化器渲染高斯点
6. **结果输出**：将渲染结果复制到帧缓冲区

### 3.3 LOD 计算

```cpp
torch::Tensor dist = torch::sqrt(torch::sum(torch::pow(ak_pos_all - eye_pos_tensor, 2), 1)).view({ -1, 1 }).to(_libtorch_device);
torch::Tensor pred_level = ((torch::log2(standard_dist / dist) / log2(_fork)) + 1.0).to(_libtorch_device);
pred_level = pred_level + ak_extra_level_all.to(_libtorch_device);
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

### 3.4 可见性过滤

```cpp
CudaRasterizer::Rasterizer::visible_filter(
    geomBufferFunc_filter,
    binningBufferFunc_filter,
    imgBufferFunc_filter,
    S, M,
    W, H,
    ak_pos_all.index({ ak_mask }).contiguous().data<float>(),
    ak_scale_1.index({ ak_mask }).contiguous().data_ptr<float>(),
    scale_modifier,
    ak_rot_all.index({ ak_mask }).contiguous().data_ptr<float>(),
    nullptr,
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    tan_fovx,
    tan_fovy,
    FALSE,
    radii.contiguous().data<int>()
);
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

### 3.5 光栅化

```cpp
CudaRasterizer::Rasterizer::forward(
    geomBufferFunc,
    binningBufferFunc,
    imgBufferFunc,
    P, 1, 16,
    background_cuda,
    _resolution.x(), _resolution.y(),
    gs_pos.contiguous().data<float>(),
    nullptr,
    gs_color.contiguous().data<float>(),
    gs_opacity.contiguous().data<float>(),
    gs_scale.contiguous().data<float>(),
    _scalingModifier,
    gs_rot.contiguous().data<float>(),
    nullptr,
    view_cuda,
    proj_cuda,
    cam_pos_cuda,
    tan_fovx,
    tan_fovy,
    false,
    image_cuda,
    nullptr,
    rects,
    boxmin,
    boxmax
);
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

## 4. 核心数据结构

### 4.1 锚点结构

```cpp
struct AnchorPoint
{
    float pos[3];
    float level;
    float extra_level;
    float info;
    float offset[30];
    float feat[32];
    float opacity;
    float scale[6];
    float rot[4];
};
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

### 4.2 PLY 文件加载

PLY 文件加载函数 `loadPly` 负责从 PLY 文件中读取锚点数据：

```cpp
int loadPly(const char* filename,
    std::vector<float>& pos,
    std::vector<int>& level,
    std::vector<float>& extra_level, 
    std::vector<float>& offset,
    std::vector<float>& feat,
    std::vector<float>& opacity,
    std::vector<float>& scale1,
    std::vector<float>& scale2,
    std::vector<float>& rot,
    std::vector<float>& gs_pos,
    float& voxel_size,
    float& standard_dist,
    int& levels,
    sibr::Vector3f& minn,
    sibr::Vector3f& maxx)
```
<mcfile name="GaussianView.cpp" path="/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp"></mcfile>

## 5. 渲染模式

SIBR 支持多种渲染模式，可通过 GUI 选择：

| 渲染模式 | 描述 |
|---------|------|
| Splats | 标准高斯点渲染 |
| Initial Points | 初始点渲染 |
| Gaussian Points | 高斯点渲染 |
| Depth | 深度图渲染 |
| Normal | 法线渲染 |
| LOD Levels | LOD级别可视化 |
| LOD Bias | LOD偏置可视化 |

## 6. 性能优化

1. **CUDA/OpenGL 互操作**：尝试使用 CUDA/OpenGL 互操作以减少 CPU-GPU 数据传输
2. **快速剔除**：使用快速剔除算法减少需要处理的点数量
3. **分层次渲染**：根据LOD级别选择合适的锚点进行渲染
4. **内存管理**：使用动态内存分配和释放策略

## 7. 代码优化建议

1. **内存使用优化**：
   - 减少不必要的内存分配和释放
   - 使用内存池管理频繁分配的内存

2. **计算优化**：
   - 并行计算 LOD 级别
   - 优化 MLP 推理，考虑使用 TensorRT 等加速库

3. **渲染优化**：
   - 实现更高效的可见性剔除算法
   - 考虑使用空间数据结构（如八叉树）加速空间查询

4. **代码结构优化**：
   - 模块化 MLP 处理逻辑
   - 分离渲染逻辑和数据管理

## 8. 输入输出示例

#### 输入输出示例

**输入**：
- PLY 文件路径：`./output/1234567890/point_cloud/iteration_10000/`
- 相机参数：位置 (0, 0, 5)，朝向 (0, 0, -1)，FOV 60度
- 渲染分辨率：1024x768

**输出**：
- 渲染图像：1024x768 RGB 图像
- 渲染信息：
  - 锚点数量：100,000
  - 高斯点数量：800,000
  - 渲染时间：15ms

## 9. 总结

SIBR 中的高斯点云渲染系统通过以下步骤实现新视角渲染：

1. **数据加载**：从 PLY 文件加载锚点数据和从 PT 文件加载 MLP 模型
2. **LOD 计算**：根据相机位置计算每个锚点的 LOD 级别
3. **可见性过滤**：过滤掉不可见的锚点
4. **MLP 推理**：使用 MLP 模型计算高斯点的属性（颜色、不透明度、协方差）
5. **光栅化**：使用 CUDA 光栅化器渲染高斯点
6. **结果输出**：将渲染结果复制到帧缓冲区

这种方法利用了 MLP 的强大表达能力和 CUDA 的并行计算能力，实现了高质量、实时的新视角渲染。

## 10. 关键函数路径索引

| 函数名 | 路径 | 功能 |
|-------|------|------|
| `loadPly` | `/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp` | 加载 PLY 文件数据 |
| `GaussianView::GaussianView` | `/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp` | 构造函数，加载 MLP 模型 |
| `GaussianView::onRenderIBR` | `/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp` | 渲染新视角 |
| `CudaRasterizer::Rasterizer::forward` | 外部库 | 高斯点光栅化 |
| `BufferCopyRenderer::process` | `/workspace/SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp` | 复制渲染结果到帧缓冲区 |