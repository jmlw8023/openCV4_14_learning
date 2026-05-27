# OpenCV 学习指南

> **版本**: OpenCV 4.14.0-pre
> **仓库**: https://github.com/jmlw8023/openCV4_14_learning
> **许可**: Apache 2.0

---

## 版本信息

| 项目 | 值 |
|------|-------|
| OpenCV 版本 | 4.14.0-pre |
| 主版本 | 4 |
| 次版本 | 14 |
| 补丁版本 | 0 |
| 状态 | pre |
| 源码 | [opencv/opencv](https://github.com/opencv/opencv) |

---

## 目录

- [学习路线图](./README.md#学习路线图)
- [模块指南](./modules/)
- [贡献指南](./CONTRIBUTING.md)

---

## 学习路线图

### Stage 1: 入门

| 模块 | 描述 | 状态 |
|--------|-------------|--------|
| [core](./modules/core/README.md) | 核心功能：Mat、数组操作、XML/YAML、并行计算 | ✅ 完成 |
| [imgcodecs](./modules/imgcodecs/README.md) | 图像编解码：imread/imwrite、图像格式解码 | ✅ 完成 |
| [highgui](./modules/highgui/README.md) | 高级GUI：namedWindow、imshow、滑动条 | ✅ 完成 |

### Stage 2: 基础

| 模块 | 描述 | 状态 |
|--------|-------------|--------|
| [imgproc](./modules/imgproc/README.md) | 图像处理：滤波、形态学、几何变换、颜色空间 | ✅ 完成 |
| [videoio](./modules/videoio/README.md) | 视频IO：VideoCapture、VideoWriter | ✅ 完成 |
| [video](./modules/video/README.md) | 视频分析：KCF、MOSSE跟踪、光流 | ✅ 完成 |

### Stage 3: 中级

| 模块 | 描述 | 状态 |
|--------|-------------|--------|
| [features2d](./modules/features2d/README.md) | 特征检测：SIFT/SURF/ORB、BFMatcher/FLANN | ✅ 完成 |
| [calib3d](./modules/calib3d/README.md) | 相机标定：内外参、立体匹配 | ✅ 完成 |
| [objdetect](./modules/objdetect/README.md) | 对象检测：Haar、HOG、SSD/YOLO | ✅ 完成 |

### Stage 4: 高级

| 模块 | 描述 | 状态 |
|--------|-------------|--------|
| [dnn](./modules/dnn/README.md) | 深度神经网络：ONNX、TensorFlow、PyTorch模型加载 | ✅ 完成 |
| [photo](./modules/photo/README.md) | 计算摄影：HDR、去噪、图像修复 | ✅ 完成 |
| [stitching](./modules/stitching/README.md) | 图像拼接：全景、多频段融合 | ✅ 完成 |

### Stage 5: 专家

| 模块 | 描述 | 状态 |
|--------|-------------|--------|
| [gapi](./modules/gapi/README.md) | 图引擎：kernel开发、异步执行 | ✅ 完成 |
| [ml](./modules/ml/README.md) | 机器学习：SVM、决策树、神经网络 | ✅ 完成 |
| [flann](./modules/flann/README.md) | FLANN：KD树、LSH、层次聚类 | ✅ 完成 |
| [world](./modules/world/README.md) | 统一入口模块 | 📋 计划中 |

---

## 模块文档结构

每个模块遵循以下结构：

```markdown
# 模块名称

## 1. 概述
## 2. 核心数据结构
## 3. 核心API
## 4. 实现分析
## 5. 代码示例
## 6. 练习题
## 7. 参考资料
```

---

## 开发指南

### Git 工作流程

```bash
# 1. 为每个模块创建新分支
git checkout -b module/core

# 2. 修改并提交
git add .
git commit -m "docs(core): 添加Mat结构详细分析"

# 3. 推送到远程
git push -u origin module/core
```

### 提交信息规范

格式: `<类型>(<模块>): <描述>`

类型:
- `docs` - 仅文档
- `fix` - Bug修复
- `feat` - 新功能
- `refactor` - 重构
- `test` - 测试更新

示例:
```bash
git commit -m "docs(core): 添加Mat内存模型分析"
git commit -m "docs(imgproc): 添加滤波算法指南"
git commit -m "feat(core): 添加练习题"
```

---

## 参考资料

- [官方文档](https://docs.opencv.org/4.14.0/)
- [OpenCV源码](https://github.com/opencv/opencv)
- [OpenCV教程](https://docs.opencv.org/4.14.0/tutorials/tutorials.html)
- [Learning OpenCV 4](https://www.oreilly.com/library/view/learning-opencv-4/)

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始提交，包含学习路线图和core模块指南 |