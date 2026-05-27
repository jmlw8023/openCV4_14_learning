# OpenCV 学习路线图（基础 → 专家）

> 从入门到专家的系统性学习指南

## 学习阶段概览

| 阶段 | 级别 | 核心模块 | 目标 |
|------|------|----------|------|
| **Stage 1** | 入门 | core, imgcodecs, highgui | 掌握图像读写、显示、基本操作 |
| **Stage 2** | 基础 | imgproc, videoio, video | 掌握图像处理、视频分析基础 |
| **Stage 3** | 进阶 | features2d, calib3d, objdetect | 掌握特征提取、相机标定、对象检测 |
| **Stage 4** | 高级 | dnn, photo, stitching | 掌握深度学习、图像修复、拼接 |
| **Stage 5** | 专家 | gapi, ml, flann | 掌握图引擎、机器学习、高维搜索 |

---

## Stage 1：入门 🚀

### 目标
- 理解 OpenCV 基本架构
- 掌握图像的加载、显示、保存
- 掌握 Mat 数据结构基本操作

### 模块与顺序

| 序号 | 模块 | 重点内容 | 预计时间 |
|------|------|----------|----------|
| 1.1 | [core](./core/README.md) | Mat数据结构、数组操作、XML/YAML | 3-5天 |
| 1.2 | [imgcodecs](./imgcodecs/README.md) | imread/imwrite、图像格式解码 | 1-2天 |
| 1.3 | [highgui](./highgui/README.md) | namedWindow、imshow、Trackbar | 1-2天 |

### 阶段产出
- 能独立完成图像的读取、处理、显示流程
- 理解 Mat 的内存模型（行连续、ROI）
- 掌握基本的像素操作

---

## Stage 2：基础 🔰

### 目标
- 掌握常用图像处理算法
- 理解视频捕获与处理流程
- 能实现简单的视频分析应用

### 模块与顺序

| 序号 | 模块 | 重点内容 | 预计时间 |
|------|------|----------|----------|
| 2.1 | [imgproc](./imgproc/README.md) | 滤波、形态学、几何变换、颜色空间 | 5-7天 |
| 2.2 | [videoio](./videoio/README.md) | VideoCapture、VideoWriter | 2-3天 |
| 2.3 | [video](./video/README.md) | KCF、MOSSE跟踪、OpticalFlow | 3-5天 |

### 阶段产出
- 实现滤镜、边缘检测、轮廓查找
- 完成视频读取、处理、保存
- 能实现简单的目标跟踪应用

---

## Stage 3：进阶 📈

### 目标
- 掌握特征检测与匹配
- 理解相机标定与三维重建原理
- 能实现对象检测基本流程

### 模块与顺序

| 序号 | 模块 | 重点内容 | 预计时间 |
|------|------|----------|----------|
| 3.1 | [features2d](./features2d/README.md) | SIFT/SURF/ORB、BFMatcher/FLANN | 5-7天 |
| 3.2 | [calib3d](./calib3d/README.md) | 相机标定、畸变校正、立体匹配 | 5-7天 |
| 3.3 | [objdetect](./objdetect/README.md) | Haar cascades、HOG、SSD/YOLO | 5-7天 |

### 阶段产出
- 能实现图像拼接（SIFT + 融合）
- 完成单目/双目相机标定
- 能部署简单的目标检测模型

---

## Stage 4：高级 🎯

### 目标
- 掌握深度神经网络部署
- 理解图像修复与增强技术
- 能实现专业的图像拼接

### 模块与顺序

| 序号 | 模块 | 重点内容 | 预计时间 |
|------|------|----------|----------|
| 4.1 | [dnn](./dnn/README.md) | ONNX/TensorFlow/PyTorch模型加载 | 5-7天 |
| 4.2 | [photo](./photo/README.md) | HDR、去噪、铅笔滤镜、修复 | 3-5天 |
| 4.3 | [stitching](./stitching/README.md) | 全景拼接、多频段融合 | 3-5天 |

### 阶段产出
- 能部署自定义深度学习模型进行推理
- 实现HDR合成、图像修复
- 完成360°全景拼接

---

## Stage 5：专家 👑

### 目标
- 理解图计算引擎架构
- 掌握高维特征索引
- 能进行机器学习模型训练

### 模块与顺序

| 序号 | 模块 | 重点内容 | 预计时间 |
|------|------|----------|----------|
| 5.1 | [gapi](./gapi/README.md) | 图引擎、kernel开发、异步执行 | 5-7天 |
| 5.2 | [flann](./flann/README.md) | KD树、LSH、层次聚类 | 3-5天 |
| 5.3 | [ml](./ml/README.md) | SVM、决策树、神经网络、Boosting | 5-7天 |

### 阶段产出
- 能开发自定义 GAPI kernel
- 实现大规模特征快速检索
- 完成模型训练与调优

---

## 学习技巧

### 1. 源码阅读方法
```
├── 先看头文件的类/函数声明
├── 再看示例代码理解用法
├── 最后读实现细节
└── 配合调试器单步跟踪
```

### 2. 实践项目建议

**入门阶段项目**：
- 图像编辑器（亮度、对比度、滤镜）
- 摄像头实时预览

**基础阶段项目**：
- 车牌检测预处理流水线
- 视频运动目标跟踪

**进阶阶段项目**：
-全景图像拼接
- AR标定助手

**高级阶段项目**：
- 部署自定义YOLO模型
- HDR照相机

**专家阶段项目**：
- 自定义GAPI图像处理pipeline
- 大规模人脸检索系统

### 3. 参考资源

- 📖 官方文档：https://docs.opencv.org/
- 📖 Learning OpenCV 4：https://www.oreilly.com/library/view/learning-opencv-4/
- 🐍 OpenCV-Python：https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- 💻 源码：https://github.com/opencv/opencv

---

## 学习记录

| 日期 | 完成阶段 | 备注 |
|------|----------|------|
| 2026/05/27 | Stage 1 | 刚开始 |

---

**下一步**：[开始学习 Stage 1 - core 模块](./core/README.md)