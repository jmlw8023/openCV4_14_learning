# OpenCV stitching 模块学习指南

> **模块**: stitching
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 4 - 高级

---

## 目录

1. [概述](./README.md#1概述)
2. [拼接流水线](./README.md#2拼接流水线)
3. [特征匹配与融合](./README.md#3特征匹配与融合)
4. [代码示例](./README.md#4代码示例)
5. [练习题](./README.md#5练习题)
6. [参考资料](./README.md#6参考资料)

---

## 1.概述

**stitching** 模块提供图像全景拼接功能:

| 功能 | 描述 |
|---------|-------------|
| **自动拼接** | 无需标定直接拼接 |
| **特征匹配** | SIFT/ORB 特征 + Bundle Adj. |
| ** Exposure Correction** | 曝光补偿与融合 |
| **多频段融合** | 多频段金字塔融合 |

**头文件**: `opencv2/stitching.hpp`

---

## 2.拼接流水线

### 2.1 拼接步骤

```
输入图像序列
       ↓
特征检测 (ORB/SIFT)
       ↓
特征匹配 (BFMatcher/FlannBasedMatcher)
       ↓
相机参数估计 (Bundle Adjustment)
       ↓
曝光补偿
       ↓
图像融合 (多频段)
       ↓
输出全景图
```

### 2.2 Stitcher 类

```cpp
// ============================================
// C++ 拼接器
// ============================================
#include <opencv2/stitching.hpp>

// 创建拼接器
Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
// 或
Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);  // 扫描式

// 设置特征检测器
stitcher->setFeaturesMatcher(makePtr<OrbFeaturesMatcher>());
// 或
stitcher->setFeaturesMatcher(makePtr<detail::FeaturesMatcher>(
    makePtr<BFMatcher>(NORM_HAMMING)));

// 设置融合器
stitcher->setBlender(makePtr<detail::MultiBandBlender>(true));
// 或
stitcher->setBlender(makePtr<detail::FeatherBlender>(0.01));

// 执行拼接
Mat panorama;
Stitcher::Status status = stitcher->stitch(images, panorama);

if (status == Stitcher::OK) {
    imwrite("panorama.jpg", panorama);
}
```

```python
# ============================================
# Python 拼接器
# ============================================
import cv2

# 创建拼接器
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
# 或
stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

# 执行拼接
status, panorama = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    cv2.imwrite("panorama.jpg", panorama)
else:
    print(f"拼接失败: {status}")
```

---

## 3.特征匹配与融合

### 3.1 特征检测器选项

```cpp
// ============================================
// C++ 特征检测器配置
// ============================================
// ORB 特征 (快速, 推荐)
Ptr<ORB> detector = ORB::create(2000);
stitcher->setFeaturesFinder(detector);

// SIFT 特征 (精确, 但较慢)
Ptr<SIFT> detector = SIFT::create(2000);
stitcher->setFeaturesFinder(detector);

// AKAZE 特征
Ptr<AKAZE> detector = AKAZE::create();
stitcher->setFeaturesFinder(detector);
```

```python
# ============================================
# Python 特征检测器配置
# ============================================
# ORB 特征 (快速)
stitcher.setFeaturesFinder(cv2.ORB_create(2000))

# SIFT 特征 (精确, 需要 xfeatures2d)
# stitcher.setFeaturesFinder(cv2.SIFT_create(2000))
```

### 3.2 融合器选项

```cpp
// ============================================
// C++ 融合器配置
// ============================================
// 多频段融合 (推荐, 质量最高)
Ptr<MultiBandBlender> blender = MultiBandBlender::create(true, 5);
stitcher->setBlender(blender);

// 羽化融合 (快速, 质量一般)
Ptr<FeatherBlender> blender = FeatherBlender::create(0.01);
stitcher->setBlender(blender);

// 无融合 (仅拼接)
stitcher->setBlender(makePtr<detail::Blender>());
```

```python
# ============================================
# Python 融合器配置
# ============================================
# 多频段融合 (推荐)
stitcher.setBlender(cv2.detail_MultiBandBlender_create(True, 5))

# 羽化融合
stitcher.setBlender(cv2.detail_FeatherBlender_create(0.01))
```

### 3.3 曝光补偿

```cpp
// ============================================
// C++ 曝光补偿
// ============================================
// 自动曝光补偿
stitcher->setExposureCompensator(
    makePtr<detail::BlocksExposureCompensator>());

// 详细曝光补偿
Ptr<detail::ExposureCompensator> compensator =
    makePtr<detail::BlocksExposureCompensator>(100, 100);
stitcher->setExposureCompensator(compensator);

// 可选: Gain 补偿, Channel 补偿, No 补偿
stitcher->setExposureCompensator(
    makePtr<detail::GainCompensator>());
stitcher->setExposureCompensator(
    makePtr<detail::ChannelsCompensator>());
```

```python
# ============================================
# Python 曝光补偿
# ============================================
# 自动曝光补偿
stitcher.setExposureCompensator(cv2.detail_BlocksExposureCompensator_create())

# Gain 补偿
stitcher.setExposureCompensator(cv2.detail_GainCompensator_create())

# No 补偿
stitcher.setExposureCompensator(cv2.detail_NoExposureCompensator_create())
```

---

## 4.代码示例

### 4.1 基础全景拼接

```cpp
// ============================================
// C++ 基础拼接
// ============================================
#include <opencv2/stitching.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // 读取图像
    vector<Mat> images;
    images.push_back(imread("image1.jpg"));
    images.push_back(imread("image2.jpg"));
    images.push_back(imread("image3.jpg"));

    // 检查图像是否有效
    for (auto& img : images) {
        if (img.empty()) {
            cout << "无法读取图像" << endl;
            return -1;
        }
    }

    // 创建拼接器
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);

    // 拼接
    Mat panorama;
    Stitcher::Status status = stitcher->stitch(images, panorama);

    if (status == Stitcher::OK) {
        imshow("全景图", panorama);
        imwrite("panorama.jpg", panorama);
        waitKey(0);
    } else {
        cout << "拼接失败: " << status << endl;
    }

    return 0;
}
```

```python
# ============================================
# Python 基础拼接
# ============================================
import cv2

def main():
    # 读取图像
    images = [
        cv2.imread("image1.jpg"),
        cv2.imread("image2.jpg"),
        cv2.imread("image3.jpg")
    ]

    # 检查图像是否有效
    for i, img in enumerate(images):
        if img is None:
            print(f"无法读取图像 {i+1}")
            return -1

    # 创建拼接器
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

    # 拼接
    status, panorama = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imshow("全景图", panorama)
        cv2.imwrite("panorama.jpg", panorama)
        cv2.waitKey(0)
    else:
        print(f"拼接失败: {status}")

    return 0

if __name__ == "__main__":
    main()
```

### 4.2 高级配置拼接

```cpp
// ============================================
// C++ 高级配置拼接
// ============================================
Mat advancedStitching(const vector<Mat>& images) {
    // 创建拼接器
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);

    // 1. 设置特征检测器 (ORB)
    Ptr<ORB> detector = ORB::create(2000);
    stitcher->setFeaturesFinder(detector);

    // 2. 设置特征匹配器
    Ptr<detail::FeaturesMatcher> matcher =
        makePtr<detail::BestOf2NearestMatcher>(false, 0.5);
    stitcher->setFeaturesMatcher(matcher);

    // 3. 设置相机参数估计器
    Ptr<detail::BundleAdjusterBase> ba =
        makePtr<detail::HomographyBasedEstimator>();
    stitcher->setBundleAdjuster(ba);

    // 4. 设置曝光补偿
    stitcher->setExposureCompensator(
        makePtr<detail::BlocksExposureCompensator>(50, 50));

    // 5. 设置融合器 (多频段, 5层)
    Ptr<detail::Blender> blender =
        makePtr<detail::MultiBandBlender>(true, 5);
    stitcher->setBlender(blender);

    // 6. 执行拼接
    Mat panorama;
    Stitcher::Status status = stitcher->stitch(images, panorama);

    if (status == Stitcher::OK) {
        return panorama;
    } else {
        cout << "拼接失败" << endl;
        return Mat();
    }
}
```

```python
# ============================================
# Python 高级配置拼接
# ============================================
def advanced_stitching(images):
    # 创建拼接器
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

    # 设置特征检测器 (ORB)
    stitcher.setFeaturesFinder(cv2.ORB_create(2000))

    # 设置特征匹配器
    # stitcher.setFeaturesMatcher(...)

    # 设置曝光补偿
    stitcher.setExposureCompensator(
        cv2.detail_BlocksExposureCompensator_create())

    # 设置融合器 (多频段)
    stitcher.setBlender(cv2.detail_MultiBandBlender_create(True, 5))

    # 执行拼接
    status, panorama = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return panorama
    else:
        print(f"拼接失败: {status}")
        return None
```

### 4.3 扫描式拼接 (适合文档扫描)

```cpp
// ============================================
// C++ 扫描式拼接
// ============================================
#include <opencv2/stitching.hpp>

Mat scanStitching(const vector<Mat>& images) {
    // 扫描模式适合平面场景 (文档、白板)
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);

    // 扫描模式不需要特征匹配, 更快
    Mat panorama;
    Stitcher::Status status = stitcher->stitch(images, panorama);

    return status == Stitcher::OK ? panorama : Mat();
}
```

```python
# ============================================
# Python 扫描式拼接
# ============================================
def scan_stitching(images):
    # 扫描模式适合平面场景
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    status, panorama = stitcher.stitch(images)

    return panorama if status == cv2.Stitcher_OK else None
```

### 4.4 鱼眼镜头拼接

```cpp
// ============================================
// C++ 鱼眼镜头拼接
// ============================================
Mat fisheyeStitching(const vector<Mat>& images,
                      const vector<Mat>& K,
                      const vector<Mat>& D) {
    // 创建拼接器
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);

    // 鱼眼相机需要校正
    vector<Mat> undistorted;
    for (size_t i = 0; i < images.size(); i++) {
        Mat und;
        fisheye::undistortImage(images[i], und, K[i], D[i]);
        undistorted.push_back(und);
    }

    // 拼接校正后的图像
    Mat panorama;
    Stitcher::Status status = stitcher->stitch(undistorted, panorama);

    return status == Stitcher::OK ? panorama : Mat();
}
```

```python
# ============================================
# Python 鱼眼镜头拼接
# ============================================
import cv2
import numpy as np

def fisheye_stitching(images, K_list, D_list):
    # 鱼眼相机需要校正
    undistorted = []
    for i, img in enumerate(images):
        und = cv2.fisheye_undistortImage(img, K_list[i], D_list[i])
        undistorted.append(und)

    # 拼接校正后的图像
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(undistorted)

    return panorama if status == cv2.Stitcher_OK else None
```

---

## 5.练习题

### 入门级
1. 使用默认配置拼接 2-3 张图像 (C++ / Python)
2. 使用扫描模式拼接文档图像
3. 比较不同特征检测器 (ORB/SIFT) 的拼接效果

### 中级
4. 配置多频段融合参数
5. 实现鱼眼镜头图像的拼接校正
6. 处理拼接中的鬼影问题

### 高级
7. 实现自定义特征匹配器
8. 实现 360° 全景相机拼接
9. 实现视频帧拼接

### 挑战题
10. 实现多相机同步拼接系统
11. 实现基于深度学习的图像拼接质量评估

---

## 6.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [stitching 模块](https://docs.opencv.org/4.14.0/d4/d8b/group__stitching.html) |
| Stitcher 类 | [Stitcher](https://docs.opencv.org/4.14.0/d8/d74/classcv_1_1Stitcher.html) |
| OpenCV Stitching 教程 | [Stitching Tutorial](https://docs.opencv.org/4.14.0/d8/dcc/tutorial_stitcher.html) |
| 拼接论文 | [Automatic Panoramic Image Stitching](https://research.microsoft.com/en-us/um/people/agarwala/publications/2007/agarwala07.pdf) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 stitching 模块文档 (C++/Python 双语) |