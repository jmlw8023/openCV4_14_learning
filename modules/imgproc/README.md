# OpenCV imgproc 模块学习指南

> **模块**: imgproc
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 2 - 基础

---

## 目录

1. [概述](./README.md#1概述)
2. [核心数据结构](./README.md#2核心数据结构)
3. [核心API](./README.md#3核心api)
4. [实现分析](./README.md#4实现分析)
5. [代码示例](./README.md#5代码示例)
6. [练习题](./README.md#6练习题)
7. [参考资料](./README.md#7参考资料)

---

## 1.概述

**imgproc** (图像处理) 提供全面的图像变换功能:

| 功能 | 描述 |
|---------|-------------|
| **几何变换** | 调整大小、旋转、扭曲、透视 |
| **滤波** | 模糊、高斯、中值、双边滤波 |
| **形态学** | 腐蚀、膨胀、开运算、闭运算、梯度 |
| **颜色空间** | BGR、HSV、Lab、YCrCb 转换 |
| **直方图** | 均衡化、反向投影 |
| **轮廓** | 查找、近似、矩 |
| **图像梯度** | Sobel、Laplacian、Canny |
| **阈值分割** | 二值化、Otsu、自适应 |

**头文件**: `opencv2/imgproc.hpp`

---

## 2.核心数据结构

### 2.1 Point, Size, Rect

```cpp
#include <opencv2/imgproc.hpp>

// 点 (支持整数和浮点)
Point2f pt(100.f, 200.f);   // 浮点坐标
Point pt2(100, 200);        // 整数坐标

// 尺寸
Size sz(640, 480);          // 整数尺寸
Size2f szf(640.5f, 480.5f); // 浮点尺寸

// 矩形 (x, y, width, height)
Rect roi(100, 100, 200, 150);
roi.x; roi.y;              // 左上角坐标
roi.width; roi.height;     // 宽和高
roi.area();                // 面积 = width * height
roi.contains(pt);          // 检查点是否在矩形内
```

### 2.2 RotatedRect (旋转矩形)

```cpp
// 旋转矩形: 中心点 + 尺寸 + 旋转角度
RotatedRect rr(Point2f(center.x, center.y), Size2f(w, h), angle);

// 访问成员
rr.center;    // 中心点
rr.size;      // 宽度和高度
rr.angle;     // 旋转角度 (度)

// 获取边界框
Rect2f bbox = rr.boundingRect();
```

---

## 3.核心API

### 3.1 调整大小和几何变换

```cpp
Mat dst;

// 调整大小
resize(src, dst, Size(w, h));                    // 指定尺寸
resize(src, dst, Size(), 0.5, 0.5, INTER_LINEAR); // 缩放因子

// 旋转 90°
rotate(src, dst, ROTATE_90_CLOCKWISE);

// 仿射变换
Mat M = getRotationMatrix2D(center, angle, scale);
warpAffine(src, dst, M, Size(w, h));

// 透视变换
Mat H = findHomography(srcPoints, dstPoints);
warpPerspective(src, dst, H, Size(w, h));
```

### 3.2 滤波

```cpp
Mat dst;

// 模糊 (均值)
blur(src, dst, Size(5, 5));

// 高斯模糊
GaussianBlur(src, dst, Size(5, 5), sigmaX);

// 中值模糊 (对椒盐噪声有效)
medianBlur(src, dst, 5);

// 双边滤波 (保边滤波)
bilateralFilter(src, dst, 9, 75, 75);

// 自定义卷积核
Mat kernel = (Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
filter2D(src, dst, -1, kernel);
```

### 3.3 形态学运算

```cpp
Mat dst;

// 腐蚀 (收缩)
erode(src, dst, getStructuringElement(MORPH_RECT, Size(3,3)));

// 膨胀 (扩张)
dilate(src, dst, getStructuringElement(MORPH_RECT, Size(3,3)));

// 开运算 (腐蚀后膨胀, 去噪)
morphologyEx(src, dst, MORPH_OPEN, kernel);

// 闭运算 (膨胀后腐蚀, 填孔)
morphologyEx(src, dst, MORPH_CLOSE, kernel);

// 形态学梯度 (外轮廓减内轮廓)
morphologyEx(src, dst, MORPH_GRADIENT, kernel);

// 黑帽和顶帽
morphologyEx(src, dst, MORPH_BLACKHAT, kernel);
morphologyEx(src, dst, MORPH_TOPHAT, kernel);
```

### 3.4 颜色空间转换

```cpp
// BGR 转灰度
cvtColor(src, dst, COLOR_BGR2GRAY);

// BGR 转 HSV
cvtColor(src, dst, COLOR_BGR2HSV);

// BGR 转 Lab (适合颜色感知差异)
cvtColor(src, dst, COLOR_BGR2Lab);

// 分割通道
vector<Mat> bgr;
split(src, bgr);

// 合并通道
merge(bgr, dst);
```

### 3.5 阈值分割

```cpp
double thresh = 128;
double maxVal = 255;

// 简单阈值
threshold(src, dst, thresh, maxVal, THRESH_BINARY);

// Otsu 法 (自动计算阈值)
threshold(src, dst, 0, maxVal, THRESH_BINARY | THRESH_OTSU);

// 自适应阈值 (适合不均匀照明)
adaptiveThreshold(src, dst, maxVal, ADAPTIVE_THRESH_GAUSSIAN_C,
                   THRESH_BINARY, 11, 2);
```

### 3.6 边缘检测

```cpp
Mat dst;

// Sobel (梯度)
Sobel(src, dst, CV_8U, 1, 0, 3);  // dx=1, dy=0, 卷积核=3

// Laplacian (二阶导数)
Laplacian(src, dst, CV_8U, 3);

// Canny 边缘检测
Canny(src, dst, 50, 150);  // 低阈值, 高阈值
```

---

## 4.实现分析

### 4.1 高斯模糊算法

```
1. 创建 1D 高斯核
2. 可分离卷积:
   - 用 1D 核卷积行
   - 用转置的核卷积结果
3. 复杂度: O(w*h*ksize) vs O(w*h*ksize²) 2D 卷积
```

### 4.2 Canny 边缘检测步骤

```
1. 高斯平滑
2. 梯度计算 (Sobel)
3. 非极大值抑制
4. 双阈值处理
5. 边缘连接 (滞后阈值)
```

---

## 5.代码示例

### 5.1 基础图像处理流水线

```cpp
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    Mat img = imread("photo.jpg");
    Mat gray, blurred, edges;

    // 转换为灰度图
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 应用高斯模糊
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);

    // 边缘检测
    Canny(blurred, edges, 50, 150);

    imshow("原图", img);
    imshow("边缘", edges);
    waitKey(0);

    return 0;
}
```

### 5.2 直方图均衡化

```cpp
// 直方图均衡化 - 增强对比度
void equalizeImage(const Mat& src) {
    Mat gray, eq;

    // 转灰度
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // 均衡化
    equalizeHist(gray, eq);

    imshow("原图", gray);
    imshow("均衡化", eq);
    waitKey(0);
}
```

### 5.3 形态学操作

```cpp
// 形态学操作处理二值图像
void processWithMorphology(const Mat& src) {
    Mat binary, opened, closed;

    // 阈值化为二值图像
    threshold(src, binary, 127, 255, THRESH_BINARY);

    // 创建结构元素
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    // 开运算 (去噪)
    morphologyEx(binary, opened, MORPH_OPEN, kernel);

    // 闭运算 (填孔)
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);

    imshow("原图", src);
    imshow("开运算", opened);
    imshow("闭运算", closed);
    waitKey(0);
}
```

---

## 6.练习题

### 入门级
1. 实现不同插值方法的图像缩放
2. 将图像转换为 HSV 并提取饱和度通道
3. 使用 Otsu 法进行图像阈值分割

### 中级
4. 实现自定义卷积滤波 (锐化、浮雕)
5. 使用形态学操作去除二值图像中的噪声
6. 查找并绘制图像轮廓

### 高级
7. 实现透视变换 (手动计算单应性)
8. 构建使用 imgproc 函数的图像拼接流水线

---

## 7.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [imgproc 模块](https://docs.opencv.org/4.14.0/d7/da4/group__imgproc.html) |
| resize 参考 | [resize](https://docs.opencv.org/4.14.0/d4/d86/group__imgproc__filter.html#ga4d0a3e5d0ae5e42fb6d18de5b70f6f02) |
| filter2D 参考 | [filter2D](https://docs.opencv.org/4.14.0/d4/d86/group__imgproc__filter.html#gae67d2b4c1ae2ac9d3fef2c90029f3a62) |
| Canny 参考 | [Canny](https://docs.opencv.org/4.14.0/dd/d1a/group__imgproc__feature.html#ga04723e100754a3e6f8858d1ac53d0db5) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 imgproc 模块文档 |