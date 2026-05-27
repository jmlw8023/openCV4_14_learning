# OpenCV photo 模块学习指南

> **模块**: photo
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 4 - 高级

---

## 目录

1. [概述](./README.md#1概述)
2. [图像去噪](./README.md#2图像去噪)
3. [HDR 合成](./README.md#3hdr-合成)
4. [图像修复](./README.md#4图像修复)
5. [其他功能](./README.md#5其他功能)
6. [代码示例](./README.md#6代码示例)
7. [练习题](./README.md#7练习题)
8. [参考资料](./README.md#8参考资料)

---

## 1.概述

**photo** (Computational Photography) 模块提供高级图像处理功能:

| 功能 | 描述 |
|---------|-------------|
| **去噪** | 非局部均值去噪 (NLB, Denoising) |
| **HDR** | 高动态范围成像 |
| **修复** | 图像修复 (Inpainting) |
| **滤镜** | 铅笔滤镜、素描效果 |

**头文件**: `opencv2/photo.hpp`

---

## 2.图像去噪

### 2.1 非局部均值去噪 (NLB)

```cpp
// ============================================
// C++ NLB 去噪
// ============================================
#include <opencv2/photo.hpp>

// 快速 NLB (推荐, 推荐参数: h=10)
// h: 滤波强度, 较大的 h 更多去噪但可能丢失细节
// hsearch: 搜索窗口大小
// templateWindowSize: 模板窗口大小
fastNlMeansDenoising(img, denoised, 10, 7, 21);

// 彩色图像 NLB
fastNlMeansDenoisingColored(img, denoised, 10, 10, 7, 21);

// 详细参数版本
fastNlMeansDenoising(img, denoised, 10, 7, 21,
                      TemplateWindowSize::W3,
                      SearchWindowSize::W21);

// 多帧去噪 (更快)
// frames: 多帧图像序列
fastNlMeansDenoisingMulti(frames, denoised, 0, 7, 21);
fastNlMeansDenoisingColoredMulti(frames, denoised, 0, 10, 10, 7, 21);
```

```python
# ============================================
# Python NLB 去噪
# ============================================
import cv2

# 快速 NLB
denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

# 彩色图像 NLB
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# 多帧去噪
denoised = cv2.fastNlMeansDenoisingMulti(frames, 0, None, 10, 7, 21)
denoised = cv2.fastNlMeansDenoisingColoredMulti(frames, 0, None, 10, 10, 7, 21)

# 参数说明:
# h: 滤波强度 (10 推荐)
# hForColorComponents: 彩色图像的 h
# templateWindowSize: 模板窗口 (默认 7)
# searchWindowSize: 搜索窗口 (默认 21)
```

### 2.2 去噪参数对比

| 参数 | 值 | 影响 |
|------|-----|------|
| h | 3-20 | 强度, 大→更多去噪但模糊 |
| templateWindow | 7 | 模板大小, 影响计算精度 |
| searchWindow | 21 | 搜索窗口, 大→更好的去噪但慢 |
| hForColor | 同 h | 彩色图像专用强度 |

---

## 3.HDR 合成

### 3.1 流程

```
拍摄不同曝光的多帧
       ↓
合并为一张 HDR (曝光融合)
       ↓
色调映射 (Tone Mapping) → 可显示的 LDR
```

### 3.2 HDR 合并

```cpp
// ============================================
// C++ HDR 合成
// ============================================
#include <opencv2/photo.hpp>

// 准备曝光序列和曝光时间
vector<Mat> exposures;  // 不同曝光的图像
vector<float> times;    // 曝光时间 (秒)

// 合并 HDR
Mat hdr;
Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
mergeDebevec->process(exposures, hdr, times);

cout << "HDR 图像范围: " << hdr << endl;

// 保存 HDR (EXR 格式)
imwrite("output.exr", hdr);

// 或者使用 Mertens 融合 (无需曝光时间)
Ptr<MergeMertens> mergeMertens = createMergeMertens();
Mat fusion;
mergeMertens->process(exposures, fusion);
fusion.convertTo(fusion, CV_8UC3, 255);
imwrite("fusion.png", fusion);
```

```python
# ============================================
# Python HDR 合成
# ============================================
import cv2
import numpy as np

# 准备曝光序列和曝光时间
# exposures: [img1, img2, img3, ...]
# times: [1/500, 1/60, 1/30, ...] (秒)

# 方法1: 曝光融合 (Debevec)
merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(exposures, times=times)
cv2.imwrite("output.exr", hdr)

# 方法2: Mertens 融合 (无需曝光时间)
merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(exposures)
fusion = (fusion * 255).astype(np.uint8)
cv2.imwrite("fusion.png", fusion)

# 方法3: Exposure fusion (简单高效)
# fusion = cv2.createMergeMertens().process(exposures)
```

### 3.3 色调映射 (Tone Mapping)

```cpp
// ============================================
// C++ 色调映射
// ============================================
// Drago 色调映射 (参数少, 效果好)
Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0, 1.0, 0.7);
Mat mapped;
tonemapDrago->process(hdr, mapped);
mapped.convertTo(mapped, CV_8UC3, 255);

// Reinhard 色调映射
Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5, 0.5, 0.24, 0.24);
Mat mapped;
tonemapReinhard->process(hdr, mapped);
mapped.convertTo(mapped, CV_8UC3, 255);

// Mantiuk 色调映射
Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(1.0, 0.7, 1.0);
Mat mapped;
tonemapMantiuk->process(hdr, mapped);
mapped.convertTo(mapped, CV_8UC3, 255);
```

```python
# ============================================
# Python 色调映射
# ============================================
# Drago 色调映射
tonemap_drago = cv2.createTonemapDrago(1.0, 1.0, 0.7)
mapped = tonemap_drago.process(hdr)
mapped = (mapped * 255).astype(np.uint8)

# Reinhard 色调映射
tonemap_reinhard = cv2.createTonemapReinhard(1.5, 0.5, 0.24, 0.24)
mapped = tonemap_reinhard.process(hdr)
mapped = (mapped * 255).astype(np.uint8)

# Mantiuk 色调映射
tonemap_mantiuk = cv2.createTonemapMantiuk(1.0, 0.7, 1.0)
mapped = tonemap_mantiuk.process(hdr)
mapped = (mapped * 255).astype(np.uint8)
```

---

## 4.图像修复

### 4.1 基于Telea的方法

```cpp
// ============================================
// C++ 图像修复
// ============================================
#include <opencv2/photo.hpp>

// 创建修复掩码 (非零像素为需要修复的区域)
Mat mask = imread("mask.png", IMREAD_GRAYSCALE);
// mask: 255 = 待修复, 0 = 正常

// 简单修复
Mat repaired;
inpaint(img, mask, repaired, 3, INPAINT_TELEA);

// 区域填充修复 (更好地保留纹理)
Mat repaired;
inpaint(img, mask, repaired, 5, INPAINT_TELEA);

// NS (Navier-Stokes) 方法
Mat repaired;
inpaint(img, mask, repaired, 3, INPAINT_NS);
```

```python
# ============================================
# Python 图像修复
# ============================================
import cv2
import numpy as np

# 创建修复掩码
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Telea 方法 (推荐)
repaired = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# NS 方法
repaired = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

# 参数说明:
# inpaint(img, mask, inpaintRadius, flags)
# inpaintRadius: 邻域半径 (影响修复纹理范围)
```

### 4.2 修复掩码制作

```cpp
// ============================================
// C++ 掩码制作
// ============================================
// 方法1: 从已知缺陷图像生成
Mat gray = cvtColor(img, COLOR_BGR2GRAY);
Mat mask = gray < 50;  // 假设暗区域是缺陷
mask.convertTo(mask, CV_8UC1, 255);

// 方法2: 从二值图像获取
Mat binary, mask;
threshold(gray, binary, 127, 255, THRESH_BINARY);
mask = binary;

// 方法3: 手动绘制
Mat mask = Mat::zeros(img.size(), CV_8UC1);
circle(mask, Point(100, 100), 20, 255, -1);  // 绘制圆形掩码
```

```python
# ============================================
# Python 掩码制作
# ============================================
# 方法1: 从已知缺陷图像生成
mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

# 方法2: 手动绘制
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, (100, 100), 20, 255, -1)

# 方法3: 从轮廓生成
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, contours, -1, 255, -1)
```

---

## 5.其他功能

### 5.1 铅笔滤镜

```cpp
// ============================================
// C++ 铅笔滤镜
// ============================================
#include <opencv2/photo.hpp>

// 彩色铅笔效果
Mat pencilImg, colorPencilImg;
pencilSketch(img, pencilImg, colorPencilImg, 10, 0.1, 0.03);
// pencilImg: 灰度素描
// colorPencilImg: 彩色素描

// 独立获取
Mat graySketch, colorSketch;
pencilSketch(img, graySketch, colorSketch, sigma_s=10, sigma_r=0.1);
```

```python
# ============================================
# Python 铅笔滤镜
# ============================================
# 彩色铅笔效果
gray_sketch, color_sketch = cv2.pencilSketch(img, sigma_s=10, sigma_r=0.1, shade_factor=0.03)

# gray_sketch: 灰度素描
# color_sketch: 彩色素描

# 参数:
# sigma_s: 邻域大小 (影响线条平滑度)
# sigma_r: 颜色相似度阈值 (影响细节保留)
# shade_factor: 明暗因子
```

### 5.2 区域去噪

```cpp
// ============================================
// C++ 区域去噪 (仅对指定区域)
// ============================================
// 创建区域掩码
Mat regionMask = Mat::zeros(img.size(), CV_8UC1);
rectangle(regionMask, Rect(100, 100, 200, 200), 255, -1);

// 仅对掩码区域去噪
Mat temp, denoised = img.clone();
fastNlMeansDenoising(img, temp, 10, 7, 21);
temp.copyTo(denoised, regionMask);
```

```python
# ============================================
# Python 区域去噪
# ============================================
# 创建区域掩码
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (100, 100), (300, 300), 255, -1)

# 仅对掩码区域去噪
temp = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
denoised = img.copy()
denoised[np.where(mask == 255)] = temp[np.where(mask == 255)]
```

---

## 6.代码示例

### 6.1 批量图像去噪

```cpp
// ============================================
// C++ 批量去噪
// ============================================
#include <opencv2/photo.hpp>

void batchDenoise(const vector<string>& inputFiles,
                  const string& outputDir) {
    for (const auto& inputFile : inputFiles) {
        Mat img = imread(inputFile);
        if (img.empty()) continue;

        Mat denoised;
        if (img.channels() == 3) {
            fastNlMeansDenoisingColored(img, denoised, 10, 10, 7, 21);
        } else {
            fastNlMeansDenoising(img, denoised, 10, 7, 21);
        }

        string outputFile = outputDir + "/" + getFilename(inputFile);
        imwrite(outputFile, denoised);
    }
}
```

```python
# ============================================
# Python 批量去噪
# ============================================
import cv2
import os

def batch_denoise(input_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        img = cv2.imread(input_file)
        if img is None:
            continue

        if len(img.shape) == 3:  # 彩色
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        else:  # 灰度
            denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        output_file = os.path.join(output_dir, os.path.basename(input_file))
        cv2.imwrite(output_file, denoised)
```

### 6.2 HDR 完整流程

```cpp
// ============================================
// C++ HDR 完整流程
// ============================================
#include <opencv2/photo.hpp>

void processHDR(const vector<Mat>& exposures,
                const vector<float>& times) {
    // 1. 合并 HDR
    Ptr<MergeDebevec> merger = createMergeDebevec();
    Mat hdr;
    merger->process(exposures, hdr, times);

    // 2. 色调映射
    Ptr<TonemapDrago> tonemap = createTonemapDrago(1.0, 1.0, 0.7);
    Mat mapped;
    tonemap->process(hdr, mapped);

    // 3. 转换到 8-bit 显示
    Mat display;
    mapped.convertTo(display, CV_8UC3, 255);

    // 4. 保存结果
    imwrite("hdr_output.png", display);

    // 5. 可选: 保存 HDR 文件
    imwrite("hdr_output.exr", hdr);
}
```

```python
# ============================================
# Python HDR 完整流程
# ============================================
import cv2
import numpy as np

def process_hdr(exposures, times):
    # 1. 合并 HDR
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(exposures, times=times)

    # 2. 色调映射
    tonemap = cv2.createTonemapDrago(1.0, 1.0, 0.7)
    mapped = tonemap.process(hdr)

    # 3. 转换到 8-bit 显示
    display = (mapped * 255).astype(np.uint8)

    # 4. 保存结果
    cv2.imwrite("hdr_output.png", display)

    return display
```

### 6.3 图像修复示例

```cpp
// ============================================
// C++ 图像修复示例
// ============================================
#include <opencv2/photo.hpp>

void removeObject(const Mat& img, const Mat& mask) {
    Mat repaired;

    // Telea 方法 (更好的纹理保留)
    inpaint(img, mask, repaired, 5, INPAINT_TELEA);

    // 或者 NS 方法 (更平滑但可能模糊)
    // inpaint(img, mask, repaired, 3, INPAINT_NS);

    imshow("原始图像", img);
    imshow("修复掩码", mask);
    imshow("修复结果", repaired);
    waitKey(0);
}
```

```python
# ============================================
# Python 图像修复示例
# ============================================
import cv2
import numpy as np

def remove_object(img, mask):
    # Telea 方法
    repaired = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    # NS 方法
    # repaired = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Repaired", repaired)
    cv2.waitKey(0)

    return repaired
```

### 6.4 素描效果

```cpp
// ============================================
// C++ 素描效果
// ============================================
#include <opencv2/photo.hpp>

void sketchEffect(const Mat& img) {
    Mat graySketch, colorSketch;

    // sigma_s: 邻域大小 (10-20 推荐)
    // sigma_r: 颜色相似度 (0.05-0.1 推荐)
    pencilSketch(img, graySketch, colorSketch, 10, 0.1, 0.03);

    imshow("灰度素描", graySketch);
    imshow("彩色素描", colorSketch);
    waitKey(0);
}
```

```python
# ============================================
# Python 素描效果
# ============================================
import cv2

def sketch_effect(img):
    gray_sketch, color_sketch = cv2.pencilSketch(img, sigma_s=10, sigma_r=0.1, shade_factor=0.03)

    cv2.imshow("Gray Sketch", gray_sketch)
    cv2.imshow("Color Sketch", color_sketch)
    cv2.waitKey(0)

    return gray_sketch, color_sketch
```

---

## 7.练习题

### 入门级
1. 对一张图片进行 NLB 去噪 (C++ / Python)
2. 实现图像修复 (去除水印)
3. 生成铅笔素描效果

### 中级
4. 实现 HDR 合成和色调映射
5. 实现多帧平均去噪
6. 实现局部区域去噪

### 高级
7. 实现批量图像处理流水线
8. 比较不同色调映射算法效果
9. 实现自定义去噪算法

### 挑战题
10. 实现多曝光 HDR 包围摄影系统
11. 实现基于深度学习的图像修复

---

## 8.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [photo 模块](https://docs.opencv.org/4.14.0/d4/d8b/group__photo.html) |
| NLB 去噪论文 | [Non-Local Means Denoising](https://www.ipol.im/w/articles/article/art_im2011_30/) |
| HDR 教程 | [HDR Imaging](https://docs.opencv.org/4.14.0/d2/df0/tutorial_photo.html) |
| Inpainting 论文 | [Object Removal](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/04/tip-2004-A三点.pdf) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 photo 模块文档 (C++/Python 双语) |