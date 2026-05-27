# OpenCV imgcodecs 模块学习指南

> **模块**: imgcodecs
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 1 - 入门

---

## 目录

1. [概述](./README.md#1概述)
2. [支持的格式](./README.md#2支持的格式)
3. [核心API](./README.md#3核心api)
4. [实现分析](./README.md#4实现分析)
5. [代码示例](./README.md#5代码示例)
6. [练习题](./README.md#6练习题)
7. [参考资料](./README.md#7参考资料)

---

## 1.概述

**imgcodecs** 提供图像解码和编码功能:

| 功能 | 描述 |
|---------|-------------|
| **图像加载** | 从文件读取各种格式的图像 |
| **图像保存** | 将图像写入各种格式的文件 |
| **格式检测** | 基于文件内容自动检测格式 |
| **参数控制** | 细粒度控制加载/保存参数 |

**头文件**: `opencv2/imgcodecs.hpp`

---

## 2.支持的格式

### 内置图像格式

| 格式 | 扩展名 | 编解码器 | 支持 |
|--------|-----------|-------|---------|
| JPEG | .jpg, .jpeg | libjpeg | ✅ |
| PNG | .png | libpng | ✅ |
| TIFF | .tif, .tiff | libtiff | ✅ |
| BMP | .bmp | 内置 | ✅ |
| WebP | .webp | libwebp | ✅ |
| OpenEXR | .exr | OpenEXR | ✅ |
| HDR | .hdr | OpenEXR | ✅ |
| PPM/PGM | .ppm, .pgm | 内置 | ✅ |
| SunRaster | .sr, .ras | 内置 | ✅ |

### 加载标志 (imread)

| 标志 | 描述 |
|------|-------------|
| `IMREAD_UNCHANGED` | 按原样加载 (含 alpha 通道) |
| `IMREAD_GRAYSCALE` | 转换为灰度图 |
| `IMREAD_COLOR` | 转换为 3通道 BGR |
| `IMREAD_ANYDEPTH` | 保留位深度 |
| `IMREAD_ANYCOLOR` | 加载为任意颜色格式 |
| `IMREAD_LOAD_GDAL` | 使用 GDAL 驱动 |

---

## 3.核心API

### 3.1 图像加载

```cpp
#include <opencv2/imgcodecs.hpp>

// 基本加载 (默认使用 IMREAD_COLOR)
Mat img = imread("photo.jpg");

// 带显式标志的加载
Mat img = imread("photo.png", IMREAD_GRAYSCALE);

// 检查是否加载成功
if (img.empty()) {
    // 处理错误
}

// 查看图像属性
img.rows;        // 高度
img.cols;        // 宽度
img.channels();  // 通道数 (灰度为1, 彩色为3)
img.depth();     // 位深度
```

### 3.2 图像保存

```cpp
#include <opencv2/imgcodecs.hpp>

// 基本保存 (格式由扩展名决定)
imwrite("output.jpg", img);
imwrite("output.png", img);

// 带参数的保存
vector<int> params = {
    IMWRITE_JPEG_QUALITY, 95,      // JPEG 质量 0-100
    IMWRITE_PNG_COMPRESSION, 9    // PNG 压缩级别 0-9
};
imwrite("output.jpg", img, params);

// PNG 带 alpha 通道保存
imwrite("output.png", img, {IMWRITE_PNG_COMPRESSION, 9});
```

### 3.3 图像参数常量

```cpp
// JPEG 参数
IMWRITE_JPEG_QUALITY       // 1-100 (默认 95)
IMWRITE_JPEG_PROGRESSIVE   // 启用渐进式 JPEG (0/1)
IMWRITE_JPEG_OPTIMIZE      // 启用优化 (0/1)
IMWRITE_JPEG_RST_INTERVAL  // 重启标记间隔
IMWRITE_JPEG_LUMA_QUALITY  // 单独亮度质量 (0-100)

// PNG 参数
IMWRITE_PNG_COMPRESSION    // 0-9 (默认 3)
IMWRITE_PNG_STRATEGY        // 策略: DEFAULT, FILTERED, Huffman, RLE
IMWRITE_PNG_BILEVEL         // 二值输出 (0/1)

// TIFF 参数
IMWRITE_TIFF_COMPRESSION   // 压缩方案
IMWRITE_TIFF_PREDICTOR      // 预测器 (0 无, 1 水平)
```

---

## 4.实现分析

### 4.1 图像加载流程

```cpp
// modules/imgcodecs/src/loadsave.cpp
Mat imread(const String& filename, int flags) {
    // 1. 根据文件扩展名创建图像解码器
    ImageDecoder decoder = findDecoder(filename);

    // 2. 打开文件并读取头部
    decoder->load(filename);

    // 3. 根据 flags 设置读取参数
    decoder->setScale(1);  // 或从 flags

    // 4. 读取图像数据
    Mat result;
    decoder->read(result);

    // 5. 如需要, 应用颜色转换
    if (flags == IMREAD_GRAYSCALE)
        cvtColor(result, result, COLOR_BGR2GRAY);
    else if (flags == IMREAD_UNCHANGED)
        ;  // 保持原样

    return result;
}
```

### 4.2 格式检测优先级

```
1. 首先检查文件扩展名 (快速路径)
2. 如果启用了 GDAL, 尝试 GDAL
3. 检查魔术字节 (JPEG: FF D8, PNG: 89 50 4E 47...)
4. 按顺序尝试每个编解码器
```

---

## 5.代码示例

### 5.1 基本加载和保存

```cpp
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // 加载图像
    Mat img = imread("photo.jpg", IMREAD_COLOR);
    if (img.empty()) {
        printf("错误: 无法加载图像\n");
        return -1;
    }

    // 以不同格式保存
    imwrite("output.png", img);
    imwrite("output.jpg", img, {IMWRITE_JPEG_QUALITY, 90});
    imwrite("output_gray.png", img, {IMWRITE_PNG_COMPRESSION, 6});

    // 显示
    imshow("图像", img);
    waitKey(0);

    return 0;
}
```

### 5.2 批量图像处理

```cpp
// 批量处理图像
void processImages(const vector<string>& inputFiles, const string& outputDir) {
    for (const auto& inputFile : inputFiles) {
        Mat img = imread(inputFile, IMREAD_COLOR);
        if (img.empty()) continue;

        // 处理
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // 保存并设置质量
        string outputFile = outputDir + "/" + getFilename(inputFile);
        imwrite(outputFile, gray, {IMWRITE_JPEG_QUALITY, 85});
    }
}
```

### 5.3 加载特定位深度的图像

```cpp
// 加载 16 位灰度 PNG
Mat img16 = imread("16bit.png", IMREAD_ANYDEPTH | IMREAD_GRAYSCALE);
if (img16.depth() == CV_16U) {
    // 归一化到 8 位以便显示
    Mat normalized;
    img16.convertTo(normalized, CV_8U, 255.0/65535.0);
    imshow("归一化", normalized);
}

// 加载 16 位彩色 PNG
Mat img16Color = imread("16bit_color.png", IMREAD_ANYDEPTH | IMREAD_COLOR);
```

---

## 6.练习题

### 入门级
1. 加载图像, 转换为灰度图, 并保存
2. 加载带透明度的 PNG 并保留 alpha 通道
3. 以不同 JPEG 质量级别保存图像 (50, 75, 95)

### 中级
4. 实现多个图像的批量转换器
5. 加载 16 位图像, 归一化并显示
6. 通过魔术字节检测图像格式 (不依赖扩展名)

### 高级
7. 实现带特定质量设置的 WebP 加载
8. 创建自定义图像加载器插件

---

## 7.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [imgcodecs 模块](https://docs.opencv.org/4.14.0/d4/da4/group__imgcodecs.html) |
| imread 参考 | [imread](https://docs.opencv.org/4.14.0/d4/da4/group__imgcodecs.html#ga288baf1028cd81d463b89f1b1919efba) |
| imwrite 参考 | [imwrite](https://docs.opencv.org/4.14.0/d4/da4/group__imgcodecs.html#ga49961cbfd440e1657c5fc4bd3f172550) |
| 图像编解码器 | [OpenCV imgcodecs](https://docs.opencv.org/4.14.0/d2/de6/group__imgcodecs.html) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 imgcodecs 模块文档 |