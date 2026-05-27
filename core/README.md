# OpenCV Core 模块学习指南

> Stage 1 - 入门级别核心模块

## 模块概述

**Core** 是 OpenCV 的基础模块，所有其他模块都依赖它。主要提供：

- **Mat 数据结构**：图像和矩阵的核心数据结构
- **基本数组操作**：算术、逻辑、统计运算
- **XML/YAML 持久化**：配置存储与读取
- **并行计算**：多线程支持
- **硬件加速层**：SIMD、OpenCL、CUDA 支持

## 核心数据结构

### 1. Mat - 多维密集数组

```cpp
#include <opencv2/core.hpp>

// 创建图像
Mat img(480, 640, CV_8UC3);              // 480x640 3通道 uint8_t
Mat img2(Size(640, 480), CV_8UC3);       // 同样效果

// 从文件加载
Mat img = imread("photo.jpg", IMREAD_COLOR);

// 基本属性
img.rows;        // 行数
img.cols;        // 列数
img.channels();  // 通道数 (3 for BGR)
img.depth();     // 深度 (CV_8U = 0)
img.type();      // 类型 (CV_8UC3 = 16)
img.step;        // 行字节数
img.data;        // 数据指针 (uchar*)
img.isContinuous();  // 是否连续内存
```

### 2. Mat 内存模型

```
图像内存布局 (3x4 BGR 图像):

行为主连续存储:
data[0]  data[1]  data[2]  data[3]  ...  B G R B G R B G R ...
           ↓
Step = cols * channels * elemSize() = 640 * 3 * 1 = 1920 bytes

ROI 提取 (不复制数据):
Mat roi = img(Rect(100, 100, 200, 200));  // 共享数据
```

### 3. Mat 创建方式对比

```cpp
// 方式1: 构造函数
Mat img1(480, 640, CV_8UC3, Scalar(0, 0, 255));  // 红色背景

// 方式2: create (延迟分配)
Mat img2;
img2.create(480, 640, CV_8UC3);

// 方式3: Mat::zeros / ones / eye
Mat img3 = Mat::zeros(480, 640, CV_8UC3);
Mat img4 = Mat::ones(Size(640, 480), CV_32FC1);
Mat img5 = Mat::eye(3, 3, CV_32FC1);

// 方式4: 逗号初始化
Mat img6 = (Mat_<float>(2, 2) << 1, 2, 3, 4);

// 方式5: 模板类 Mat_
Mat_<uchar> img7(480, 640);
img7(100, 100) = 255;  // 使用 () 而不是 at<>
```

### 4. 输入输出数组类型

OpenCV 使用 InputArray/OutputArray 作为函数参数，支持多种类型：

```cpp
// 可接受的类型
void process(InputArray src, OutputArray dst);

// 调用方式
Mat m;
vector<Point2f> vec;
Matx33f matx;
UMat um;

process(m, result);       // Mat
process(vec, result);     // std::vector
process(matx, result);     // Matx
process(um, result);       // UMat (GPU)
```

## 核心 API

### 算术运算

```cpp
// 加法: dst = src1 + src2  (饱和截断)
Mat dst = src1 + src2;
add(src1, src2, dst);

// 加权加法 (混合): dst = alpha*src1 + beta*src2 + gamma
addWeighted(src1, 0.5, src2, 0.5, 0, dst);

// 减法
subtract(src1, src2, dst);

// 乘法 (逐元素)
multiply(src1, src2, dst, scale=1);

// 点乘
Mat dotResult = src1.t() * src2;  // 矩阵点乘

// 比较
compare(src1, src2, dst, CMP_EQ);  // dst = (src1 == src2)
```

### 逻辑运算

```cpp
bitwise_and(src1, src2, dst);   // 逐位与
bitwise_or(src1, src2, dst);    // 逐位或
bitwise_xor(src1, src2, dst);   // 逐位异或
bitwise_not(src, dst);          // 逐位非
inRange(src, lower, upper, dst); // src 在 [lower, upper] 范围内
```

### 统计运算

```cpp
Scalar sum = sum(src);                    // 求和
Scalar mean = mean(src, mask);           // 均值
meanStdDev(src, mean, stddev, mask);     // 均值和标准差
double minVal, maxVal;
Point minLoc, maxLoc;
minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask);  // 最值
int nz = countNonZero(src);              // 非零计数
double normL2 = norm(src, NORM_L2);      // L2 范数
```

### 数组变换

```cpp
// 分割/合并通道
vector<Mat> channels;
split(src, channels);          // 分割为单通道
merge(channels, dst);          // 合并

// 通道混合
Mat bgr, alpha;
mixChannels(&src, 1, &bgr, 1, fromTo, 4);  // 自定义通道映射

// 翻转
flip(src, dst, 0);   // 0: 垂直翻转, >0: 水平, <0: 同时

// 转置
transpose(src, dst);

// 旋转 90°
rotate(src, dst, ROTATE_90_CLOCKWISE);

// 拼接
hconcat(src1, src2, dst);  // 水平
vconcat(src1, src2, dst);  // 垂直

// 重复/复制
repeat(src, 2, 3, dst);  // 复制 2x3 次
```

### 类型转换

```cpp
// 转换数据类型
src.convertTo(dst, CV_32FC1, alpha=1, beta=0);

// 饱和截断转换到 uchar
convertScaleAbs(src, dst);

// 归一化
normalize(src, dst, 1.0, 0, NORM_L2);  // L2 归一化
```

### 查找表 (LUT)

```cpp
// 创建查找表
Mat lut(256, 1, CV_8U);
for (int i = 0; i < 256; i++)
    lut.at<uchar>(i) = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255);

// 应用 LUT
LUT(src, lut, dst);
```

## 重要概念

### 1. 行连续性

```cpp
// 判断是否行连续 (isContinuous)
Mat img(480, 640, CV_8UC3);

// 内存布局:
// 连续: data 有 img.rows * img.step 个字节
// 不连续: 有额外的行间隔

// 快速遍历 (行连续时):
uchar* p = img.data;
for (size_t i = 0; i < img.total() * img.channels(); i++) {
    p[i] = processing(p[i]);
}
```

### 2. ROI (感兴趣区域)

```cpp
Mat img = imread("photo.jpg");

// 提取 ROI (不复制数据，共享内存)
Rect roi(100, 100, 200, 200);
Mat region = img(roi);

// 修改 region 会影响 img
region = Scalar(0, 0, 0);  // 左上角变黑

// 复制 ROI
Mat regionCopy = img(roi).clone();
```

### 3. Scalar 类型

```cpp
// 4 元素向量，用于表示像素值(多通道)
Scalar s(255);           // 单通道: (255)
Scalar s(255, 128, 64);  // 3通道: B=255, G=128, R=64

// 在 BGR 图像中使用
img = Scalar(0, 0, 255);  // 红色
```

### 4. 数据类型深度

```cpp
// CV_<depth>{C<channels>}
CV_8U   = 0   // uint8_t  (0-255)
CV_8S   = 1   // int8_t   (-128 to 127)
CV_16U  = 2   // uint16_t (0-65535)
CV_16S  = 3   // int16_t  (-32768 to 32767)
CV_32S  = 4   // int32_t
CV_32F  = 5   // float    (-1 to 1)
CV_64F  = 6   // double
CV_16F  = 7   // half float (FP16)

// 类型代码
CV_8UC3  = CV_8U + 8*3  = 16
CV_32FC1 = CV_32F + 8*1 = 5 + 8 = 13
```

### 5. saturate_cast 饱和转换

```cpp
// 防止溢出
uchar val = saturate_cast<uchar>(300.0);  // 返回 255
uchar val = saturate_cast<uchar>(-10.0); // 返回 0

// 所有算术运算自动使用饱和运算
Mat src1(1, 1, CV_8U, Scalar(250));
Mat src2(1, 1, CV_8U, Scalar(10));
Mat dst;
add(src1, src2, dst);  // dst = 255 (不是 260)
```

## 实用技巧

### 像素访问方法对比

```cpp
Mat img(480, 640, CV_8UC3);

// 方法1: at<> (类型安全，编译时检查)
img.at<Vec3b>(y, x)[0] = 255;  // B

// 方法2: ptr<> (指针访问，行首)
uchar* row = img.ptr<uchar>(y);
row[x * 3] = 255;     // B
row[x * 3 + 1] = 0;   // G

// 方法3: data + 偏移 (最快)
uchar* p = img.data + y * img.step + x * img.elemSize();
p[0] = 255;  // B

// 方法4: MatIterator_
for (auto it = img.begin<Vec3b>(); it != img.end<Vec3b>(); ++it) {
    (*it)[0] = 255;
}
```

### 内存效率建议

```cpp
// ❌ 不好: 多次复制
Mat img = imread("photo.jpg");
Mat tmp = img(Rect(0, 0, 100, 100));
Mat result = tmp.clone();
process(result);

// ✅ 好: 避免不必要的复制
Mat img = imread("photo.jpg");
Mat result;
img(Rect(0, 0, 100, 100)).copyTo(result);
process(result);

// ✅ 最好: 原地操作
Mat img = imread("photo.jpg");
process(img(Rect(0, 0, 100, 100)), result);
```

## 源码关键实现

### Mat 构造函数流程

```cpp
// modules/core/src/mat.cpp
Mat::Mat(int _rows, int _cols, int _type, void* _data, size_t _step)
    : flags(Mat_MAGIC_MASK + ((_type) & TYPE_MASK) + (((int64)_step << MAT_NTREE_STEP_SHIFT)),
      dims(0), rows(_rows), cols(_cols), data((uchar*)_data),
      datastart((uchar*)_data), dataend((uchar*)_data + _rows * _step),
      datalimit(dataend), allocator(0), usageFlags(0)
{
    if (_rows > 0 && _cols > 0) {
        step = _step;
        initInnerData(0);  // 分配头部信息
    }
}
```

### 算术运算实现示例

```cpp
// modules/core/src/arithm.cpp
void add(InputArray _src1, InputArray _src2, OutputArray _dst, InputArray _mask, int dtype) {
    // 1. 获取 Mat 指针
    Mat src1 = _src1.getMat(), src2 = _src2.getMat();

    // 2. 准备输出
    _dst.create(src1.size(), src1.type());
    Mat dst = _dst.getMat();

    // 3. 处理掩码
    if (_mask.empty()) {
        // 无掩码: 调用 SIMD 优化版本
        arithm_op(src1, src2, dst, noArray(), dtype, "add");
    } else {
        // 有掩码: 使用循环处理
        for (int i = 0; i < src1.rows; i++)
            for (int j = 0; j < src1.cols; j++)
                if (_mask.at<uchar>(i, j))
                    dst.at<uchar>(i, j) = saturate_cast<uchar>(
                        src1.at<uchar>(i, j) + src2.at<uchar>(i, j));
    }
}
```

## 示例代码

### 基础图像操作

```cpp
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // 读取图像
    Mat img = imread("photo.jpg", IMREAD_COLOR);

    // 创建灰度图
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 创建反色
    Mat inverted;
    bitwise_not(img, inverted);

    // 调整亮度和对比度
    Mat adjusted;
    img.convertTo(adjusted, -1, 1.5, 50);  // alpha=1.5, beta=50

    // 显示
    imshow("Original", img);
    imshow("Gray", gray);
    imshow("Inverted", inverted);
    waitKey(0);

    return 0;
}
```

### 批量图像处理

```cpp
#include <opencv2/core.hpp>

void batchProcess(const vector<string>& filenames) {
    vector<Mat> images;
    for (const auto& filename : filenames) {
        Mat img = imread(filename);
        if (img.empty()) continue;

        // 处理
        Mat processed;
        cvtColor(img, processed, COLOR_BGR2GRAY);
        normalize(processed, processed, 0, 255, NORM_MINMAX);

        images.push_back(processed);
    }

    // 计算平均值
    Mat avgImg = Mat::zeros(images[0].size(), CV_32FC1);
    for (auto& img : images) {
        Mat floatImg;
        img.convertTo(floatImg, CV_32FC1);
        avgImg += floatImg;
    }
    avgImg /= images.size();
}
```

## 练习题

### 入门题
1. 创建 100x100 的红色图像并保存
2. 读取图像，提取 ROI 区域并显示
3. 将彩色图像转换为灰度并反色

### 进阶题
4. 实现自定义滤波（不使用 filter2D）
5. 实现图像混合（线性混合两幅图）
6. 实现伽马校正（使用 LUT）

### 挑战题
7. 实现 Otsu 自动阈值分割（不使用 built-in）
8. 实现 Convolution 相关的优化版本

## 参考资源

- [官方文档](https://docs.opencv.org/4.x/group__core.html)
- [Mat 详解](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html)
- [OpenCV 教程 - 核心操作](https://docs.opencv.org/4.x/d7/d16/tutorial_py_core_ops.html)

## 下一步

- [学习 imgcodecs 模块](./imgcodecs/README.md) - 图像读写
- [学习 highgui 模块](./highgui/README.md) - 图像显示
- [学习 imgproc 模块](./imgproc/README.md) - 图像处理