# OpenCV Core 模块学习指南

> **模块**: core
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 1 - 入门

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

**core** 是 OpenCV 的基础模块，所有其他模块都依赖它。主要提供：

| 功能 | 描述 |
|---------|-------------|
| **Mat 数据结构** | 用于图像和矩阵的多维密集数组 |
| **基本数组操作** | 算术、逻辑和统计运算 |
| **XML/YAML 持久化** | 配置存储和读取 |
| **并行计算** | 多线程支持 |
| **硬件加速层** | SIMD、OpenCL、CUDA 支持 |

**头文件**: `opencv2/core.hpp`

---

## 2.核心数据结构

### 2.1 Mat - 多维密集数组

Mat 是 OpenCV 中最核心的数据结构，用于存储图像和矩阵。

```cpp
#include <opencv2/core.hpp>

// 创建图像
Mat img(480, 640, CV_8UC3);              // 480行 x 640列, 3通道 uint8_t
Mat img2(Size(640, 480), CV_8UC3);       // 使用 Size 的同样效果

// 从文件加载图像
Mat img = imread("photo.jpg", IMREAD_COLOR);

// 基本属性
img.rows;           // 行数 (高度)
img.cols;           // 列数 (宽度)
img.channels();      // 通道数 (BGR为3)
img.depth();         // 深度标识符 (CV_8U = 0)
img.type();          // 类型标识符 (CV_8UC3 = 16)
img.step;            // 每行字节数
img.data;           // 原始数据指针 (uchar*)
img.isContinuous(); // 内存是否连续
img.total();        // 元素总数 = rows × cols
img.elemSize();     // 每个元素的字节数
```

### 2.2 Mat 内存模型

```
内存布局 (3x4 BGR 图像):

行主序连续存储:
data[0]  data[1]  data[2]  data[3]  ...  B G R B G R B G R ...
            ↓
Step = cols × channels × elemSize() = 640 × 3 × 1 = 1920 字节

ROI 提取 (不复制数据, 共享内存):
Mat roi = img(Rect(100, 100, 200, 200));  // 共享数据指针
```

### 2.3 Mat 创建方法对比

```cpp
// 方法1: 构造函数初始化
Mat img1(480, 640, CV_8UC3, Scalar(0, 0, 255));  // 红色背景

// 方法2: create (延迟分配)
Mat img2;
img2.create(480, 640, CV_8UC3);

// 方法3: 工厂方法
Mat img3 = Mat::zeros(480, 640, CV_8UC3);  // 全零
Mat img4 = Mat::ones(Size(640, 480), CV_32FC1);  // 全一
Mat img5 = Mat::eye(3, 3, CV_32FC1);  // 单位矩阵

// 方法4: 逗号初始化器
Mat img6 = (Mat_<float>(2, 2) << 1, 2, 3, 4);

// 方法5: 模板类 Mat_
Mat_<uchar> img7(480, 640);
img7(100, 100) = 255;  // 使用 () 而不是 at<>
```

### 2.4 输入输出数组类型

OpenCV 使用 `InputArray`/`OutputArray` 实现灵活的函数参数:

```cpp
// 支持的类型: Mat, Matx, vector, UMat 等
void process(InputArray src, OutputArray dst);

// 调用示例
Mat m;
vector<Point2f> vec;
Matx33f matx;
UMat um;

process(m, result);    // Mat
process(vec, result); // std::vector
process(matx, result); // Matx
process(um, result);  // UMat (GPU)
```

### 2.5 数据类型深度

| 深度 ID | 类型 | 范围 |
|----------|------|-------|
| CV_8U | uint8_t | 0-255 |
| CV_8S | int8_t | -128 到 127 |
| CV_16U | uint16_t | 0-65535 |
| CV_16S | int16_t | -32768 到 32767 |
| CV_32S | int32_t | 约 ±21亿 |
| CV_32F | float | 归一化范围 -1.0 到 1.0 |
| CV_64F | double | 全范围 |
| CV_16F | half float | FP16 半精度 |

**类型公式**: `CV_<深度>C<通道数>`

```cpp
CV_8UC3  = 0 + 8×3  = 16   // 3通道 uint8_t (如 BGR)
CV_32FC1 = 5 + 8×1  = 13   // 1通道 float
```

### 2.6 UMat (GPU 矩阵)

UMat 是 OpenCL/GPU 后端的矩阵类型，接口与 Mat 兼容:

```cpp
// UMat 用于 GPU 加速操作
UMat img_uma = imread("photo.jpg", IMREAD_COLOR).getUMat(ACCESS_RW);

// OpenCL 自动调度
GaussianBlur(img_uma, blurred, Size(5, 5), 1.5);

// 下载到 CPU
Mat result = blurred.getMat(ACCESS_READ);
```

### 2.7 Matx (小型矩阵模板)

Matx 用于小型固定尺寸矩阵，性能更高:

```cpp
// 2x3 浮点矩阵
Matx23f M(1.f, 2.f, 3.f,
           4.f, 5.f, 6.f);

// 3x3 单位矩阵
Matx33f I = Matx33f::eye();

// 访问元素
float val = M(0, 1);  // 第0行第1列 = 2.f
```

---

## 3.核心API

### 3.1 算术运算

```cpp
// 加法: dst = src1 + src2 (饱和截断)
Mat dst = src1 + src2;
add(src1, src2, dst);
add(src1, src2, dst, mask);  // 带掩码

// 加权加法 (混合): dst = α·src1 + β·src2 + γ
addWeighted(src1, 0.5, src2, 0.5, 0, dst);

// 减法
subtract(src1, src2, dst);

// 逐元素乘法
multiply(src1, src2, dst, scale=1);

// 逐元素除法
divide(src1, src2, dst, scale=1);

// 矩阵乘法
gemm(src1, src2, alpha, src3, beta, dst);

// 矩阵点乘
Mat dotResult = src1.t() * src2;

// 比较
compare(src1, src2, dst, CMP_EQ);  // dst = (src1 == src2)
```

### 3.2 逻辑运算

```cpp
bitwise_and(src1, src2, dst);    // 按位与
bitwise_or(src1, src2, dst);     // 按位或
bitwise_xor(src1, src2, dst);   // 按位异或
bitwise_not(src, dst);           // 按位非
inRange(src, lower, upper, dst); // src 在 [lower, upper] 范围内返回 255
```

### 3.3 统计运算

```cpp
Scalar sum = sum(src);              // 所有通道元素求和
Scalar mean = mean(src, mask);     // 均值

// 均值和标准差
Mat mean, stddev;
meanStdDev(src, mean, stddev, mask);

// 最值及位置
double minVal, maxVal;
Point minLoc, maxLoc;
minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask);

int nz = countNonZero(src);        // 非零元素计数
double normL2 = norm(src, NORM_L2); // L2 范数
double normL1 = norm(src, NORM_L1); // L1 范数
double normInf = norm(src, NORM_INF); // 无穷范数
double normDiffL2 = norm(src1, src2, NORM_L2); // 两矩阵差异 L2
```

### 3.4 数组变换

```cpp
// 分割/合并通道
vector<Mat> channels;
split(src, channels);   // 分割为单通道
merge(channels, dst);   // 合并

// 通道混合 (高级通道操作)
Mat bgr, alpha;
int fromTo[] = {0,2, 1,1, 2,0, 3,3};  // BGR → GRB + alpha
mixChannels(&src, 1, &bgr, 1, fromTo, 4);

// 翻转
flip(src, dst, 0);   // 0: 垂直翻转上下
flip(src, dst, 1);   // >0: 水平翻转左右
flip(src, dst, -1);  // <0: 同时翻转

// 转置
transpose(src, dst);

// 旋转 90°
rotate(src, dst, ROTATE_90_CLOCKWISE);      // 顺时针90°
rotate(src, dst, ROTATE_180);                 // 180°
rotate(src, dst, ROTATE_90_COUNTERCLOCKWISE); // 逆时针90°

// 水平拼接: [src1 src2]
hconcat(src1, src2, dst);

// 垂直拼接: [src1; src2]
vconcat(src1, src2, dst);

// 重复/复制
repeat(src, 2, 3, dst);  // 复制 2×3 次 (2行3列)
```

### 3.5 类型转换

```cpp
// 数据类型转换
src.convertTo(dst, CV_32FC1, alpha=1, beta=0);
// 参数: 目标深度, scale(α), offset(β)
// 公式: dst = src * α + β

// 饱和截断转换为 uchar
convertScaleAbs(src, dst);
// 公式: dst = saturate_cast<uchar>(|src * α + β|)

// 归一化
normalize(src, dst, 1.0, 0, NORM_L2);  // L2 归一化, 结果范数为1
normalize(src, dst, 0, 255, NORM_MINMAX); // 归一化到 [0, 255]
```

### 3.6 查找表 (LUT)

```cpp
// 创建查找表
Mat lut(256, 1, CV_8U);
for (int i = 0; i < 256; i++)
    lut.at<uchar>(i) = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255);

// 应用 LUT (O(1) 复杂度的像素变换)
LUT(src, lut, dst);
```

### 3.7 矩阵运算

```cpp
// 转置
Mat T = src.t();

// 逆矩阵
Mat inv;
invert(src, inv, DECOMP_LU);  // LU 分解 (方阵)
invert(src, inv, DECOMP_SVD); // SVD 分解 (通用)

// 行列式
double d = determinant(src);

// 迹 (对角元素和)
double t = trace(src)[0];

// 矩阵范数
double n = norm(src, NORM_L2);

// 重组矩阵形状
Mat reshaped = src.reshape(src.channels(), newRows);
```

### 3.8 复制与填充

```cpp
// 复制
Mat copy = src.clone();           // 深拷贝
src.copyTo(dst);                  // 深拷贝 (可带掩码)
src.copyTo(dst, mask);           // 仅复制掩码位置

// 填充
Mat filled = Mat(100, 100, CV_8UC3, Scalar(255, 0, 0)); // 蓝色

// ROI 复制
Mat roi = src(Rect(0, 0, 50, 50));
roi.copyTo(dst);  // 复制 ROI 到 dst
```

---

## 4.实现分析

### 4.1 行连续性

```cpp
// 判断: isContinuous()
Mat img(480, 640, CV_8UC3);

// 内存布局:
// 连续: data 包含 img.rows × img.step 个字节
// 不连续: 存在额外的行填充 (对齐到4/8字节边界)

// 快速遍历 (行连续时):
uchar* p = img.data;
for (size_t i = 0; i < img.total() * img.channels(); i++) {
    p[i] = processing(p[i]);
}

// 非连续时需要双重循环:
for (int r = 0; r < img.rows; r++) {
    uchar* row = img.ptr(r);
    for (int c = 0; c < img.cols; c++) {
        // 处理每个像素
    }
}
```

### 4.2 ROI (感兴趣区域)

```cpp
Mat img = imread("photo.jpg");

// 提取 ROI (不复制数据, 共享内存)
Rect roi(100, 100, 200, 200);
Mat region = img(roi);

// 修改 region 会影响 img
region = Scalar(0, 0, 0);  // 左上角区域变黑

// 需要时复制 ROI
Mat regionCopy = img(roi).clone();

// 避免 ROI 引用失效: 确保 img 生命周期大于 region
```

### 4.3 Scalar 类型

```cpp
// 4元素向量, 用于表示多通道像素值
Scalar s(255);              // 单通道: (255)
Scalar s(255, 128, 64);     // 3通道: B=255, G=128, R=64
Scalar s(255, 128, 64, 255); // 4通道: B=255, G=128, R=64, A=255

// 在 BGR 图像中使用
img = Scalar(0, 0, 255);   // 红色

// 算术运算
Scalar s1(100), s2(50);
Scalar s3 = s1 + s2;  // (150, 150, 150, 150)
```

### 4.4 saturate_cast - 饱和转换

```cpp
// 防止溢出
uchar val = saturate_cast<uchar>(300.0);  // 返回 255 (截断)
uchar val = saturate_cast<uchar>(-10.0);   // 返回 0 (截断)

// 所有算术运算自动使用饱和运算
Mat src1(1, 1, CV_8U, Scalar(250));
Mat src2(1, 1, CV_8U, Scalar(10));
Mat dst;
add(src1, src2, dst);  // dst = 255 (不是 260)

// saturate_cast 模板
saturate_cast<uchar>(x)   // x -> uchar
saturate_cast<char>(x)    // x -> int8
saturate_cast<ushort>(x)  // x -> uint16
saturate_cast<short>(x)   // x -> int16
saturate_cast<int>(x)      // x -> int32
saturate_cast<float>(x)   // x -> float
saturate_cast<double>(x)   // x -> double
```

### 4.5 Mat 内部结构

```cpp
// Mat 内部结构 (mat.hpp)
class Mat {
public:
    // 维度
    int dims;
    // 行数 (2D时)
    int rows;
    // 列数 (2D时)
    int cols;
    // 数据类型
    int type;     // CV_8UC3 等
    int depth;    // CV_8U 等
    int channels; // 通道数

    // 数据指针
    uchar* data;          // 数据起始指针
    uchar* datastart;    // 数据起始位置 (含 padding)
    uchar* dataend;      // 数据结束位置
    uchar* datalimit;    // 数据限制 (allocated end)

    // 步长 (每行字节数)
    size_t step;

    // 引用计数
    int* refcount;

    // 分配器
    MatAllocator* allocator;
};
```

### 4.6 源码参考

| 文件 | 描述 |
|------|-------------|
| `modules/core/src/mat.cpp` | Mat 实现 (创建、销毁、访问) |
| `modules/core/src/arithm.cpp` | 算术运算 (add/subtract/multiply/divide) |
| `modules/core/src/convert.cpp` | 类型转换 (convertTo) |
| `modules/core/src/copy.cpp` | 复制和 ROI 操作 |
| `modules/core/src/stat.cpp` | 统计运算 (sum/mean/norm) |
| `modules/core/src/matx.cpp` | Matx 小矩阵实现 |

---

## 5.代码示例

### 5.1 基础图像操作

```cpp
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // 读取图像
    Mat img = imread("photo.jpg", IMREAD_COLOR);

    // 转换为灰度图
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 创建反色图像
    Mat inverted;
    bitwise_not(img, inverted);

    // 调整亮度和对比度
    // alpha=1.5 (对比度), beta=50 (亮度)
    Mat adjusted;
    img.convertTo(adjusted, -1, 1.5, 50);

    // 显示
    imshow("原图", img);
    imshow("灰度", gray);
    imshow("反色", inverted);
    imshow("调整后", adjusted);
    waitKey(0);

    return 0;
}
```

### 5.2 像素访问方法对比

```cpp
Mat img(480, 640, CV_8UC3);

// 方法1: at<> (类型安全, 编译时检查, 推荐)
img.at<Vec3b>(y, x)[0] = 255;  // B 通道 (blue)
img.at<Vec3b>(y, x)[1] = 0;    // G 通道 (green)
img.at<Vec3b>(y, x)[2] = 0;    // R 通道 (red)

// 灰度图
Mat gray(480, 640, CV_8UC1);
gray.at<uchar>(y, x) = 128;

// 方法2: ptr<> (指针访问, 行首, 常用)
uchar* row = img.ptr<uchar>(y);
row[x * 3] = 255;              // B
row[x * 3 + 1] = 0;           // G
row[x * 3 + 2] = 0;           // R

// 方法3: data + 偏移 (最快, 用于性能关键代码)
uchar* p = img.data + y * img.step + x * img.elemSize();
p[0] = 255;                   // B
p[1] = 0;                     // G
p[2] = 0;                     // R

// 方法4: MatIterator_ (遍历所有像素)
for (auto it = img.begin<Vec3b>(); it != img.end<Vec3b>(); ++it) {
    (*it)[0] = 255;  // B
    (*it)[1] = 0;    // G
    (*it)[2] = 0;    // R
}

// 方法5: 整图像素访问 (Mat_)
Mat_<Vec3b> img_(img);  // 创建视图
img_(y, x) = Vec3b(255, 0, 0);
```

### 5.3 批量图像处理

```cpp
// 批量处理图像并计算统计数据
void batchProcess(const vector<string>& filenames) {
    vector<Mat> images;
    for (const auto& filename : filenames) {
        Mat img = imread(filename);
        if (img.empty()) continue;

        Mat processed;
        cvtColor(img, processed, COLOR_BGR2GRAY);
        normalize(processed, processed, 0, 255, NORM_MINMAX);
        images.push_back(processed);
    }

    // 计算平均图像
    Mat avgImg = Mat::zeros(images[0].size(), CV_32FC1);
    for (auto& img : images) {
        Mat floatImg;
        img.convertTo(floatImg, CV_32FC1);
        avgImg += floatImg;
    }
    avgImg /= images.size();

    // 计算每个像素的标准差
    Mat stdDevImg(images[0].size(), CV_32FC1);
    for (auto& img : images) {
        Mat floatImg;
        img.convertTo(floatImg, CV_32FC1);
        Mat diff;
        absdiff(floatImg, avgImg, diff);
        Mat diff2;
        multiply(diff, diff, diff2);
        stdDevImg += diff2;
    }
    sqrt(stdDevImg / images.size(), stdDevImg);
}
```

### 5.4 伽马校正 (使用 LUT)

```cpp
// 伽马校正函数
// gamma < 1: 提高暗部亮度 (变亮)
// gamma > 1: 提高亮部亮度 (变暗)
Mat gammaCorrect(const Mat& src, double gamma) {
    // 创建查找表
    Mat lut(256, 1, CV_8U);
    for (int i = 0; i < 256; i++)
        lut.at<uchar>(i) = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255);

    Mat dst;
    LUT(src, lut, dst);
    return dst;
}

// 使用示例
int main() {
    Mat img = imread("dark.jpg");

    Mat bright = gammaCorrect(img, 0.5);   // gamma=0.5 提亮
    Mat dark = gammaCorrect(img, 2.0);    // gamma=2.0 变暗

    imshow("原图", img);
    imshow("提亮 (γ=0.5)", bright);
    imshow("变暗 (γ=2.0)", dark);
    waitKey(0);
}
```

### 5.5 图像混合与融合

```cpp
// 线性图像混合
void blendImages(const Mat& img1, const Mat& img2, double alpha, Mat& dst) {
    // 确保类型一致
    Mat m1, m2;
    img1.convertTo(m1, CV_32FC3);
    img2.convertTo(m2, CV_32FC3);

    // 混合公式: dst = α*src1 + (1-α)*src2
    Mat temp;
    multiply(m1, Scalar(alpha, alpha, alpha), temp);
    dst = temp + m2 * (1 - alpha);

    // 截断并转换回 8UC3
    dst.convertTo(dst, CV_8UC3);
}

// 高斯金字塔混合
void pyramidBlend(const Mat& img1, const Mat& img2, int levels, Mat& dst) {
    vector<Mat> g1, g2, gA, gB, pA, pB;

    // 构建高斯金字塔
    g1.push_back(img1);
    g2.push_back(img2);
    for (int i = 1; i < levels; i++) {
        Mat down1, down2;
        pyrDown(g1[i-1], down1);
        pyrDown(g2[i-1], down2);
        g1.push_back(down1);
        g2.push_back(down2);
    }
}
```

### 5.6 矩阵运算示例

```cpp
// 矩阵基本运算
void matrixOperations() {
    // 创建矩阵
    Mat A = (Mat_<float>(2, 2) << 1, 2, 3, 4);
    Mat B = Mat::eye(2, 2, CV_32FC1);
    Mat C = Mat::ones(2, 2, CV_32FC1);

    // 加减乘除
    Mat result;

    // 逐元素乘法 (对应位置相乘)
    multiply(A, B, result);

    // 矩阵乘法 (线性代数乘)
    gemm(A, B, 1.0, Mat(), 0, result);  // result = A * B

    // 转置
    Mat AT = A.t();

    // 逆矩阵
    Mat Ainv;
    invert(A, Ainv, DECOMP_LU);

    // 特征值和特征向量 (对称矩阵)
    Mat eigenvalues, eigenvectors;
    eigen(A.t() * A, eigenvalues, eigenvectors);

    // SVD 分解
    Mat w, u, vt;
    SVD::compute(A, w, u, vt);
}
```

---

## 6.练习题

### 入门级
1. 创建一个 100×100 的红色图像并保存
2. 读取图像, 提取 ROI 区域并显示
3. 将彩色图像转换为灰度并反色
4. 使用三种方式遍历图像并计算像素总和

### 中级
5. 实现自定义滤波 (不使用 filter2D) - 锐化/模糊
6. 实现两幅图像的线性混合
7. 使用 LUT 实现颜色表变换 (胶片效果)
8. 实现图像归一化到 [0, 255] 范围

### 高级
9. 实现 Otsu 自动阈值分割 (不使用内置函数)
10. 实现卷积操作的自定义 SIMD 优化版本
11. 实现矩阵的 SVD 分解并用于图像压缩
12. 实现并行图像处理流水线 (多线程)

### 挑战题
13. 从零实现一个简单的图像卷积神经网络forward (仅 Mat 操作)
14. 实现图像拼接中的特征匹配和融合

---

## 7.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [docs.opencv.org/4.14.0/](https://docs.opencv.org/4.14.0/) |
| Mat 类参考 | [Class Mat](https://docs.opencv.org/4.14.0/d3/d63/classcv_1_1Mat.html) |
| 核心操作教程 | [Core Operations](https://docs.opencv.org/4.14.0/d7/d16/tutorial_py_core_ops.html) |
| Matx 文档 | [Matx](https://docs.opencv.org/4.14.0/d9/d52/classcv_1_1Matx.html) |
| 源码 | [github.com/opencv/opencv](https://github.com/opencv/opencv) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 core 模块文档 |
| 2026-05-27 | 4.14.0-pre | 增强文档: 添加 Matx、UMat、矩阵运算、更多示例 |