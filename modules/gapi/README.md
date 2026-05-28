# OpenCV GAPI 模块学习指南

> **模块**: gapi
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 5 - 专家

---

## 目录

1. [概述](./README.md#1概述)
2. [G-API 核心概念](./README.md#2g-api-核心概念)
3. [Kernel 开发](./README.md#3kernel-开发)
4. [异步执行](./README.md#4异步执行)
5. [代码示例](./README.md#5代码示例)
6. [练习题](./README.md#6练习题)
7. [参考资料](./README.md#7参考资料)

---

## 1.概述

**G-API** (Graph API) 是 OpenCV 的图计算引擎:

| 功能 | 描述 |
|---------|-------------|
| **图计算** | 将操作表达为有向无环图 (DAG) |
| **Kernel 开发** | 自定义可在 GPU/CPU 切换的操作 |
| **异步执行** | 流水线并行执行 |
| **编译优化** | 图级别优化 (kernel fusion, constant folding) |

**头文件**: `opencv2/gapi.hpp`, `opencv2/gapi/core.hpp`, `opencv2/gapi/imgproc.hpp`

---

## 2.G-API 核心概念

### 2.1 Graph 和 Operation

```cpp
// ============================================
// C++ G-API 基础
// ============================================
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

// 定义图输入
GMat in1, in2;

// 定义操作 (构建有向无环图)
GMat blurred = gaussianBlur(in1, Size(5, 5), 1.5);
GMat gray = RGB2Gray(in1);
GMat edges = Canny(gray, 50, 150);
GMat merged = absdiff(blurred, in2);

// 编译图
auto pkg = cv::gapi::compileArgs(in1, in2);

// 执行图
Mat result = pkg.run({inputImg1, inputImg2});
```

```python
# ============================================
# Python G-API 基础
# ============================================
import cv2
import numpy as np

# 注意: G-API Python 支持有限, 主要用于 C++

# 定义图输入
in1 = cv2.GMat()
in2 = cv2.GMat()

# 定义操作
blurred = cv2.gapi.gaussianBlur(in1, ksize=(5, 5), sigma=1.5)
gray = cv2.gapi.RGB2Gray(in1)
edges = cv2.gapi.Canny(gray, 50, 150)
merged = cv2.gapi.absdiff(blurred, in2)
```

### 2.2 数据类型

```cpp
// ============================================
// C++ G-API 数据类型
// ============================================
// GMat: 图像矩阵 (类似 cv::Mat)
GMat img1, img2;

// GScalar: 标量 (类似 cv::Scalar)
GScalar value;

// GArray: 数组 (类似 std::vector)
GArray<Point> points;
GArray<int> indices;

// GOpaque: 不透明类型
GOpaque<Size> sizeOpaque;

// 源操作 (图输入)
GMat in = cv::gapi::in("input1");
GScalar constVal = cv::gapi::constant(5.0);
```

```python
# ============================================
# Python G-API 数据类型
# ============================================
# 注意: Python G-API 主要通过 cv2 GAPI 模块

# GMat
img1 = cv2.GMat()
img2 = cv2.GMat()

# GScalar
value = cv2.GScalar()

# GArray
points = cv2.GArray()
```

### 2.3 常用 Operation

```cpp
// ============================================
// C++ 核心操作
// ============================================
// 基础算术
GMat add(GMat src1, GMat src2);           // 加法
GMat sub(GMat src1, GMat src2);           // 减法
GMat mul(GMat src, double alpha);         // 乘法
GMat div(GMat src, double alpha);           // 除法

// 逻辑运算
GMat bitwise_and(GMat src1, GMat src2);   // 按位与
GMat bitwise_not(GMat src);                 // 按位非

// 颜色转换
GMat RGB2Gray(GMat rgb);                   // RGB 转灰度
GMat BGR2HSV(GMat bgr);                    // BGR 转 HSV

// 滤波
GMat gaussianBlur(GMat src, Size ksize, double sigma);
GMat medianBlur(GMat src, int ksize);
GMat Sobel(GMat src, int ddepth, int dx, int dy, int ksize);

// 形态学
GMat erode(GMat src, Mat element);
GMat dilate(GMat src, Mat element);

// 阈值
GMat threshold(GMat src, double thresh, double maxval, int type);
```

```python
# ============================================
# Python 核心操作
# ============================================
# 基础算术
result = cv2.gapi.add(src1, src2)
result = cv2.gapi.sub(src1, src2)
result = cv2.gapi.mul(src, alpha)

# 颜色转换
gray = cv2.gapi.RGB2Gray(rgb)
hsv = cv2.gapi.BGR2HSV(bgr)

# 滤波
blurred = cv2.gapi.gaussianBlur(src, ksize=(5, 5), sigma=1.5)

# 阈值
thresh = cv2.gapi.threshold(src, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
```

---

## 3.Kernel 开发

### 3.1 自定义 Kernel

```cpp
// ============================================
// C++ 自定义 Kernel
// ============================================
// 定义 Kernel 接口 (类似抽象类)
GAPI_LIB OP PolymorphicIdentity {
    static GMat    on(GMat in);      // GMat 输入 → GMat 输出
    static GScalar on(GScalar in);  // GScalar 输入 → GScalar 输出
};

// 实现 Kernel
struct PolymorphicIdentityImpl {
    static GMat on(GMat in) {
        // 调用 OpenCV 实际操作
        return gapi::identity(in);
    }
};

// 注册 Kernel
GAPI_REGISTER_KERNEL(PolymorphicIdentity, PolymorphicIdentityImpl);

// 使用
GMat result = gapi::identity(input);
```

### 3.2 完整 Kernel 示例

```cpp
// ============================================
// C++ 完整自定义 Kernel
// ============================================
// 定义 Kernel 接口
struct MyCustomKernel {
    static const char* id() { return "org.opencv.custom.MyCustomKernel"; }

    static GMat on(GMat src, double factor) {
        // 创建 Kernel API (私有实现)
        GMatResult out;
        GCall::setup().passOn(googd::Arg<GMat>::define(), googd::Arg<double>::define());
        return out;
    }
};

// 实现 Kernel
struct MyCustomKernelImpl {
    static cv::GMat on(cv::GMat src, double factor) {
        // 获取实际 cv::Mat
        cv::Mat in_mat = cv::gapi::access<cv::GMat::Access::R>(src);

        // 执行操作
        cv::Mat out_mat;
        cv::multiply(in_mat, factor, out_mat);

        return out_mat;
    }
};

// 注册
GAPI_REGISTER_KERNEL(MyCustomKernel, MyCustomKernelImpl);

// 使用
GMat result = MyCustomKernel::on(input, 2.0);
```

### 3.3 GPU/CPU 后端切换

```cpp
// ============================================
// C++ 后端切换
// ============================================
// 定义 Kernel 时指定后端
GAPI_REGISTER_KERNEL(OCVKernel, MyOpenCVKernel);
GAPI_REGISTER_KERNEL(OPENCLKernel, MyOpenCLKernel);

// 使用
GMat input, output;

// CPU 执行
auto pkg_cpu = cv::gapi::compileArgs(output, cv::compile_args(cv::gapi::use_only<OCVKernel>()));
Mat result_cpu = pkg_cpu.run({input});

// GPU 执行
auto pkg_gpu = cv::gapi::compileArgs(output, cv::compile_args(cv::gapi::use_only<OPENCLKernel>()));
Mat result_gpu = pkg_gpu.run({input});
```

---

## 4.异步执行

### 4.1 异步图执行

```cpp
// ============================================
// C++ 异步执行
// ============================================
// 创建异步图
cv::gapi::GAsyncNode node = cv::gapi::async([&]() {
    // 定义图
    GMat in = cv::gapi::in("input");
    GMat out = gaussianBlur(in, Size(5, 5), 1.5);

    // 编译并启动
    return cv::gapi::compileAndRun(out, cv::gapi::use_only("input"));
});

// 设置输入
node.setInput("input", inputImage);

// 获取异步执行句柄
cv::gapi::GAsyncHandle handle = node.executeAsync();

// ... 做其他事情 ...

// 等待结果
Mat result = handle.getResult();
```

### 4.2 流水线并行

```cpp
// ============================================
// C++ 流水线并行
// ============================================
// 创建流水线
cv::gapi::pipeline pipeline;

// 添加阶段
pipeline.addStage([](GMat in) {
    return gaussianBlur(in, Size(5, 5), 1.5);
});

pipeline.addStage([](GMat in) {
    return Canny(in, 50, 150);
});

// 执行
Mat result = pipeline.run(inputImage);
```

---

## 5.代码示例

### 5.1 基础图构建

```cpp
// ============================================
// C++ 基础图构建
// ============================================
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

int main() {
    // 定义图输入
    GMat input = cv::gapi::in("input");

    // 构建处理图
    GMat gray = RGB2Gray(input);
    GMat blurred = gaussianBlur(gray, Size(5, 5), 1.5);
    GMat edges = Canny(blurred, 50, 150);

    // 编译图
    auto pkg = cv::gapi::compileArgs(edges, input);

    // 读取输入并执行
    Mat img = imread("input.jpg");
    Mat result = pkg.run({img})[0];

    imshow("边缘检测", result);
    waitKey(0);

    return 0;
}
```

```python
# ============================================
# Python 基础图构建 (有限支持)
# ============================================
import cv2
import numpy as np

# 注意: Python G-API 主要用于定义图结构
# 实际执行通常需要 C++ 后端

# 简单的 G-API 操作 (Python)
gray = cv2.gapi.RGB2Gray(img)
blurred = cv2.gapi.gaussianBlur(gray, ksize=(5, 5), sigma=1.5)
edges = cv2.gapi.Canny(blurred, 50, 150)
```

### 5.2 自定义 Kernel 开发

```cpp
// ============================================
// C++ 自定义 Kernel 完整示例
// ============================================
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>

// 定义自定义 Kernel
struct CustomScaleKernel {
    static const char* id() { return "org.opencv.custom.ScaleKernel"; }

    static GMat on(GMat src, double scale) {
        GAPI_LOG_INFO("CustomScaleKernel::on called");
        GMatResult out;
        return out;
    }
};

// 实现
struct CustomScaleKernelImpl {
    static cv::GMat on(cv::GMat src, double scale) {
        cv::Mat in_mat = cv::gapi::access<cv::GMat::Access::R>(src);
        cv::Mat out_mat;
        cv::convertScaleAbs(in_mat, out_mat, scale, 0);
        return out_mat;
    }
};

// 注册
GAPI_REGISTER_KERNEL(CustomScaleKernel, CustomScaleKernelImpl);

int main() {
    // 使用自定义 Kernel
    GMat input = cv::gapi::in("input");
    GMat output = CustomScaleKernel::on(input, 1.5);

    // 编译和执行
    auto pkg = cv::gapi::compileArgs(output, input);

    Mat img = imread("input.jpg");
    Mat result = pkg.run({img})[0];

    return 0;
}
```

### 5.3 后端切换示例

```cpp
// ============================================
// C++ 后端切换
// ============================================
int main() {
    GMat input = cv::gapi::in("input");

    // 定义图 (使用标准 OpenCV kernels)
    GMat blurred = gaussianBlur(input, Size(5, 5), 1.0);
    GMat gray = RGB2Gray(blurred);

    // 编译为 CPU 后端
    auto pkg_cpu = cv::gapi::compileArgs(gray,
        cv::compile_args(cv::gapi::use_only<cv::gapi::CPU>()));

    // 编译为 OpenCL 后端
    auto pkg_gpu = cv::gapi::compileArgs(gray,
        cv::compile_args(cv::gapi::use_only<cv::gapi::OCL>()));

    Mat img = imread("input.jpg");

    // CPU 执行
    Mat result_cpu = pkg_cpu.run({img})[0];

    // GPU 执行
    Mat result_gpu = pkg_gpu.run({img})[0];

    return 0;
}
```

### 5.4 复杂流水线

```cpp
// ============================================
// C++ 复杂流水线
// ============================================
Mat complexPipeline(const Mat& input) {
    // 输入
    GMat in = cv::gapi::in("input");

    // 并行分支
    GMat gray = RGB2Gray(in);
    GMat blur = gaussianBlur(in, Size(3, 3), 1.0);
    GMat edges = Canny(gray, 50, 150);

    // 合并结果
    GMat merged = hconcat(blur, gray);
    GMat final = resize(merged, Size(640, 480));

    // 编译执行
    auto pkg = cv::gapi::compileArgs(final, in);

    return pkg.run({input})[0];
}
```

---

## 6.练习题

### 入门级
1. 使用 G-API 重构简单的图像处理流水线 (C++)
2. 理解 GMat/GScalar/GArray 的区别
3. 使用自定义 Kernel 实现图像缩放

### 中级
4. 实现自定义双边滤波 Kernel
5. 比较 CPU 和 OpenCL 后端的性能
6. 实现异步图像处理流水线

### 高级
7. 实现自定义图优化 pass
8. 实现多后端 Kernel (CPU/GPU/Vulkan)
9. 实现端到端的 G-API 图像处理应用

### 挑战题
10. 实现自定义图级别优化 (kernel fusion)
11. 实现 G-API 与 TensorFlow 的互操作

---

## 7.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [gapi 模块](https://docs.opencv.org/4.14.0/da/d9e/group__gapi.html) |
| G-API 核心 | [G-API Core](https://docs.opencv.org/4.14.0/d0/d5e/group__gapi__core.html) |
| G-API imgproc | [G-API Image Processing](https://docs.opencv.org/4.14.0/d3/d9e/group__gapi__imgproc.html) |
| G-API 教程 | [G-API Introduction](https://docs.opencv.org/4.14.0/d6/d2e/tutorial_gapi_intro.html) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 gapi 模块文档 (C++/Python 双语) |