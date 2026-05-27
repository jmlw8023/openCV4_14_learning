# OpenCV highgui 模块学习指南

> **模块**: highgui
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 1 - 入门

---

## 目录

1. [概述](./README.md#1概述)
2. [窗口管理](./README.md#2窗口管理)
3. [鼠标和滑动条事件](./README.md#3鼠标和滑动条事件)
4. [核心API](./README.md#4核心api)
5. [实现分析](./README.md#5实现分析)
6. [代码示例](./README.md#6代码示例)
7. [练习题](./README.md#7练习题)
8. [参考资料](./README.md#8参考资料)

---

## 1.概述

**highgui** 提供高级 GUI 功能:

| 功能 | 描述 |
|---------|-------------|
| **窗口管理** | 创建、显示和销毁窗口 |
| **图像显示** | 在窗口中显示图像, 支持自动缩放 |
| **鼠标事件** | 交互式鼠标回调处理 |
| **滑动条控制** | 创建滑块和进度条 |
| **键盘输入** | 等待并检测按键 |

**头文件**:
- `opencv2/highgui.hpp` (C API)
- `opencv2/highgui.hpp` (WASM/JS 兼容)

---

## 2.窗口管理

### 窗口标志

| 标志 | 描述 |
|------|-------------|
| `WINDOW_NORMAL` | 可调整大小的窗口 |
| `WINDOW_FULLSCREEN` | 全屏窗口 |
| `WINDOW_FREERATIO` | 自动适应图像而不扩展窗口 |
| `WINDOW_KEEPRATIO` | 保持图像宽高比 |
| `WINDOW_AUTOSIZE` | 固定大小以匹配图像 |

### 窗口操作

```cpp
#include <opencv2/highgui.hpp>

// 创建窗口 (默认 WINDOW_AUTOSIZE)
namedWindow("显示窗口", WINDOW_NORMAL);
namedWindow("显示窗口", WINDOW_KEEPRATIO | WINDOW_FREERATIO);

// 移动和调整窗口大小
moveWindow("显示窗口", x, y);      // 位置
resizeWindow("显示窗口", w, h);   // 大小

// 销毁窗口
destroyWindow("显示窗口");

// 销毁所有窗口
destroyAllWindows();
```

---

## 3.鼠标和滑动条事件

### 3.1 鼠标回调

```cpp
#include <opencv2/highgui.hpp>

// 鼠标事件回调函数
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    switch(event) {
        case EVENT_MOUSEMOVE:     // 鼠标移动
            break;
        case EVENT_LBUTTONDOWN:   // 左键按下
            break;
        case EVENT_RBUTTONDOWN:   // 右键按下
            break;
        case EVENT_MBUTTONDOWN:   // 中键按下
            break;
        case EVENT_LBUTTONUP:     // 左键释放
            break;
        case EVENT_LBUTTONDBLCLK:// 双击
            break;
    }
}

// 注册回调
setMouseCallback("窗口名称", mouseCallback, nullptr);
```

### 3.2 滑动条 (滑块)

```cpp
#include <opencv2/highgui.hpp>

int trackbarValue = 50;

// 创建滑动条
createTrackbar("亮度", "窗口名称", &trackbarValue, 100,
    [](int pos, void* userdata) {
        // 滑动条值改变时的回调
        printf("值: %d\n", pos);
    }, nullptr);

// 获取滑动条当前值
int currentValue = getTrackbarPos("亮度", "窗口名称");

// 程序化设置滑动条值
setTrackbarPos("亮度", "窗口名称", 75);
```

---

## 4.核心API

### 4.1 图像显示

```cpp
#include <opencv2/highgui.hpp>

// 在窗口中显示图像
imshow("窗口名称", img);

// 更新窗口 (用于 waitKey 后或使用 WINDOW_FREERATIO 时)
waitKey(1);

// 组合 waitKey 使用 (推荐)
int key = waitKey(0);  // 无限等待按键
int key = waitKey(30); // 等待 30ms (用于视频帧)

if (key == 'q' || key == 'Q')
    break;
if (key == 27)  // ESC 键
    break;
if (key == 's' || key == 'S')
    saveImage();
```

### 4.2 显示模式

```cpp
// WINDOW_NORMAL - 允许手动调整大小
namedWindow("查看器", WINDOW_NORMAL);
resizeWindow("查看器", 800, 600);
imshow("查看器", img);
waitKey(0);

// WINDOW_AUTOSIZE - 固定为图像大小 (默认)
namedWindow("查看器", WINDOW_AUTOSIZE);
imshow("查看器", img);
waitKey(0);

// WINDOW_FREERATIO + WINDOW_KEEPRATIO - 智能调整
namedWindow("查看器", WINDOW_FREERATIO | WINDOW_KEEPRATIO);
```

---

## 5.实现分析

### 5.1 事件循环架构

```
┌─────────────────────────────────────────┐
│           事件循环 (waitKey)            │
├─────────────────────────────────────────┤
│ 1. 轮询系统事件 (鼠标, 键盘)            │
│ 2. 更新滑动条位置                         │
│ 3. 触发回调                             │
│ 4. 如果启用则调用 OpenGL/glfw 渲染       │
│ 5. 返回键码或超时                        │
└─────────────────────────────────────────┘
```

### 5.2 平台后端

| 平台 | 后端 |
|---------|---------|
| Windows | Win32 API |
| Linux | GTK+ 3.0 或 Qt |
| macOS | Cocoa |
| Web/WASM | JavaScript Canvas |
| iOS | UIKit |
| Android | Java UI |

---

## 6.代码示例

### 6.1 基础图像查看器

```cpp
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("用法: %s <图像路径>\n", argv[0]);
        return -1;
    }

    Mat img = imread(argv[1]);
    if (img.empty()) {
        printf("错误: 无法加载图像\n");
        return -1;
    }

    namedWindow("图像查看器", WINDOW_KEEPRATIO | WINDOW_FREERATIO);
    imshow("图像查看器", img);

    printf("按 'q' 或 ESC 键退出\n");
    int key;
    while ((key = waitKey(0)) != 'q' && key != 27) {
        // 等待退出
    }

    destroyAllWindows();
    return 0;
}
```

### 6.2 交互式亮度控制

```cpp
#include <opencv2/highgui.hpp>

Mat originalImg;  // 原始图像
int brightness = 50;  // 亮度值

// 更新亮度回调函数
void updateBrightness(int, void*) {
    Mat adjusted;
    // alpha=1.0, beta=brightness-50 (偏移量)
    originalImg.convertTo(adjusted, -1, 1.0, brightness - 50);
    imshow("亮度控制", adjusted);
}

int main() {
    originalImg = imread("photo.jpg");
    if (originalImg.empty()) return -1;

    namedWindow("亮度控制", WINDOW_AUTOSIZE);

    // 创建滑动条: 名称, 窗口名, 绑定变量, 最大值, 回调
    createTrackbar("亮度", "亮度控制", &brightness, 100, updateBrightness);

    updateBrightness(0, nullptr);  // 初始化显示
    waitKey(0);

    return 0;
}
```

### 6.3 鼠标事件处理

```cpp
#include <opencv2/highgui.hpp>

Mat img;
vector<Point> clickPoints;  // 点击位置列表

// 鼠标回调函数
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        // 画圆标记点击位置
        circle(img, Point(x, y), 5, Scalar(0, 0, 255), -1);
        clickPoints.push_back(Point(x, y));
        imshow("点击演示", img);
        printf("点击位置: (%d, %d)\n", x, y);
    }
}

int main() {
    img = imread("photo.jpg");
    namedWindow("点击演示");
    setMouseCallback("点击演示", onMouse);

    imshow("点击演示", img);
    waitKey(0);

    printf("总点击次数: %zu\n", clickPoints.size());
    return 0;
}
```

### 6.4 滑动条 + 鼠标组合

```cpp
#include <opencv2/highgui.hpp>

Mat img;
int radius = 10;      // 圆的半径
int colorIdx = 0;     // 颜色索引
vector<Point> clickPoints;  // 点击位置

// 获取颜色
Scalar getColor(int idx) {
    Scalar colors[] = {Scalar(0,0,255), Scalar(0,255,0), Scalar(255,0,0)};
    return colors[idx % 3];
}

// 绘制更新回调
void draw(int, void*) {
    Mat display = img.clone();
    for (const auto& pt : clickPoints) {
        circle(display, pt, radius, getColor(colorIdx), -1);
    }
    imshow("交互演示", display);
}

// 鼠标回调
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        clickPoints.push_back(Point(x, y));
        draw(0, nullptr);
    }
}

int main() {
    img = imread("photo.jpg");
    namedWindow("交互演示", WINDOW_AUTOSIZE);

    // 创建滑动条
    createTrackbar("半径", "交互演示", &radius, 50, draw);
    createTrackbar("颜色", "交互演示", &colorIdx, 2, draw);

    setMouseCallback("交互演示", onMouse);
    draw(0, nullptr);

    waitKey(0);
    return 0;
}
```

---

## 7.练习题

### 入门级
1. 创建一个支持键盘导航的简单图像查看器 (方向键)
2. 添加滑动条来控制图像亮度
3. 单击图像时显示鼠标坐标

### 中级
4. 实现带鼠标滚轮支持的缩放工具
5. 实现颜色选择器, 使用滑动条控制 B, G, R 通道
6. 构建带有不同形状的图像标注工具

### 高级
7. 使用鼠标回调实现具有类 Qt 行为的自定义窗口
8. 创建带多个滑动条的实时滤镜查看器

---

## 8.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [highgui 模块](https://docs.opencv.org/4.14.0/d4/da4/group__highgui.html) |
| namedWindow 参考 | [namedWindow](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#ga9d805922020e34e47b805da6f6a44d57) |
| imshow 参考 | [imshow](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#ga5813e1ef30c55bcbb9ed9b820e5ffe9f) |
| setMouseCallback 参考 | [setMouseCallback](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#ga84e2f77f525b50d8c8c0e4a7b9f0eae8) |
| createTrackbar 参考 | [createTrackbar](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#gac5a3d2c47c26c1e340c8e8af3dd637a3) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 highgui 模块文档 |