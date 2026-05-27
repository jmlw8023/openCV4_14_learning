# OpenCV videoio 模块学习指南

> **模块**: videoio
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 2 - 基础

---

## 目录

1. [概述](./README.md#1概述)
2. [支持的视频格式](./README.md#2支持的视频格式)
3. [VideoCapture 详解](./README.md#3videocapture-详解)
4. [VideoWriter 详解](./README.md#4videowriter-详解)
5. [核心API](./README.md#5核心api)
6. [实现分析](./README.md#6实现分析)
7. [代码示例](./README.md#7代码示例)
8. [练习题](./README.md#8练习题)
9. [参考资料](./README.md#9参考资料)

---

## 1.概述

**videoio** 提供视频捕获和写入功能:

| 功能 | 描述 |
|---------|-------------|
| **VideoCapture** | 从摄像头/文件/网络流读取视频帧 |
| **VideoWriter** | 将视频帧写入文件 |
| **自动检测** | 基于后端自动检测格式 |
| **API 统一** | 统一的 C++/Python 接口 |

**头文件**: `opencv2/videoio.hpp`

---

## 2.支持的视频格式

### 后端及优先级

| 后端 | flag | 描述 |
|------|------|-------------|
| MSMF | CAP_MSMF | Windows Media Foundation (默认 Windows) |
| FFmpeg | CAP_FFMPEG | FFmpeg (需要编译支持) |
| OpenCV Manager | CAP_OPENCV_MJPEG | OpenCV 内置 |
| Unicap | CAP_UNICAP | Linux Unicap |
| V4L2 | CAP_V4L2 | Video4Linux2 (默认 Linux) |
| FireWire | CAP_FIREWIRE | IEEE 1394 |
| IMGS | CAP_IMGS | 图像序列 |

### 后端选择

```cpp
// 自动选择 (默认)
VideoCapture cap(filename);

// 强制指定后端
VideoCapture cap(filename, CAP_FFMPEG);
VideoCapture cap(CAP_V4L2 + CAP_CHANNEL_V4L2);

// 检查后端支持
bool support = VideoCapture::open(filename, CAP_FFMPEG);
```

### API 方式检测

| API | Windows | Linux | macOS |
|-----|---------|-------|-------|
| MSMF | ✅ | ❌ | ❌ |
| FFmpeg | ✅ | ✅ | ✅ |
| V4L2 | ❌ | ✅ | ❌ |
| AVFoundation | ❌ | ❌ | ✅ |

---

## 3.VideoCapture 详解

### 3.1 创建方式

```cpp
#include <opencv2/videoio.hpp>

// 方式1: 从文件打开
VideoCapture cap("video.mp4");

// 方式2: 从摄像头打开 (设备索引)
VideoCapture cap(0);           // 默认摄像头
VideoCapture cap(1);           // 第二个摄像头
VideoCapture cap(CAP_V4L2);    // 指定后端

// 方式3: 从网络流打开
VideoCapture cap("rtsp://192.168.1.100:554/stream");

// 方式4: 检查是否成功打开
if (!cap.isOpened()) {
    cout << "无法打开视频" << endl;
    return -1;
}
```

### 3.2 属性读取

```cpp
// 常用属性
cap.get(CAP_PROP_FRAME_WIDTH);   // 帧宽度
cap.get(CAP_PROP_FRAME_HEIGHT);  // 帧高度
cap.get(CAP_PROP_FPS);           // 帧率
cap.get(CAP_PROP_FRAME_COUNT);   // 总帧数
cap.get(CAP_PROP_POS_MSEC);      // 当前播放位置 (毫秒)
cap.get(CAP_PROP_POS_FRAMES);    // 当前帧编号
cap.get(CAP_PROP_POS_AVI_RATIO); // 当前位置比例 [0,1]
cap.get(CAP_PROP_CURRENT_BACKEND); // 当前后端

// 设置属性
cap.set(CAP_PROP_FRAME_WIDTH, 1280);
cap.set(CAP_PROP_FRAME_HEIGHT, 720);
cap.set(CAP_PROP_FPS, 30);
```

### 3.3 帧读取

```cpp
Mat frame;

// 方式1: 读取一帧 (自动判断颜色空间)
cap >> frame;           // operator>>
cap.read(frame);       // explicit

// 方式2: 逐帧循环
while (true) {
    cap >> frame;
    if (frame.empty()) break;  // 播放结束
    imshow("视频", frame);
    if (waitKey(30) == 'q') break;  // 30ms ≈ 30fps
}

// 方式3: 按帧号读取
cap.set(CAP_PROP_POS_FRAMES, 100);  // 跳到第100帧
cap.read(frame);

// 方式4: 按时间读取 (毫秒)
cap.set(CAP_PROP_POS_MSEC, 5000);   // 跳到第5秒
cap.read(frame);
```

### 3.4 摄像头特定设置

```cpp
// V4L2 后端特定设置
cap.set(CAP_PROP_V4L2_BUFFERSIZE, 4);  // 缓冲区大小

// MSMF 后端特定设置
cap.set(CAP_PROP_VFW_CAP_COMMON Guide, 1);

// 自动曝光
cap.set(CAP_PROP_AUTO_EXPOSURE, 0.75);  // 0.75 = auto, 0.25 = manual

// 亮度、对比度、饱和度
cap.set(CAP_PROP_BRIGHTNESS, 128);
cap.set(CAP_PROP_CONTRAST, 128);
cap.set(CAP_PROP_SATURATION, 128);

// 曝光和增益
cap.set(CAP_PROP_EXPOSURE, 100);
cap.set(CAP_PROP_GAIN, 100);
```

---

## 4.VideoWriter 详解

### 4.1 创建方式

```cpp
#include <opencv2/videoio.hpp>

// 方式1: 构造函数
VideoWriter writer;
writer.open("output.mp4", CAP_FFMPEG, VideoWriter::fourcc('H','2','6','4'),
            fps, Size(width, height));

// 方式2: 一步创建
VideoWriter writer("output.avi", CAP_FOURCC('M','J','P','G'),
                   30, Size(640, 480));
```

### 4.2 FourCC 编码

```cpp
// FourCC 码 (4字符代码)
// 常见编码格式:
VideoWriter::fourcc('H','2','6','4')  // H.264
VideoWriter::fourcc('M','J','P','G')  // Motion JPEG
VideoWriter::fourcc('X','V','I','D')  // Xvid
VideoWriter::fourcc('X','2','6','4')  // X264
VideoWriter::fourcc('M','P','4','V') // MPEG-4 Part 2
VideoWriter::fourcc('A','V','C','1') // H.264 with Annex A
VideoWriter::fourcc('Y','U','Y','2') // YUV 4:2:2

// 检查 FourCC 支持
cout << "支持的编码: " << writer.getBackendName() << endl;
```

### 4.3 属性设置

```cpp
VideoWriter writer;

// 获取属性
double fps = writer.get(CAP_PROP_FPS);
int fourcc = writer.get(CAP_PROP_FOURCC);

// 检查是否正常
if (!writer.isOpened()) {
    cout << "无法创建视频写入器" << endl;
    return -1;
}

// 写入帧
writer.write(frame);
// 或
writer << frame;
```

---

## 5.核心API

### 5.1 VideoCapture API

```cpp
class VideoCapture {
public:
    // 构造函数
    VideoCapture();                           // 空
    VideoCapture(const String& filename);    // 文件
    VideoCapture(int device);                 // 摄像头

    // 打开
    bool open(const String& filename);
    bool open(int device);
    bool open(const String& filename, int apiPreference);

    // 检查
    bool isOpened() const;
    void release();  // 释放资源

    // 读取
    VideoCapture& operator>>(Mat& image);
    VideoCapture& operator>>(UMat& image);
    bool read(Mat& image);
    bool read(UMat& image);

    // 属性
    double get(int propId) const;
    bool set(int propId, double value);

    // 多媒体格式信息
    int getBackendName() const;
};
```

### 5.2 VideoWriter API

```cpp
class VideoWriter {
public:
    // 构造函数
    VideoWriter();
    VideoWriter(const String& filename, int fourcc, double fps,
                Size frameSize, bool isColor = true);

    // 打开
    bool open(const String& filename, int fourcc, double fps,
              Size frameSize, bool isColor = true);

    // 检查
    bool isOpened() const;
    void release();

    // 写入
    VideoWriter& operator<<(const Mat& image);
    void write(const Mat& image);

    // 属性
    double get(int propId) const;
    bool set(int propId, double value);
    String getBackendName() const;
};
```

### 5.3 高级函数

```cpp
// 获取摄像头列表 (Windows)
vector<int> getAvailableCameras();

// 查询后端能力
bool backendAvailable(int apiPreference);

// 创建带有后端偏好的捕获
VideoCapture cap(CAP_DSHOW + 0);  // DirectShow on Windows

// 查询格式支持
bool queryFrame(VideoCapture& cap, Mat& frame);
```

---

## 6.实现分析

### 6.1 VideoCapture 架构

```
┌─────────────────────────────────────────────┐
│           VideoCapture (用户接口)            │
├─────────────────────────────────────────────┤
│ 1. API 分发 (open/read/get/set)            │
│ 2. 后端工厂 (选择合适的后端)                │
│ 3. 后端实现                                 │
│    ├─ MSMF (Windows Media Foundation)      │
│    ├─ V4L2 (Video4Linux2)                   │
│    ├─ FFmpeg (通用)                         │
│    └─ AVFoundation (macOS)                  │
│ 4. 帧缓冲 (循环缓冲)                        │
│ 5. 格式转换 (解码后转换)                    │
└─────────────────────────────────────────────┘
```

### 6.2 帧缓冲机制

```cpp
// 典型实现 (简化)
class VideoCapture {
    vector<Mat> buffers;  // 帧缓冲队列
    int head, tail;      // 读写指针

    bool readFrame() {
        // 从后端获取新帧
        Mat frame = backend->readFrame();
        buffers[tail] = frame;
        tail = (tail + 1) % buffers.size();
        return !frame.empty();
    }
};
```

### 6.3 后端选择逻辑

```cpp
// 优先级顺序 (Windows 示例)
1. 检查文件扩展名 (.mp4 → MSMF 或 FFmpeg)
2. 检查 URL 前缀 (rtsp:// → MSMF 或 FFmpeg)
3. 检查设备索引 → DirectShow/MSMF
4. 回退到默认后端
```

---

## 7.代码示例

### 7.1 基础视频读取与显示

```cpp
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // 打开视频文件
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cout << "无法打开视频文件" << endl;
        return -1;
    }

    // 获取视频信息
    double fps = cap.get(CAP_PROP_FPS);
    int frames = cap.get(CAP_PROP_FRAME_COUNT);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    cout << "FPS: " << fps << endl;
    cout << "总帧数: " << frames << endl;
    cout << "分辨率: " << width << "x" << height << endl;

    Mat frame;
    namedWindow("视频播放", WINDOW_NORMAL);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        imshow("视频播放", frame);

        // 按 ESC 退出, 按空格暂停
        int key = waitKey(30);
        if (key == 27) break;          // ESC
        if (key == ' ') {
            waitKey(0);                  // 暂停
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
```

### 7.2 摄像头捕获与录像

```cpp
// 摄像头捕获并保存到文件
int main() {
    VideoCapture cap(0);  // 默认摄像头
    if (!cap.isOpened()) {
        cout << "无法打开摄像头" << endl;
        return -1;
    }

    // 设置分辨率
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    // 创建 VideoWriter
    int fourcc = VideoWriter::fourcc('H','2','6','4');
    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter writer("output.mp4", fourcc, fps, Size(width, height));
    if (!writer.isOpened()) {
        cout << "无法创建视频写入器" << endl;
        return -1;
    }

    Mat frame;
    namedWindow("摄像头", WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 写入
        writer.write(frame);

        imshow("摄像头", frame);

        // 按 q 退出
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    writer.release();
    destroyAllWindows();

    cout << "视频已保存到 output.mp4" << endl;
    return 0;
}
```

### 7.3 视频处理流水线

```cpp
// 视频处理: 逐帧处理并输出
void processVideo(const string& inputFile, const string& outputFile) {
    VideoCapture cap(inputFile);
    if (!cap.isOpened()) return;

    // 获取输入视频属性
    int fourcc = VideoWriter::fourcc('H','2','6','4');
    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter writer(outputFile, fourcc, fps, Size(width, height));

    Mat frame, processed;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 处理: 边缘检测
        Canny(frame, processed, 50, 150);

        // 转换回彩色以便保存
        cvtColor(processed, processed, COLOR_GRAY2BGR);

        writer.write(processed);

        imshow("处理结果", processed);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    writer.release();
    destroyAllWindows();
}
```

### 7.4 按时间/帧号读取

```cpp
// 视频导航: 跳转到指定位置
void navigateVideo(const string& filename) {
    VideoCapture cap(filename);
    if (!cap.isOpened()) return;

    double totalFrames = cap.get(CAP_PROP_FRAME_COUNT);
    double fps = cap.get(CAP_PROP_FPS);
    double duration = totalFrames / fps;

    cout << "总时长: " << duration << " 秒" << endl;

    Mat frame;
    namedWindow("视频", WINDOW_NORMAL);

    while (true) {
        // 跳到指定位置 (百分比)
        double pos = 0.0;
        cout << "输入位置 (0-100%): ";
        if (!(cin >> pos)) break;

        cap.set(CAP_PROP_POS_AVI_RATIO, pos / 100.0);
        cap.read(frame);

        if (frame.empty()) break;

        // 显示当前时间
        double currentTime = cap.get(CAP_PROP_POS_MSEC) / 1000.0;
        putText(frame, format("%.1f s", currentTime),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        imshow("视频", frame);
    }

    cap.release();
    destroyAllWindows();
}
```

### 7.5 多摄像头同步

```cpp
// 多摄像头捕获
void multiCameraCapture() {
    // 尝试打开多个摄像头
    vector<VideoCapture> caps;
    for (int i = 0; i < 4; i++) {
        VideoCapture cap(i);
        if (cap.isOpened()) {
            caps.push_back(cap);
            cout << "摄像头 " << i << " 已打开" << endl;
        }
    }

    if (caps.empty()) {
        cout << "没有找到可用摄像头" << endl;
        return;
    }

    namedWindow("多摄像头", WINDOW_NORMAL);

    vector<Mat> frames(caps.size());
    while (true) {
        for (size_t i = 0; i < caps.size(); i++) {
            caps[i] >> frames[i];

            string windowName = "摄像头 " + to_string(i);
            imshow(windowName, frames[i]);
        }

        if (waitKey(30) == 'q') break;
    }

    destroyAllWindows();
}
```

---

## 8.练习题

### 入门级
1. 读取视频文件并逐帧显示
2. 打开摄像头并实时显示
3. 从摄像头捕获视频并保存到文件

### 中级
4. 实现视频播放器的暂停/继续功能
5. 实现视频的时间跳转功能 (百分比/秒)
6. 实现批量视频格式转换 (MP4 → AVI)

### 高级
7. 实现多摄像头同步捕获
8. 实现视频处理流水线 (如: 边缘检测输出)
9. 实现视频水印添加功能

### 挑战题
10. 实现自定义视频编码器后端
11. 实现 RTSP 流媒体服务器推送

---

## 9.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [videoio 模块](https://docs.opencv.org/4.14.0/d4/d15/group__videoio.html) |
| VideoCapture 参考 | [VideoCapture](https://docs.opencv.org/4.14.0/d8/dfe/classcv_1_1VideoCapture.html) |
| VideoWriter 参考 | [VideoWriter](https://docs.opencv.org/4.14.0/d7/dd1/classcv_1_1VideoWriter.html) |
| FFmpeg 安装 | [OpenCV FFmpeg](https://docs.opencv.org/4.14.0/d0/da7/tutorial_ios_install.html) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 videoio 模块文档 |