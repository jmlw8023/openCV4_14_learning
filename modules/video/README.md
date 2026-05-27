# OpenCV video 模块学习指南

> **模块**: video
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 2 - 基础

---

## 目录

1. [概述](./README.md#1概述)
2. [目标跟踪 API](./README.md#2目标跟踪-api)
3. [核心算法详解](./README.md#3核心算法详解)
4. [光流算法](./README.md#4光流算法)
5. [代码示例](./README.md#5代码示例)
6. [练习题](./README.md#6练习题)
7. [参考资料](./README.md#7参考资料)

---

## 1.概述

**video** 模块提供视频分析和目标跟踪功能:

| 功能 | 描述 |
|---------|-------------|
| **目标跟踪** | KCF, MOSSE, MEDIAN_FLOW, GOTURN |
| **光流计算** | Farneback, Lucas-Kanade, TVL1 |
| **视频分割** | 基于颜色的分割 |
| **运动分析** | 背景减除 |

**头文件**: `opencv2/video.hpp`

---

## 2.目标跟踪 API

### 2.1 跟踪器类层次

```
Tracking API
├── Legacy (旧版, 仍可用)
│   ├── MIL
│   ├── BOOSTING
│   ├── MEDIAN_FLOW
│   ├── TLD
│   └── KCF (原版)
│
└── NanoTrack (新版, 2023+)
    └── NanoTrackTracket
```

### 2.2 通用跟踪接口

```cpp
#include <opencv2/video.hpp>
#include <opencv2/tracking/tracker.hpp>  // 新版

// 创建跟踪器
Ptr<Tracker> tracker = TrackerKCF::create();

// 初始化 (第一帧 + 初始位置)
Rect2d bbox = selectROI("选择目标", frame);
tracker->init(frame, bbox);

// 更新 (后续帧)
bool success = tracker->update(frame, bbox);

// 使用 bbox 绘制
if (success) {
    rectangle(frame, bbox, Scalar(255, 0, 0), 2);
}
```

### 2.3 各跟踪器对比

| 跟踪器 | 速度 | 精度 | 抗遮挡 | 适用场景 |
|--------|------|------|--------|----------|
| KCF | 快 | 高 | 中 | 通用 |
| MOSSE | 极快 | 中 | 差 | 高速需求 |
| MEDIAN_FLOW | 快 | 中 | 差 | 短期跟踪 |
| GOTURN | 中 | 高 | 中 | 需要深度学习 |
| NanoTrack | 极快 | 高 | 中 | 实时应用 |
| TLD | 慢 | 高 | 好 | 长期跟踪 |

---

## 3.核心算法详解

### 3.1 KCF (Kernelized Correlation Filters)

KCF 是基于核相关滤波的目标跟踪算法:

```cpp
#include <opencv2/tracking/kcf.hpp>

// 创建 KCF 跟踪器
Ptr<TrackerKCF> tracker = TrackerKCF::create();

// 可选参数
tracker->setFeatureExtractor();
tracker->setKernel();
```

### 3.2 MOSSE (Minimum Output Sum of Squared Error)

MOSSE 是基于自适应相关的跟踪器, 速度极快:

```cpp
#include <opencv2/tracking/mosse.hpp>

Ptr<TrackerMOSSE> tracker = TrackerMOSSE::create();

// 初始化
tracker->init(frame, boundingBox);

// 更新
tracker->update(frame, boundingBox);
```

### 3.3 GOTURN (Generic Object Tracking Using Regression Networks)

基于深度学习的跟踪器, 需要模型文件:

```cpp
#include <opencv2/tracking/goturn.hpp>

// 需要下载模型文件
Ptr<TrackerGOTURN> tracker = TrackerGOTURN::create();

// 注意: GOTURN 需要额外的模型文件
// 下载地址: https://raw.githubusercontent.com/AaronJin2016/GOTURN_Trained_Model/master/tracker.gob
```

### 3.4 NanoTrack

最新高性能跟踪器 (OpenCV 4.5+):

```cpp
#include <opencv2/tracking/nanotrack.hpp>

Ptr<TrackerNanoTrack> tracker = TrackerNanoTrack::create();
```

---

## 4.光流算法

### 4.1 光流概念

光流是图像中物体运动的速度场, 描述每个像素在时间上的位移:

```
I(x, y, t) → I(x+dx, y+dy, t+dt)
光流: (u, v) = (dx/dt, dy/dt)
```

### 4.2 Farneback 光流

稠密光流, 计算每个像素的运动:

```cpp
#include <opencv2/video.hpp>

Mat prevGray, currentGray;

// 计算光流
Mat flow;
calcOpticalFlowFarneback(prevGray, currentGray,
                         flow, 0.5, 3, 15, 3, 5, 1.2, 0);

// flow 是 2通道 Mat, (u, v) = (flow.at<Vec2f>(y,x)[0], flow.at<Vec2f>(y,x)[1])

// 绘制光流
void drawFlow(const Mat& flow, Mat& result) {
    result = Mat::zeros(flow.size(), CV_8UC3);
    for (int y = 0; y < flow.rows; y += 16) {
        for (int x = 0; x < flow.cols; x += 16) {
            Point2f f = flow.at<Point2f>(y, x);
            arrowedLine(result, Point(x, y), Point(x + f.x, y + f.y),
                        Scalar(0, 255, 0), 1, 8, 0, 0.5);
        }
    }
}
```

### 4.3 Lucas-Kanade 光流

稀疏光流, 只在特征点计算:

```cpp
#include <opencv2/video.hpp>

// 特征点检测
vector<Point2f> prevPoints, currentPoints;
goodFeaturesToTrack(prevGray, prevPoints, 100, 0.3, 7);

// Lucas-Kanade 光流
vector<uchar> status;
vector<float> err;
calcOpticalFlowPyrLK(prevGray, currentGray,
                     prevPoints, currentPoints,
                     status, err, Size(21, 21), 3);

// 绘制跟踪结果
for (size_t i = 0; i < currentPoints.size(); i++) {
    if (status[i]) {
        circle(result, currentPoints[i], 3, Scalar(0, 255, 0), -1);
        line(result, prevPoints[i], currentPoints[i], Scalar(0, 0, 255), 2);
    }
}
```

### 4.4 Dense Optical Flow API

```cpp
// TVL1 稠密光流 (L1范数优化)
OpticalFlowDual_TVL1::create()->calc(prevGray, currentGray, flow);

// 稀疏到稠密扩展
buildOpticalFlowDictionary(prevGray, currentGray, prevPoints, flow);
```

---

## 5.代码示例

### 5.1 基础目标跟踪

```cpp
#include <opencv2/video.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/highgui.hpp>

int main() {
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) return -1;

    Mat frame;
    cap >> frame;

    // 选择初始跟踪区域
    Rect2d bbox = selectROI("选择跟踪目标", frame);
    if (bbox.width == 0 || bbox.height == 0) return -1;

    // 创建跟踪器
    Ptr<Tracker> tracker = TrackerKCF::create();
    tracker->init(frame, bbox);

    namedWindow("跟踪", WINDOW_AUTOSIZE);

    while (cap >> frame) {
        // 更新跟踪
        double fps = cap.get(CAP_PROP_FPS);
        bool success = tracker->update(frame, bbox);

        // 绘制结果
        if (success) {
            rectangle(frame, bbox, Scalar(255, 0, 0), 2);
            putText(frame, "KCF Tracker", Point(10, 30),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        }

        imshow("跟踪", frame);

        int key = waitKey(30);
        if (key == 'q' || key == 27) break;
    }

    destroyAllWindows();
    return 0;
}
```

### 5.2 多目标跟踪

```cpp
// Multi-target tracking with separate trackers
void multiTargetTracking(const Mat& firstFrame, const vector<Rect2d>& targets) {
    vector<Ptr<Tracker>> trackers;
    vector<Rect2d> bboxes;
    vector<string> labels;

    // 为每个目标创建跟踪器
    for (size_t i = 0; i < targets.size(); i++) {
        Ptr<Tracker> tracker = TrackerKCF::create();
        tracker->init(firstFrame, targets[i]);

        trackers.push_back(tracker);
        bboxes.push_back(targets[i]);
        labels.push_back("目标 " + to_string(i + 1));
    }

    VideoCapture cap(0);
    Mat frame;
    int labelIdx = 0;

    while (cap >> frame) {
        for (size_t i = 0; i < trackers.size(); i++) {
            bool success = trackers[i]->update(frame, bboxes[i]);

            if (success) {
                Scalar color = Scalar(rand() % 256, rand() % 256, rand() % 256);
                rectangle(frame, bboxes[i], color, 2);
                putText(frame, labels[i], Point(bboxes[i].x, bboxes[i].y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
        }

        imshow("多目标跟踪", frame);
        if (waitKey(30) == 'q') break;
    }
}
```

### 5.3 光流可视化

```cpp
// 视频光流可视化
void visualizeOpticalFlow(const string& videoFile) {
    VideoCapture cap(videoFile);
    if (!cap.isOpened()) return;

    Mat frame, gray, prevGray;
    vector<Point2f> prevPoints, currentPoints;

    cap >> frame;
    cvtColor(frame, prevGray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(prevGray, prevPoints, 100, 0.3, 7);

    namedWindow("光流", WINDOW_NORMAL);

    while (cap >> frame) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Lucas-Kanade 光流
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prevGray, gray,
                             prevPoints, currentPoints,
                             status, err, Size(21, 21), 3);

        // 绘制
        Mat result = frame.clone();
        for (size_t i = 0; i < currentPoints.size(); i++) {
            if (status[i]) {
                // 绘制轨迹
                circle(result, currentPoints[i], 2, Scalar(0, 255, 0), -1);
                line(result, prevPoints[i], currentPoints[i],
                     Scalar(0, 0, 255), 2);
            }
        }

        imshow("光流", result);

        // 更新
        swap(prevPoints, currentPoints);
        swap(prevGray, gray);

        if (waitKey(30) == 'q') break;
    }

    destroyAllWindows();
}
```

### 5.4 Farneback 稠密光流

```cpp
// 稠密光流场可视化
void denseOpticalFlowDemo(const string& videoFile) {
    VideoCapture cap(videoFile);
    if (!cap.isOpened()) return;

    Mat frame, gray, prevGray, flow, flowDisplay;

    cap >> frame;
    cvtColor(frame, prevGray, COLOR_BGR2GRAY);

    namedWindow("稠密光流", WINDOW_NORMAL);

    while (cap >> frame) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 计算 Farneback 光流
        calcOpticalFlowFarneback(prevGray, gray, flow,
                                 0.5, 3, 15, 3, 5, 1.2, 0);

        // 转换为显示格式
        Mat flowGray;
        Mat magnitude, angle;
        cartToPolar(flow.at<Vec2f>(Point(y, x))[0],
                    flow.at<Vec2f>(Point(y, x))[1],
                    magnitude, angle, true);
        normalize(magnitude, flowGray, 0, 255, NORM_MINMAX);
        flowGray.convertTo(flowGray, CV_8UC1);

        // HSV 着色
        Mat hsv(flow.size(), CV_32FC3);
        vector<Mat> hsvChannels;
        split(hsv, hsvChannels);
        hsvChannels[0] = angle * (180 / CV_PI / 2);
        hsvChannels[1] = Mat::ones(flow.size(), CV_32FC1) * 255;
        hsvChannels[2] = flowGray;
        merge(hsvChannels, hsv);
        cvtColor(hsv, flowDisplay, COLOR_HSV2BGR);

        imshow("稠密光流", flowDisplay);

        swap(prevGray, gray);

        if (waitKey(30) == 'q') break;
    }

    destroyAllWindows();
}
```

### 5.5 运动检测 (背景减除)

```cpp
// 简单运动检测
void motionDetection(const string& videoFile) {
    VideoCapture cap(videoFile);
    if (!cap.isOpened()) return;

    // 创建背景减除器
    Ptr<BackgroundSubtractor> mog = createBackgroundSubtractorMOG2(500, 16.0, false);
    Ptr<BackgroundSubtractor> knn = createBackgroundSubtractorKNN(500, 400.0, false);

    Mat frame, fgMask, fgMaskKNN;

    namedWindow("原图", WINDOW_NORMAL);
    namedWindow("MOG2", WINDOW_NORMAL);
    namedWindow("KNN", WINDOW_NORMAL);

    while (cap >> frame) {
        // MOG2
        mog->apply(frame, fgMask);
        threshold(fgMask, fgMask, 200, 255, THRESH_BINARY);

        // KNN
        knn->apply(frame, fgMaskKNN);
        threshold(fgMaskKNN, fgMaskKNN, 200, 255, THRESH_BINARY);

        // 形态学处理
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);
        morphologyEx(fgMaskKNN, fgMaskKNN, MORPH_OPEN, kernel);

        // 显示
        imshow("原图", frame);
        imshow("MOG2", fgMask);
        imshow("KNN", fgMaskKNN);

        if (waitKey(30) == 'q') break;
    }

    destroyAllWindows();
}
```

---

## 6.练习题

### 入门级
1. 实现 KCF 目标跟踪器的基本使用
2. 使用 Lucas-Kanade 光流跟踪视频中的特征点
3. 实现简单的运动检测 (背景减除)

### 中级
4. 实现多目标跟踪系统
5. 实现光流可视化 (稠密光流)
6. 比较不同跟踪器 (KCF, MOSSE, MEDIAN_FLOW) 的性能

### 高级
7. 实现基于光流的运动估计
8. 实现基于跟踪的航迹维持 (tracking-by-detection)
9. 实现长期目标跟踪 (TLD 或 GOTURN)

### 挑战题
10. 实现多摄像头跟踪协同
11. 结合深度学习的跟踪器 (如 DaSiamRPN)

---

## 7.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [video 模块](https://docs.opencv.org/4.14.0/d2/d05/group__video.html) |
| Tracking API | [Tracking](https://docs.opencv.org/4.14.0/d2/d77/group__tracking.html) |
| KCF Paper | [KCF: High-Speed Tracking with Kernelized Correlation Filters](https://arxiv.org/abs/1404.7584) |
| MOSSE Paper | [Visual Object Tracking using Adaptive Correlation Filters](https://www.cs.colostate.edu/~vision/publications/bolme_davis2010.pdf) |
| NanoTrack | [NanoTrack: Lightweight Object Tracking for Edge Devices](https://arxiv.org/abs/2209.00616) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 video 模块文档 |