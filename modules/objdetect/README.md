# OpenCV objdetect 模块学习指南

> **模块**: objdetect
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 3 - 中级

---

## 目录

1. [概述](./README.md#1概述)
2. [级联分类器](./README.md#2级联分类器)
3. [HOG 特征](./README.md#3hog-特征)
4. [QR码检测](./README.md#4qr码检测)
5. [ArUco 标记](./README.md#5aruco-标记)
6. [深度学习检测器](./README.md#6深度学习检测器)
7. [代码示例](./README.md#7代码示例)
8. [练习题](./README.md#8练习题)
9. [参考资料](./README.md#9参考资料)

---

## 1.概述

**objdetect** 模块提供目标检测功能:

| 功能 | 描述 |
|---------|-------------|
| **级联分类器** | Haar, LBP 分类器 (人脸检测等) |
| **HOG** | 方向梯度直方图 (行人检测) |
| **QR码** | 二维码检测和解码 |
| **ArUco** | AR 标记检测 |
| **DNN检测** | SSD, YOLO, Faster-RCNN |

**头文件**: `opencv2/objdetect.hpp`

---

## 2.级联分类器

### 2.1 概念

级联分类器由多个弱分类器级联组成, 用于快速排除非目标区域:

```
输入图像
    ↓
级联 Stage 1 (快速检测) → 否 → 排除
    ↓ 是
级联 Stage 2 (稍慢检测) → 否 → 排除
    ↓ 是
...
    ↓
最终结果 (是人脸)
```

### 2.2 使用流程

```cpp
#include <opencv2/objdetect.hpp>

// 加载预训练分类器
CascadeClassifier face_cascade;
face_cascade.load("haarcascade_frontalface_default.xml");

// 或从 OpenCV 数据加载
face_cascade.load("/path/to/opencv/data/haarcascades/haarcascade_frontalface_default.xml");

// 检测
vector<Rect> faces;
Mat gray;
cvtColor(img, gray, COLOR_BGR2GRAY);
equalizeHist(gray, gray);  // 直方图均衡化提高检测率

face_cascade.detectMultiScale(gray, faces,
    1.1,    // scaleFactor: 搜索窗口缩放比例
    3,      // minNeighbors: 最少邻居数
    0,      // flags: 检测标志
    Size(30, 30));  // minSize: 最小检测窗口
```

### 2.3 预训练分类器

| 分类器文件 | 检测目标 | 描述 |
|-----------|----------|------|
| haarcascade_frontalface_default.xml | 正面人脸 | 默认参数 |
| haarcascade_frontalface_alt2.xml | 正面人脸 | alt2变体 |
| haarcascade_profileface.xml | 侧面人脸 | 侧脸 |
| haarcascade_upper_body.xml | 上半身 | 身体上半部分 |
| haarcascade_fullbody.xml | 全身 | 完整人体 |
| haarcascade_smile.xml | 微笑 | 笑脸检测 |
| haarcascade_eye.xml | 眼睛 | 眼睛检测 |
| lbpcascades/lbpcascade_frontalface.xml | 人脸 | LBP 版本 (更快) |

### 2.4 参数调优

```cpp
// scaleFactor 参数
// 值越大 → 检测越快, 但可能漏检
// 值越小 → 检测越慢, 但更准确
face_cascade.detectMultiScale(gray, faces, 1.05);  // 慢但准确
face_cascade.detectMultiScale(gray, faces, 1.3);   // 快但可能漏检

// minNeighbors 参数
// 值越大 → 检测要求更严格, 较少误检但可能漏检
// 值越小 → 检测较宽松, 较多误检但少漏检
face_cascade.detectMultiScale(gray, faces, 1.1, 1);   // 宽松
face_cascade.detectMultiScale(gray, faces, 1.1, 10);  // 严格

// minSize 和 maxSize
Size minSize(30, 30);
Size maxSize(200, 200);
face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, minSize, maxSize);
```

---

## 3.HOG 特征

### 3.1 HOG 原理

方向梯度直方图 (Histogram of Oriented Gradients) 用于行人检测:

```
1. 计算图像梯度 (Sobel)
2. 分割为 cells (如 8×8 像素)
3. 每个 cell 计算梯度方向直方图 (9 bins)
4. Block 归一化 (2×2 cells)
5. 串联所有 block 的直方图 → HOG 特征
```

### 3.2 HOG 检测器

```cpp
#include <opencv2/objdetect.hpp>

// 创建 HOG 检测器
Ptr<HOGDescriptor> hog = HOGDescriptor(
    Size(64, 128),    // 检测窗口大小
    Size(16, 16),     // block 大小
    Size(8, 8),       // cell 大小
    Size(8, 8),       // block 步长
    9                 // bins 数量
);

// 设置 SVM
hog->setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

// 检测
vector<Rect> found;
vector<double> weights;
hog->detectMultiScale(img, found, weights, 0, Size(8, 8), Size(16, 16), 1.05);

// 绘制结果
for (auto& r : found) {
    rectangle(img, r, Scalar(0, 255, 0), 2);
}
```

### 3.3 自定义 HOG + SVM

```cpp
// 计算 HOG 特征
Mat computeHOG(const Mat& img) {
    HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8),
                      Size(8, 8), 9);
    vector<float> descriptors;
    hog.compute(img, descriptors);
    return Mat(descriptors).reshape(1, 1);
}

// 训练 SVM (使用 cv::ml::SVM)
void trainSVM(const vector<Mat>& positiveSamples,
              const vector<Mat>& negativeSamples) {
    // 准备训练数据
    Mat trainData;
    Mat labels;
    // ... 填充 trainData 和 labels ...

    // 训练
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::RBF);
    svm->train(trainData, ml::ROW_SAMPLE, labels);

    // 设置到 HOG
    Mat sv, rho, alpha;
    svm->getDecisionFunction(0, rho, sv, alpha);
    vector<float> detector = -alpha * sv;
    detector.push_back(-rho.at<float>(0));

    HOGDescriptor hog;
    hog.setSVMDetector(detector);
}
```

---

## 4.QR码检测

### 4.1 检测和解码

```cpp
#include <opencv2/objdetect.hpp>

// 创建 QR 码检测器
QRCodeDetector detector;

// 检测
vector<Point2f> corners;
MatStraightBarcode = detector.detect(img, corners);

// 解码 (仅支持一个二维码)
string data = detector.decode(img, corners);

// 检测 + 解码
Mat straightBarcode;
string data = detector.detectAndDecode(img, corners, straightBarcode);

// 如果有多个二维码
vector<Point2f> corners;
vector<Mat> straightBarcodes;
vector<string> data;
detector.detectMulti(img, corners);
detector.decodeMulti(img, corners, data, straightBarcodes);
```

### 4.2 完整示例

```cpp
// QR 码实时检测
void qrCodeDetection(const string& videoFile) {
    VideoCapture cap(videoFile);
    Mat frame, gray;

    QRCodeDetector detector;
    vector<Point2f> corners;

    while (cap >> frame) {
        // 检测
        corners.clear();
        Mat straight;
        string data = detector.detectAndDecode(frame, corners, straight);

        if (!data.empty()) {
            // 绘制角点
            for (int i = 0; i < 4; i++) {
                line(frame, corners[i], corners[(i + 1) % 4],
                     Scalar(0, 255, 0), 3);
            }

            // 显示解码内容
            putText(frame, data, corners[0] - Point(10, 10),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        imshow("QR 检测", frame);
        if (waitKey(30) == 'q') break;
    }
}
```

---

## 5.ArUco 标记

### 5.1 基本概念

ArUco 是一种方形 AR 标记, 用于相机标定和 AR 应用:

```cpp
#include <opencv2/aruco.hpp>

// 预定义字典
// DICT_4X4_50: 4×4 像素, 50 个标记
// DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000
// DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000
// DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000
```

### 5.2 检测标记

```cpp
// 创建字典和检测器
Ptr<aruco::Dictionary> dict = aruco::getPredefinedDictionary(
    aruco::DICT_6X6_250);
Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();

vector<vector<Point2f>> corners, rejected;
vector<int> ids;

// 检测
aruco::detectMarkers(img, dict, corners, ids, params, rejected);

// 绘制检测结果
Mat result = img.clone();
aruco::drawDetectedMarkers(result, corners, ids);

// 估计姿态 (需要相机内参)
Mat cameraMatrix, distCoeffs;
// 假设已知相机标定参数
vector<Vec3d> rvecs, tvecs;
aruco::estimatePoseSingleMarkers(corners, 0.1, cameraMatrix,
                                   distCoeffs, rvecs, tvecs);

// 绘制轴
for (size_t i = 0; i < rvecs.size(); i++) {
    aruco::drawAxis(result, cameraMatrix, distCoeffs,
                    rvecs[i], tvecs[i], 0.1);
}
```

### 5.3 创建标记图像

```cpp
// 生成 ArUco 标记图像
void generateArucoMarkers() {
    Ptr<aruco::Dictionary> dict = aruco::getPredefinedDictionary(
        aruco::DICT_6X6_50);

    Mat marker;
    for (int id = 0; id < 50; id++) {
        aruco::drawMarker(dict, id, 200, marker, 1);
        string filename = "marker_" + to_string(id) + ".png";
        imwrite(filename, marker);
    }
}
```

---

## 6.深度学习检测器

### 6.1 DNN 模块检测

```cpp
#include <opencv2/dnn.hpp>

// 加载模型
Net net = readNetFromONNX("ssd_mobilenet.onnx");
net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);

// 预处理输入
Mat blob = blobFromImage(img, 1.0, Size(300, 300),
                        Scalar(127.5, 127.5, 127.5), true, false);

// 前向传播
net.setInput(blob);
Mat output = net.forward();

// 解析输出 (SSD 格式)
// ... 解析检测框 ...
```

### 6.2 OpenCV DNN 对象检测

```cpp
// 使用 OpenCV 提供的预训练模型
// 模型: https://github.com/opencv/opencv/tree/master/samples/dnn

// YOLOv3
Net netYOLO = readNetFromDarknet("yolov3.cfg", "yolov3.weights");

// SSD MobileNet
Net netSSD = readNetFromTensorflow("ssd_mobilenet_v2_coco.pbtxt",
                                    "ssd_mobilenet_v2_coco.pb");

// 使用 DNN 对象检测 (参见 dnn 模块)
```

---

## 7.代码示例

### 7.1 人脸检测

```cpp
// 基于 Haar 级联分类器的人脸检测
void faceDetection(const Mat& img) {
    CascadeClassifier face_cascade;
    face_cascade.load("haarcascade_frontalface_default.xml");

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces,
        1.1,    // scaleFactor
        3,      // minNeighbors
        0,      // flags
        Size(30, 30));  // minSize

    // 绘制检测结果
    Mat result = img.clone();
    for (auto& face : faces) {
        rectangle(result, face, Scalar(255, 0, 0), 2);

        // 眼睛检测 (在脸部区域内)
        Mat faceROI = gray(face);
        CascadeClassifier eye_cascade;
        eye_cascade.load("haarcascade_eye.xml");
        vector<Rect> eyes;
        eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 3);
        for (auto& eye : eyes) {
            Point center(face.x + eye.x + eye.width/2,
                        face.y + eye.y + eye.height/2);
            circle(result, center, eye.width/2,
                   Scalar(0, 255, 0), 2);
        }
    }

    imshow("人脸检测", result);
}
```

### 7.2 行人检测 (HOG + SVM)

```cpp
// 使用 HOG 进行行人检测
void pedestrianDetection(const Mat& img) {
    // 初始化 HOG 检测器 (使用预训练模型)
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // 检测
    vector<Rect> found;
    vector<double> weights;
    hog.detectMultiScale(img, found, weights, 0, Size(8, 8),
                        Size(32, 32), 1.05);

    // 绘制结果 (过滤低权重检测)
    Mat result = img.clone();
    for (size_t i = 0; i < found.size(); i++) {
        if (weights[i] > 0.5) {  // 过滤低置信度
            rectangle(result, found[i], Scalar(0, 255, 0), 2);
        }
    }

    imshow("行人检测", result);
}
```

### 7.3 ArUco 相机标定

```cpp
// 使用 ArUco 标定板进行相机标定
void calibrateCameraWithAruco(const vector<string>& imageFiles) {
    // 标定板参数
    Ptr<aruco::Dictionary> dict = aruco::getPredefinedDictionary(
        aruco::DICT_6X6_250);
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(
        Size(5, 7), 0.035, 0.02, dict);
    Ptr<aruco::CharucoDetector> detector = aruco::CharucoDetector(board);

    vector<vector<Point2f>> charucoCorners;
    vector<vector<int>> charucoIds;
    Size imageSize;

    for (const auto& file : imageFiles) {
        Mat img = imread(file);
        if (img.empty()) continue;

        // 检测
        vector<int> ids;
        vector<vector<Point2f>> corners;
        detector.detectBoard(img, corners, ids);

        if (!ids.empty()) {
            // 获取 Charuco 角点
            vector<Point2f> charucoCornersSingle;
            vector<int> charucoIdsSingle;
            board->matchImagePoints(corners, ids,
                                   charucoCornersSingle, charucoIdsSingle);
            charucoCorners.push_back(charucoCornersSingle);
            charucoIds.push_back(charucoIdsSingle);
            imageSize = img.size();
        }
    }

    if (charucoCorners.size() >= 10) {
        // 相机标定
        Mat cameraMatrix, distCoeffs;
        vector<Mat> rvecs, tvecs;
        double rms = calibrateCamera(
            vector<vector<Point3f>>(charucoCorners.size(), board->getChessboardCorners()),
            charucoCorners, imageSize,
            cameraMatrix, distCoeffs, rvecs, tvecs);

        cout << "RMS 误差: " << rms << endl;
        cout << "相机内参:\n" << cameraMatrix << endl;
        cout << "畸变系数:\n" << distCoeffs << endl;
    }
}
```

### 7.4 实时 QR 码检测

```cpp
// 实时 QR 码扫描
void qrCodeScanner() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return;

    QRCodeDetector detector;
    Mat frame;

    while (cap >> frame) {
        vector<Point2f> corners;
        Mat straight;

        string data = detector.detectAndDecode(frame, corners, straight);

        if (!data.empty()) {
            // 绘制检测框
            for (int i = 0; i < 4; i++) {
                line(frame, corners[i], corners[(i + 1) % 4],
                     Scalar(0, 255, 0), 3);
            }

            // 显示数据
            putText(frame, "Data: " + data, Point(10, 30),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        imshow("QR Scanner", frame);

        int key = waitKey(30);
        if (key == 'q' || key == 27) break;
    }

    destroyAllWindows();
}
```

---

## 8.练习题

### 入门级
1. 使用 Haar 级联检测人脸
2. 检测并解码 QR 码
3. 检测 ArUco 标记

### 中级
4. 实现带眼睛检测的多级人脸检测
5. 使用 HOG 进行行人检测
6. 实现基于 ArUco 的相机标定

### 高级
7. 使用 DNN 进行对象检测 (SSD/YOLO)
8. 实现自定义级联分类器训练
9. 实现 AR 应用 (基于 ArUco 标记)

### 挑战题
10. 结合 DNN 和传统方法的混合检测系统
11. 实现基于 YOLO 的实时目标检测优化

---

## 9.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [objdetect 模块](https://docs.opencv.org/4.14.0/d2/d99/group__objdetect.html) |
| Haar 分类器 | [Viola-Jones Object Detection](https://www.cs.cmu.edu/~efros/courses/LBVS/LBVS04/ viola_jones.pdf) |
| HOG 特征 | [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) |
| ArUco 文档 | [ArUco: A library for camera calibration and AR](https://docs.opencv.org/4.14.0/d4/d26/tutorial_aruco_calibration.html) |
| OpenCV DNN 检测 | [DNN Object Detection](https://docs.opencv.org/4.14.0/d6/d0f/group__dnn.html) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 objdetect 模块文档 |