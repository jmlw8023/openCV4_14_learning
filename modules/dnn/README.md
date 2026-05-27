# OpenCV DNN 模块学习指南

> **模块**: dnn
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 4 - 高级

---

## 目录

1. [概述](./README.md#1概述)
2. [模型加载与推理](./README.md#2模型加载与推理)
3. [模型格式支持](./README.md#3模型格式支持)
4. [预处理与后处理](./README.md#4预处理与后处理)
5. [目标检测模型](./README.md#5目标检测模型)
6. [代码示例](./README.md#6代码示例)
7. [练习题](./README.md#7练习题)
8. [参考资料](./README.md#8参考资料)

---

## 1.概述

**dnn** (Deep Neural Network) 模块提供深度学习模型推理功能:

| 功能 | 描述 |
|---------|-------------|
| **模型加载** | ONNX, TensorFlow, PyTorch, Caffe, Darknet |
| **推理加速** | CPU (OpenBLAS/MKL), GPU (CUDA/OpenCL), Vulkan |
| **预处理** | 缩放、归一化、色彩转换 |
| **目标检测** | SSD, YOLO, Faster-RCNN, EfficientDet |

**头文件**: `opencv2/dnn.hpp`

---

## 2.模型加载与推理

### 2.1 支持的模型格式

| 格式 | 加载函数 | 说明 |
|------|----------|------|
| ONNX | readNetFromONNX | 推荐, 跨框架标准 |
| TensorFlow | readNetFromTensorflow | .pb, .pbtxt |
| PyTorch | readNetFromTorchscript | TorchScript 格式 |
| Caffe | readNetFromCaffe | .prototxt, .caffemodel |
| Darknet | readNetFromDarknet | .cfg, .weights |
| TensorFlow Lite | readNetFromTFLite | TFLite 模型 |

### 2.2 基本推理流程

```cpp
// ============================================
// C++ 推理流程
// ============================================
#include <opencv2/dnn.hpp>

// 加载网络
Net net = readNetFromONNX("model.onnx");
// 或
Net net = readNetFromTensorflow("model.pb");
Net net = readNetFromDarknet("yolov4.cfg", "yolov4.weights");

// 设置后端和目标
net.setPreferableBackend(DNN_BACKEND_OPENCV);    // 默认 CPU
net.setPreferableTarget(DNN_TARGET_CPU);
// GPU 加速 (需要 CUDA)
net.setPreferableBackend(DNN_BACKEND_CUDA);
net.setPreferableTarget(DNN_TARGET_CUDA);

// 预处理输入
Mat blob = blobFromImage(inputImage,   // 输入图像
                        1.0/255.0,   // scalefactor
                        Size(416, 416), // 目标尺寸
                        Scalar(0, 0, 0), // 均值 (BGR)
                        true,          // 是否交换RB
                        false);        // crop

// 推理
net.setInput(blob);
Mat output = net.forward();        // 最后一层输出
Mat output = net.forward("layer_name");  // 指定层输出

// 获取中间层输出
Mat midOutput = net.forward("intermediate_layer");
```

```python
# ============================================
# Python 推理流程
# ============================================
import cv2
import numpy as np

# 加载网络
net = cv2.dnn.readNetFromONNX("model.onnx")
# 或
net = cv2.dnn.readNetFromTensorflow("model.pb")
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

# 设置后端和目标
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# GPU 加速
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 预处理输入
blob = cv2.dnn.blobFromImage(
    inputImage,
    scalefactor=1.0/255.0,
    size=(416, 416),
    mean=(0, 0, 0),
    swapRB=True,
    crop=False
)

# 推理
net.setInput(blob)
output = net.forward()
# 或指定层
output = net.forward("layer_name")
```

---

## 3.模型格式支持

### 3.1 ONNX 模型

```cpp
// ============================================
// C++ ONNX 加载
// ============================================
Net net = readNetFromONNX("resnet50.onnx");

// ONNX 模型通常已包含预处理信息
// 检查输入维度
MatShape inShape = net.getLayerShapes("input", 0);
cout << "输入形状: " << inShape << endl;

// 检查输出维度
MatShape outShape = net.getLayerShapes("output", 0);
cout << "输出形状: " << outShape << endl;
```

```python
# ============================================
# Python ONNX 加载
# ============================================
net = cv2.dnn.readNetFromONNX("resnet50.onnx")

# 获取网络信息
print(f"输入数量: {net.getNumInputs()}")
print(f"输出数量: {net.getNumOutputs()}")
print(f"输入名称: {[net.getInputNames()]}")
print(f"输出名称: {[net.getOutputNames()]}")
```

### 3.2 TensorFlow 模型

```cpp
// ============================================
// C++ TensorFlow 加载
// ============================================
// 冻结图模型 (.pb)
Net net = readNetFromTensorflow("frozen_inference_graph.pb");
// 或保存为 .pbtxt
Net net = readNetFromTensorflow("graph.pbtxt", "graph.pb");

// 获取所有层名称
vector<String> layerNames = net.getLayerNames();
for (const auto& name : layerNames)
    cout << name << endl;
```

```python
# ============================================
# Python TensorFlow 加载
# ============================================
# 冻结图模型
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb")

# 检查模型输入输出
input_name = net.getInputNames()[0]
output_name = net.getOutputNames()[0]
print(f"输入: {input_name}")
print(f"输出: {output_name}")
```

### 3.3 YOLO (Darknet) 模型

```cpp
// ============================================
// C++ YOLO 加载
// ============================================
Net net = readNetFromDarknet("yolov4.cfg", "yolov4.weights");

// 设置计算后端
net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);

// 获取输出层名称
vector<String> outNames = net.getUnconnectedOutLayersNames();
for (const auto& name : outNames)
    cout << "输出层: " << name << endl;
```

```python
# ============================================
# Python YOLO 加载
# ============================================
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

# 获取输出层名称
out_names = net.getUnconnectedOutLayersNames()
print(f"输出层: {out_names}")
```

---

## 4.预处理与后处理

### 4.1 blobFromImage 参数详解

```cpp
// ============================================
// C++ 预处理
// ============================================
// 语法: blobFromImage(image, scalefactor, size, mean, swapRB, crop)
// 公式: (image - mean) * scalefactor

// 标准 ImageNet 预处理
// scalefactor = 1/255 (将 [0,255] 转为 [0,1])
// mean = (104, 117, 123) (BGR 顺序)
// swapRB = true (BGR → RGB)

Mat blob = blobFromImage(img,
                         1.0/255.0,           // 缩放因子
                         Size(224, 224),        // 目标尺寸
                         Scalar(104, 117, 123), // 均值减法
                         true,                 // 交换 R 和 B
                         false);               // 不裁剪

// YOLO 预处理 (无均值减法)
Mat blob = blobFromImage(img, 1/255.0, Size(416, 416),
                         Scalar(0, 0, 0), false, false);

// 无缩放的简单尺寸调整
Mat blob = blobFromImageNoScale(img, Size(640, 640));
```

```python
# ============================================
# Python 预处理
# ============================================
# 标准 ImageNet 预处理
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0/255.0,
    size=(224, 224),
    mean=(104, 117, 123),
    swapRB=True,
    crop=False
)

# ImageNet 预处理 (简写)
blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), None, True)

# YOLO 预处理
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), None, False)

# 无缩放
blob = cv2.dnn.blobFromImageNoScale(img, (640, 640))
```

### 4.2 后处理: 解析检测结果

```cpp
// ============================================
// C++ 后处理示例 (YOLO)
// ============================================
// 假设 YOLO 输出 [1, 255, 13, 13] 每行 85 个值 (4 bbox + 1 conf + 80 classes)

vector<String> classNames = {"person", "bicycle", "car", ...};

vector<Rect> boxes;
vector<float> confidences;
vector<int> classIds;

for (const auto& output : outputs) {
    for (int i = 0; i < output.rows; i++) {
        Mat row = output.row(i);
        Vec4f bbox = row.at<Vec4f>(0, 1, 2, 3);  // cx, cy, w, h

        // 提取类别分数
        Point classIdPoint;
        double confidence;
        minMaxLoc(row.colRange(5, 85), 0, &confidence, 0, &classIdPoint);

        if (confidence > 0.5) {
            // 转换为边界框 (x, y, w, h)
            float cx = bbox[0] * width;
            float cy = bbox[1] * height;
            float w = bbox[2] * width;
            float h = bbox[3] * height;
            int x = int(cx - w/2);
            int y = int(cy - h/2);

            boxes.push_back(Rect(x, y, int(w), int(h)));
            confidences.push_back(confidence);
            classIds.push_back(classIdPoint.y);
        }
    }
}

// 非极大值抑制
vector<int> indices;
dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

// 绘制结果
for (int idx : indices) {
    Rect box = boxes[idx];
    rectangle(frame, box, Scalar(0, 255, 0), 2);
    string label = classNames[classIds[idx]];
    putText(frame, label, Point(box.x, box.y - 10),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
}
```

```python
# ============================================
# Python 后处理示例 (YOLO)
# ============================================
class_names = ["person", "bicycle", "car", ...]

boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        # detection: [cx, cy, w, h, obj_conf, class1, class2, ...]
        obj_conf = detection[4]
        class_probs = detection[5:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id] * obj_conf

        if confidence > 0.5:
            cx, cy, w, h = detection[0:4] * np.array([width, height, width, height])
            x = int(cx - w/2)
            y = int(cy - h/2)

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 非极大值抑制
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 绘制结果
for i in indices:
    x, y, w, h = boxes[i]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.putText(frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

---

## 5.目标检测模型

### 5.1 YOLOv4 推理

```cpp
// ============================================
// C++ YOLOv4 推理
// ============================================
// 加载模型
Net net = readNetFromDarknet("yolov4.cfg", "yolov4.weights");
net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);

// 输入预处理
Mat blob = blobFromImage(frame, 1/255.0, Size(608, 608));
net.setInput(blob);

// 推理
vector<String> outNames = net.getUnconnectedOutLayersNames();
vector<Mat> outputs;
net.forward(outputs, outNames);

// 后处理...
```

```python
# ============================================
# Python YOLOv4 推理
# ============================================
# 加载模型
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

# 输入预处理
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608))
net.setInput(blob)

# 推理
ln = net.getUnconnectedOutLayersNames()
outputs = [net.forward(ln[0]), net.forward(ln[1]), net.forward(ln[2])]

# 后处理...
```

### 5.2 SSD MobileNet 推理

```cpp
// ============================================
// C++ SSD MobileNet 推理
// ============================================
Net net = readNetFromTensorflow("ssd_mobilenet_v2_coco.pbtxt",
                                "ssd_mobilenet_v2_coco.pb");

net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);

// 预处理 (MobileNet V2 通常使用 [-1, 1] 范围)
Mat blob = blobFromImage(frame, 0.007843, Size(300, 300),
                         Scalar(127.5, 127.5, 127.5), false);

net.setInput(blob);
Mat output = net.forward();

// 解析 SSD 输出格式 [1, 1, N, 7] 每行 [batch, class, confidence, x1, y1, x2, y2]
```

```python
# ============================================
# Python SSD MobileNet 推理
# ============================================
net = cv2.dnn.readNetFromTensorflow("ssd_mobilenet_v2_coco.pbtxt",
                                    "ssd_mobilenet_v2_coco.pb")

# 预处理
blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300),
                             (127.5, 127.5, 127.5), False)
net.setInput(blob)
output = net.forward()
# 输出格式 [batch, class, confidence, x1, y1, x2, y2]
```

### 5.3 EfficientDet 推理

```cpp
// ============================================
// C++ EfficientDet ONNX 推理
// ============================================
Net net = readNetFromONNX("efficientdet-d4.onnx");

// NHWC → NCHW (EfficientDet 需要 NHWC)
Mat blob = blobFromImage(frame, 1/255.0, Size(512, 512),
                         Scalar(0.485, 0.456, 0.406),  // ImageNet mean
                         Scalar(0.229, 0.224, 0.225),  // ImageNet std
                         true, false);

// 后端选择
net.setPreferableBackend(DNN_BACKEND_CUDA);
net.setPreferableTarget(DNN_TARGET_CUDA);

// 推理
net.setInput(blob);
Mat output = net.forward();
```

```python
# ============================================
# Python EfficientDet ONNX 推理
# ============================================
import numpy as np

net = cv2.dnn.readNetFromONNX("efficientdet-d4.onnx")

# EfficientDet 通常使用 ImageNet 标准化
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# 预处理
img_resized = cv2.resize(frame, (512, 512))
img_normalized = (img_resized / 255.0 - mean) / std
blob = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
blob = np.ascontiguousarray(blob, dtype=np.float32)

net.setInput(blob)
output = net.forward()
```

---

## 6.代码示例

### 6.1 图像分类 (ResNet)

```cpp
// ============================================
// C++ ResNet 图像分类
// ============================================
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // 加载模型
    Net net = readNetFromONNX("resnet50.onnx");

    // 加载 ImageNet 类别
    vector<string> classNames = ...;  // 1000 个类别

    // 读取并预处理图像
    Mat img = imread("input.jpg");
    Mat blob = blobFromImage(img, 1.0/255.0, Size(224, 224),
                             Scalar(0.485, 0.456, 0.406),
                             true, true);

    // 推理
    net.setInput(blob);
    Mat output = net.forward();

    // 解析输出
    Mat softmax;
    exp(-output, softmax);  // softmax
    softmax /= sum(softmax);

    // 获取 Top-5
    Mat sortedIdx;
    sortIdx(softmax.reshape(1, 1), sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);

    cout << "Top-5 预测:" << endl;
    for (int i = 0; i < 5; i++) {
        int classId = sortedIdx.at<int>(0, i);
        float prob = softmax.at<float>(classId);
        cout << i+1 << ". " << classNames[classId]
             << " (" << prob * 100 << "%)" << endl;
    }

    return 0;
}
```

```python
# ============================================
# Python ResNet 图像分类
# ============================================
import cv2
import numpy as np

# 加载模型
net = cv2.dnn.readNetFromONNX("resnet50.onnx")

# 读取并预处理图像
img = cv2.imread("input.jpg")
blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224),
                              (0.485, 0.456, 0.406), True, True)

# 推理
net.setInput(blob)
output = net.forward()

# Softmax
output = output.flatten()
exp_output = np.exp(output - np.max(output))
softmax = exp_output / np.sum(exp_output)

# Top-5
top5_idx = np.argsort(softmax)[-5:][::-1]
for i, idx in enumerate(top5_idx):
    print(f"{i+1}. {class_names[idx]} ({softmax[idx]*100:.2f}%)")
```

### 6.2 目标检测 (YOLOv4)

```cpp
// ============================================
// C++ YOLOv4 目标检测
// ============================================
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

void detectObjects(Net& net, Mat& frame,
                  vector<string>& classNames,
                  float confThreshold = 0.5,
                  float nmsThreshold = 0.4) {
    // 预处理
    Mat blob = blobFromImage(frame, 1/255.0, Size(608, 608));
    net.setInput(blob);

    // 推理
    vector<String> outNames = net.getUnconnectedOutLayersNames();
    vector<Mat> outputs;
    net.forward(outputs, outNames);

    // 后处理
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;

    float* data;
    int numClasses = classNames.size();

    for (const auto& output : outputs) {
        data = (float*)output.data;
        for (int i = 0; i < output.rows; i++, data += output.cols) {
            float objConf = data[4];
            if (objConf < confThreshold) continue;

            // 找最大类别分数
            Point classPoint;
            Mat scores = output.row(i).colRange(5, 5 + numClasses);
            minMaxLoc(scores, 0, 0, 0, &classPoint);
            float conf = data[classPoint.x + 5] * objConf;

            if (conf < confThreshold) continue;

            // 边界框 (归一化 → 像素)
            int cx = int(data[0] * frame.cols);
            int cy = int(data[1] * frame.rows);
            int w = int(data[2] * frame.cols);
            int h = int(data[3] * frame.rows);
            int x = int(cx - w/2);
            int y = int(cy - h/2);

            boxes.push_back(Rect(x, y, w, h));
            confidences.push_back(conf);
            classIds.push_back(classPoint.x);
        }
    }

    // NMS
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // 绘制
    for (int idx : indices) {
        Rect box = boxes[idx];
        rectangle(frame, box, Scalar(0, 255, 0), 2);

        string label = format("%s: %.2f",
            classNames[classIds[idx]].c_str(), confidences[idx]);
        putText(frame, label, Point(box.x, box.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
}
```

```python
# ============================================
# Python YOLOv4 目标检测
# ============================================
def detect_objects(net, frame, class_names,
                   conf_threshold=0.5, nms_threshold=0.4):
    # 预处理
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608))
    net.setInput(blob)

    # 推理
    layer_names = net.getUnconnected_out_layers_names()
    outputs = [net.forward(layer) for layer in layer_names]

    # 后处理
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            obj_conf = detection[4]
            if obj_conf < conf_threshold:
                continue

            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id] * obj_conf

            if conf < conf_threshold:
                continue

            # 边界框
            cx, cy, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0],
                                                        frame.shape[1], frame.shape[0]])
            x, y = int(cx - w/2), int(cy - h/2)

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(conf))
            class_ids.append(class_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # 绘制
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
```

### 6.3 实时视频检测

```cpp
// ============================================
// C++ 实时视频检测
// ============================================
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    Net net = readNetFromONNX("yolov4.onnx");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    vector<string> classNames = {"person", "car", "dog", ...};

    Mat frame;
    while (cap.read(frame)) {
        detectObjects(net, frame, classNames);
        imshow("Detection", frame);

        int key = waitKey(30);
        if (key == 'q' || key == 27) break;
    }

    destroyAllWindows();
    return 0;
}
```

```python
# ============================================
# Python 实时视频检测
# ============================================
def main():
    cap = cv2.VideoCapture(0)
    net = cv2.dnn.readNetFromONNX("yolov4.onnx")
    class_names = ["person", "car", "dog", ...]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(net, frame, class_names)
        cv2.imshow("Detection", frame)

        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
```

---

## 7.练习题

### 入门级
1. 使用 ONNX 模型进行图像分类 (C++ / Python)
2. 使用 YOLOv4 进行目标检测
3. 实现视频流实时推理

### 中级
4. 比较不同后端 (CPU/GPU) 性能
5. 实现自定义模型推理 (如 OCR)
6. 实现多模型组合 (检测 + 分类)

### 高级
7. 量化模型推理优化 (INT8)
8. 实现模型剪枝后的推理
9. 实现异步推理 pipeline

### 挑战题
10. 实现自定义层/自定义算子
11. 实现模型 ensemble (多模型融合)

---

## 8.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [dnn 模块](https://docs.opencv.org/4.14.0/d6/d0f/group__dnn.html) |
| DNN 教程 | [Deep Learning Networks](https://docs.opencv.org/4.14.0/d2/d58/tutorial__dnn__deepflow.html) |
| ONNX 模型下载 | [ONNX Model Zoo](https://github.com/onnx/models) |
| OpenCV DNN 示例 | [OpenCV Samples DNN](https://github.com/opencv/opencv/tree/master/samples/dnn) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 dnn 模块文档 (C++/Python 双语) |