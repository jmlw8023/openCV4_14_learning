# OpenCV calib3d 模块学习指南

> **模块**: calib3d
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 3 - 中级

---

## 目录

1. [概述](./README.md#1概述)
2. [相机模型](./README.md#2相机模型)
3. [单目标定](./README.md#3单目标定)
4. [双目标定与立体匹配](./README.md#4双目标定与立体匹配)
5. [PnP 问题](./README.md#5pnp-问题)
6. [代码示例](./README.md#6代码示例)
7. [练习题](./README.md#7练习题)
8. [参考资料](./README.md#8参考资料)

---

## 1.概述

**calib3d** (Camera Calibration 3D) 提供相机标定和三维重建功能:

| 功能 | 描述 |
|---------|-------------|
| **相机标定** | 内参、外参、畸变系数 |
| **畸变校正** | 去畸变 (undistort) |
| **立体视觉** | 双目匹配、深度图 |
| **PnP 问题** | 3D-2D 对应求解姿态 |
| **三维重建** | 三角测量恢复深度 |

**头文件**: `opencv2/calib3d.hpp`

---

## 2.相机模型

### 2.1 针孔相机模型

```
世界坐标系 → 相机坐标系 → 图像坐标系 → 像素坐标系
    (Rt)         (P)         (1/dx, 1/dy)     (fx, fy)
```

### 2.2 内参矩阵 (Camera Matrix)

```cpp
// C++ 内参矩阵
Mat cameraMatrix = (Mat_<double>(3, 3) <<
    fx,  0, cx,
     0, fy, cy,
     0,  0,  1);

// Python (NumPy)
# cameraMatrix = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0,  0,  1]], dtype=np.float64)
```

| 参数 | 描述 |
|------|------|
| fx, fy | 焦距 (像素单位) |
| cx, cy | 主点坐标 (光心) |
| width, height | 图像尺寸 |

### 2.3 畸变系数 (Distortion Coefficients)

```cpp
// 径向 + 切向畸变
// k1, k2, k3: 径向畸变
// p1, p2: 切向畸变

Mat distCoeffs = (Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
// 如果无畸变: Mat::zeros(5, 1)

// Python
# distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
```

### 2.4 外参矩阵 (Extrinsic Parameters)

```cpp
// 外参 = 旋转矩阵 R + 平移向量 t
// [R|t] 将世界坐标系点变换到相机坐标系

Mat rvecs, tvecs;  // 旋转向量 (罗德里格斯形式) + 平移向量

// 旋转向量转旋转矩阵
Mat R;
Rodrigues(rvecs, R);  // 罗德里格斯变换

// Python
# import cv2
# R, _ = cv2.Rodrigues(rvecs)
```

---

## 3.单目标定

### 3.1 标定板类型

| 类型 | 创建方式 | 用途 |
|------|----------|------|
| Chessboard | findChessboardCorners | 最常用 |
| ChArUco | CharucoBoard | 更精确 |
| CirclesGrid | findCirclesGrid | 圆点网格 |
| AprilTag | apriltag 库 | 高精度 |

### 3.2 标定流程

```cpp
// ============================================
// C++ 单目相机标定
// ============================================
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// 准备标定板角点 (世界坐标)
vector<vector<Point3f>> objectPoints;  // 3D 点
vector<vector<Point2f>> imagePoints; // 2D 点

Size boardSize(9, 6);  // 内部角点数
float squareSize = 20.0;  // 棋盘格物理尺寸 (mm)

// 生成世界坐标
vector<Point3f> obj;
for (int i = 0; i < boardSize.height; i++)
    for (int j = 0; j < boardSize.width; j++)
        obj.push_back(Point3f(j * squareSize, i * squareSize, 0));

// 提取角点
Mat image, gray;
vector<Point2f> corners;

for (const auto& img : calibrationImages) {
    image = imread(img);
    cvtColor(image, gray, COLOR_BGR2GRAY);

    bool found = findChessboardCorners(gray, boardSize, corners);

    if (found) {
        // 亚像素精度优化
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001);
        cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1), criteria);

        objectPoints.push_back(obj);
        imagePoints.push_back(corners);
    }
}

// 标定
Mat cameraMatrix, distCoeffs;
vector<Mat> rvecs, tvecs;

double rms = calibrateCamera(objectPoints, imagePoints,
                           image.size(),  // 或 Size(w, h)
                           cameraMatrix, distCoeffs,
                           rvecs, tvecs);

cout << "RMS 误差: " << rms << endl;
cout << "内参矩阵:\n" << cameraMatrix << endl;
cout << "畸变系数:\n" << distCoeffs << endl;

// 保存标定结果
FileStorage fs("calibration.yml", FileStorage::WRITE);
fs << "cameraMatrix" << cameraMatrix;
fs << "distCoeffs" << distCoeffs;
fs.release();
```

```python
# ============================================
# Python 单目相机标定
# ============================================
import numpy as np
import cv2

# 准备标定板角点 (世界坐标)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20.0  # 20mm 方格

objpoints = []  # 3D 点
imgpoints = []  # 2D 点

for img_path in calibration_images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print(f"RMS 误差: {ret}")
print(f"内参矩阵:\n{mtx}")
print(f"畸变系数:\n{dist}")

# 保存
np.savez('calibration.npz', mtx=mtx, dist=dist)
```

### 3.3 畸变校正

```cpp
// ============================================
// C++ 畸变校正
// ============================================
#include <opencv2/calib3d.hpp>

// 方法1: undistort (一步完成)
Mat undistorted;
undistort(distorted, undistorted, cameraMatrix, distCoeffs);

// 方法2: 获取映射表 + remap (更适合视频处理)
Mat map1, map2;
initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                        cameraMatrix, image.size(), CV_32FC2, map1, map2);

Mat undistorted;
remap(distorted, undistorted, map1, map2, INTER_LINEAR);

// 获取无畸变图像的 ROI
Mat newCameraMatrix;
Rect roi = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,
                                     image.size(), 1, image.size());

// Python
# undistorted = cv2.undistort(distorted, mtx, dist)
# map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC2)
# undistorted = cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR)
```

---

## 4.双目标定与立体匹配

### 4.1 双目相机模型

```
左目                    右目
   \                    /
    \                  /
     Z ------> Y (baseline)
    /
   X
```

### 4.2 双目标定

```cpp
// ============================================
// C++ 双目标定
// ============================================
#include <opencv2/calib3d.hpp>

// 双目标定 (与单目类似, 但同时处理左右相机)
vector<vector<Point3f>> objectPoints;
vector<vector<Point2f>> leftImagePoints, rightImagePoints;

// ... 提取角点 ...

// 双目标定
Mat M1, D1, M2, D2, R, T, E, F;
double rms = stereoCalibrate(objectPoints,
                             leftImagePoints, rightImagePoints,
                             M1, D1, M2, D2,
                             imageSize, R, T, E, F,
                             CALIB_FIX_INTRINSIC);  // 固定内参

cout << "旋转矩阵 R:\n" << R << endl;
cout << "平移向量 T:\n" << T << endl;
cout << "本质矩阵 E:\n" << E << endl;
cout << "基础矩阵 F:\n" << F << endl;
```

```python
# ============================================
# Python 双目标定
# ============================================
# 标定左右相机
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, leftImgpoints, gray.shape[::-1], None, None)

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, rightImgpoints, gray.shape[::-1], None, None)

# 双目标定
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, leftImgpoints, rightImgpoints,
    mtxL, distL, mtxR, distR, gray.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC)

print(f"旋转矩阵 R:\n{R}")
print(f"平移向量 T:\n{T}")
```

### 4.3 立体匹配与深度图

```cpp
// ============================================
// C++ 立体匹配
// ============================================
#include <opencv2/calib3d.hpp>

// 计算校正映射
Mat R1, R2, Q;
stereoRectify(M1, D1, M2, D2, imageSize, R, T,
             R1, R2, Q, CALIB_ZERO_DISPARITY,
             -1, imageSize);

// 初始化校正映射
Mat lmap1, lmap2, rmap1, rmap2;
initUndistortRectifyMap(M1, D1, R1, Q, imageSize, CV_32FC1, lmap1, lmap2);
initUndistortRectifyMap(M2, D2, R2, Q, imageSize, CV_32FC1, rmap1, rmap2);

// 校正图像
Mat leftRect, rightRect;
remap(left, leftRect, lmap1, lmap2, INTER_LINEAR);
remap(right, rightRect, rmap1, rmap2, INTER_LINEAR);

// 创建 StereoBM / StereoSGBM
Ptr<StereoBM> stereoBM = StereoBM::create(16 * 5, 21);
Ptr<StereoSGBM> stereoSGBM = StereoSGBM::create(0, 16 * 5, 21);

// 计算视差图
Mat disparity;
stereoBM->compute(leftRect, rightRect, disparity);

// 转换视差图到深度图
Mat depth;
reprojectImageTo3D(disparity, depth, Q, true);
```

```python
# ============================================
# Python 立体匹配
# ============================================
# 校正
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, imageSize, R, T)

# 校正映射
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, imageSize, cv2.CV_32FC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, imageSize, cv2.CV_32FC2)

# 校正图像
leftRect = cv2.remap(left, mapL1, mapL2, cv2.INTER_LINEAR)
rightRect = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)

# 计算视差图
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
# 或 StereoSGBM
# stereo = cv2.StereoSGBM_create(0, 16*5, 21)

disparity = stereo.compute(leftRect, rightRect)

# 视差图转深度图
depth = cv2.reprojectImageTo3D(disparity, Q)
```

---

## 5.PnP 问题

### 5.1 PnP 概述

PnP (Perspective-n-Point) 通过已知的 3D-2D 对应点求解相机外参 (R, t):

```
已知: 3D 点 (世界坐标系) + 2D 点 (图像坐标系) + 内参
求解: 相机外参 (R, t)
```

### 5.2 PnP 算法

```cpp
// ============================================
// C++ PnP 求解
// ============================================
// 3D 点 (世界坐标系, 单位: mm)
vector<Point3f> objectPoints = {
    Point3f(0, 0, 0),
    Point3f(100, 0, 0),
    Point3f(100, 100, 0),
    Point3f(0, 100, 0)
};

// 2D 点 (像素坐标系, 从检测得到)
vector<Point2f> imagePoints = {
    Point2f(u1, v1),
    Point2f(u2, v2),
    Point2f(u3, v3),
    Point2f(u4, v4)
};

// 内参和畸变 (已标定)
Mat cameraMatrix, distCoeffs;

// PnP 求解
Mat rvec, tvec;
solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
         rvec, tvec, false, SOLVEPNP_ITERATIVE);
// 还有: SOLVEPNP_P3P, SOLVEPNP_EPNP, SOLVEPNP_AP3P, SOLVEPNP_SQPNP

// 旋转向量转旋转矩阵
Mat R;
Rodrigues(rvec, R);

// 计算相机坐标系中的 3D 点
vector<Point3f> points3D_cam;
projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs,
              points3D_cam);
```

```python
# ============================================
# Python PnP 求解
# ============================================
# 3D 点 (世界坐标系)
objp = np.array([
    [0, 0, 0],
    [100, 0, 0],
    [100, 100, 0],
    [0, 100, 0]
], dtype=np.float32)

# 2D 点 (从检测得到)
imgp = np.array([
    [u1, v1],
    [u2, v2],
    [u3, v3],
    [u4, v4]
], dtype=np.float32)

# PnP 求解
ret, rvec, tvec = cv2.solvePnP(
    objp, imgp, mtx, dist,
    flags=cv2.SOLVEPNP_ITERATIVE)

# 旋转向量转旋转矩阵
R, _ = cv2.Rodrigues(rvec)

# 验证重投影误差
imgp_reproj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
error = cv2.norm(imgp, imgp_reproj, cv2.NORM_L2)
```

---

## 6.代码示例

### 6.1 完整标定流程

```cpp
// ============================================
// C++ 完整标定流程
// ============================================
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

class CameraCalibrator {
public:
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    double rms;

    void addCalibrationPoints(const vector<Point3f>& obj,
                               const vector<Point2f>& img,
                               Size imageSize) {
        objectPoints.push_back(obj);
        imagePoints.push_back(img);
        this->imageSize = imageSize;
    }

    double calibrate() {
        rms = calibrateCamera(objectPoints, imagePoints,
                             imageSize, cameraMatrix, distCoeffs,
                             rvecs, tvecs, CALIB_FIX_PRINCIPAL_POINT |
                             CALIB_USE_INTRINSIC_GUESS);
        return rms;
    }

    Mat undistort(const Mat& img) {
        Mat undistorted;
        undistort(img, undistorted, cameraMatrix, distCoeffs);
        return undistorted;
    }

private:
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    Size imageSize;
};
```

```python
# ============================================
# Python 完整标定流程
# ============================================
class CameraCalibrator:
    def __init__(self):
        self.cameraMatrix = None
        self.distCoeffs = None
        self.rvecs = None
        self.tvecs = None
        self.rms = None
        self.objpoints = []
        self.imgpoints = []

    def add_points(self, objp, imgp):
        self.objpoints.append(objp)
        self.imgpoints.append(imgp)

    def calibrate(self, image_size):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None)
        self.cameraMatrix = mtx
        self.distCoeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.rms = ret
        return ret

    def undistort(self, img):
        return cv2.undistort(img, self.cameraMatrix, self.distCoeffs)
```

### 6.2 实时姿态估计

```cpp
// ============================================
// C++ AR 姿态估计
// ============================================
// 标定板参数
Size boardSize(9, 6);
float squareSize = 20.0;

// 假设已标定得到 cameraMatrix, distCoeffs
// 加载标定结果
FileStorage fs("calibration.yml", FileStorage::READ);
fs["cameraMatrix"] >> cameraMatrix;
fs["distCoeffs"] >> distCoeffs;

// AR 标记角点 (世界坐标, 与标定板相同)
vector<Point3f> objp;
for (int i = 0; i < boardSize.height; i++)
    for (int j = 0; j < boardSize.width; j++)
        objp.push_back(Point3f(j * squareSize, i * squareSize, 0));

// 实时处理
VideoCapture cap(0);
Mat frame, gray;

while (cap >> frame) {
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    vector<Point2f> corners;
    bool found = findChessboardCorners(gray, boardSize, corners);

    if (found) {
        // 亚像素优化
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001);
        cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1), criteria);

        // PnP 求解
        Mat rvec, tvec;
        solvePnP(objp, corners, cameraMatrix, distCoeffs,
                 rvec, tvec, false);

        // 绘制轴
        vector<Point3f> axis = {Point3f(0,0,0), Point3f(3*squareSize,0,0),
                               Point3f(0,3*squareSize,0), Point3f(0,0,-3*squareSize)};
        vector<Point2f> axisImg;
        projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs, axisImg);

        line(frame, corners[0], axisImg[0], Scalar(0, 0, 255), 3);  // X 轴 (红)
        line(frame, corners[0], axisImg[1], Scalar(0, 255, 0), 3);  // Y 轴 (绿)
        line(frame, corners[0], axisImg[2], Scalar(255, 0, 0), 3);  // Z 轴 (蓝)
    }

    imshow("Pose Estimation", frame);
    if (waitKey(30) == 'q') break;
}
```

```python
# ============================================
# Python AR 姿态估计
# ============================================
# AR 标记角点 (世界坐标)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20.0

# 实时处理
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        # 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # PnP 求解
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

        # 绘制轴
        if ret:
            axis = np.float32([[0,0,0], [60,0,0], [0,60,0], [0,0,-60]])
            imgp, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
            imgp = imgp.astype(int)

            cv2.line(frame, tuple(imgp[0][0]), tuple(imgp[1][0]), (0,0,255), 3)  # X
            cv2.line(frame, tuple(imgp[0][0]), tuple(imgp[2][0]), (0,255,0), 3)  # Y
            cv2.line(frame, tuple(imgp[0][0]), tuple(imgp[3][0]), (255,0,0), 3)  # Z

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 7.练习题

### 入门级
1. 使用棋盘格进行单目相机标定 (C++ / Python)
2. 实现图像去畸变校正
3. 比较不同 PnP 算法 (ITERATIVE, P3P, EPNP)

### 中级
4. 实现双目标定和立体校正
5. 计算并可视化深度图
6. 实现基于棋盘格的实时姿态估计

### 高级
7. 使用 ChArUco 标定板进行高精度标定
8. 实现基于 ArUco 的 AR 应用
9. 实现多相机系统标定

### 挑战题
10. 实现 SfM (Structure from Motion) 系统
11. 实现基于深度相机的三维重建

---

## 8.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [calib3d 模块](https://docs.opencv.org/4.14.0/d9/d0c/group__calib3d.html) |
| 相机标定教程 | [Camera Calibration](https://docs.opencv.org/4.14.0/d4/d94/tutorial_camera_calibration.html) |
| calibrateCamera | [calibrateCamera](https://docs.opencv.org/4.14.0/d9/d0c/group__calib3d.html#gaad3e4d6c4e94c5e2d3c23d3ad1f47826) |
| solvePnP | [solvePnP](https://docs.opencv.org/4.14.0/d9/d0c/group__calib3d.html#ga549c2efb20b6e35d1ef1bf8433d4a5c2) |
| stereoCalibrate | [stereoCalibrate](https://docs.opencv.org/4.14.0/d9/d0c/group__calib3d.html#gaf364d2b8cb5b4b650f6f71e70c70a2f4) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 calib3d 模块文档 (C++/Python 双语) |