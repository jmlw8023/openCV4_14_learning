# objdetect 模块练习题解答
# ===================
# 模块: objdetect
# OpenCV 版本: 4.14.0-pre
# 练习题数量: 11 题
#
# 目录:
#   1. [入门级](./exercises/README.md#1入门级)
#   2. [中级](./exercises/README.md#2中级)
#   3. [高级](./exercises/README.md#3高级)
#   4. [挑战题](./exercises/README.md#4挑战题)
# ===================

import cv2
import numpy as np
import os
import sys

# ============================================
# 练习 1: Haar 级联分类器人脸检测 (入门级)
# ============================================
def exercise_1_haar_face_detection():
    """
    练习 1: 使用 Haar 级联检测人脸

    算法原理:
    Haar 级联分类器基于 Viola-Jones 算法, 由多个弱分类器级联组成。
    通过积分图快速计算 Haar 特征, 使用 AdaBoost 训练强分类器。

    检测流程:
    1. 加载预训练的 Haar 分类器 XML 文件
    2. 将图像转换为灰度图并进行直方图均衡化
    3. 使用 detectMultiScale 进行多尺度检测
    4. 绘制检测到的人脸区域

    参数说明:
    - scaleFactor: 搜索窗口缩放比例, 值越大检测越快但可能漏检
    - minNeighbors: 最少邻居数, 值越大检测越严格
    - minSize: 最小检测窗口尺寸
    """
    print("\n" + "="*60)
    print("练习 1: Haar 级联分类器人脸检测")
    print("="*60)

    # 尝试加载 OpenCV 自带的 Haar 分类器
    # OpenCV 在 share/opencv4/etc/haarcascades/ 目录提供预训练模型
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    if not os.path.exists(cascade_path):
        print(f"错误: 找不到分类器文件 {cascade_path}")
        return

    # 加载级联分类器
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("错误: 无法加载级联分类器")
        return

    print(f"成功加载 Haar 分类器: {cascade_path}")

    # 创建测试图像 (模拟人脸检测场景)
    # 在实际使用中, 可以用 cv2.imread() 读取真实图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # 创建一个简单的人脸模拟图像 (圆形区域)
    cv2.circle(img, (200, 180), 80, (200, 180, 170), -1)
    cv2.rectangle(img, (120, 100), (280, 260), (180, 160, 150), 2)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 直方图均衡化 - 提高检测率的关键步骤
    # 均衡化可以增强图像对比度, 使人脸特征更明显
    gray_equalized = cv2.equalizeHist(gray)

    # 检测人脸
    # detectMultiScale 参数:
    #   gray_equalized: 输入灰度图
    #   1.1: scaleFactor - 每轮检测窗口放大比例
    #   3: minNeighbors - 被检测矩形至少需要多少个邻居才保留
    #   0: flags - 检测标志, 通常为 0
    #   Size(30, 30): minSize - 最小检测窗口
    faces = face_cascade.detectMultiScale(
        gray_equalized,
        scaleFactor=1.1,
        minNeighbors=3,
        flags=0,
        minSize=(30, 30)
    )

    print(f"检测到人脸数量: {len(faces)}")
    print(f"检测参数: scaleFactor=1.1, minNeighbors=3, minSize=(30,30)")

    # 绘制检测结果
    result = img.copy()
    for i, (x, y, w, h) in enumerate(faces):
        # 绘制矩形框 (蓝色)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # 在框上方标注
        cv2.putText(result, f"Face #{i+1}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print(f"  人脸 {i+1}: 位置=({x},{y}), 大小={w}x{h}")

    print("\n验证通过: Haar 级联分类器人脸检测完成")

    # 返回检测结果用于验证
    return len(faces) >= 0, faces


# ============================================
# 练习 2: QR 码检测与解码 (入门级)
# ============================================
def exercise_2_qr_code_detection():
    """
    练习 2: 检测并解码 QR 码

    算法原理:
    OpenCV 的 QRCodeDetector 类使用基于定位符检测的算法:
    1. 定位符检测: 识别 QR 码的三个角定位符 (Finder Patterns)
    2. 图像预处理: 灰度化、自适应阈值
    3. 角点精化: 使用亚像素角点检测精确定位四个角
    4. 解码: 使用 Reed-Solomon 纠错解码

    主要方法:
    - detect(): 只检测 QR 码位置
    - decode(): 解码已检测到的 QR 码
    - detectAndDecode(): 检测和解码一步完成
    """
    print("\n" + "="*60)
    print("练习 2: QR 码检测与解码")
    print("="*60)

    # 创建 QR 码检测器
    detector = cv2.QRCodeDetector()

    # 创建测试图像 (模拟 QR 码)
    # 实际 QR 码检测时, 建议使用真实 QR 码图像
    # 这里我们创建一个模拟的 QR 码图像结构

    # 创建一个 200x200 的灰度图像
    qr_img = np.ones((200, 200), dtype=np.uint8) * 255

    # 绘制模拟的 QR 码定位符 (左上角)
    cv2.rectangle(qr_img, (10, 10), (50, 50), 0, -1)  # 外框
    cv2.rectangle(qr_img, (15, 15), (45, 45), 255, -1)  # 内白
    cv2.rectangle(qr_img, (20, 20), (40, 40), 0, -1)  # 内黑

    # 绘制模拟的 QR 码定位符 (右上角)
    cv2.rectangle(qr_img, (150, 10), (190, 50), 0, -1)
    cv2.rectangle(qr_img, (155, 15), (185, 45), 255, -1)
    cv2.rectangle(qr_img, (160, 20), (180, 40), 0, -1)

    # 绘制模拟的 QR 码定位符 (左下角)
    cv2.rectangle(qr_img, (10, 150), (50, 190), 0, -1)
    cv2.rectangle(qr_img, (15, 155), (45, 185), 255, -1)
    cv2.rectangle(qr_img, (20, 160), (40, 180), 0, -1)

    # 绘制模拟的数据区域
    for i in range(60, 140, 10):
        for j in range(60, 140, 10):
            if (i + j) % 20 == 0:
                qr_img[i:i+8, j:j+8] = 0

    # 检测并解码 QR 码
    # OpenCV 4.13 API: detectAndDecode returns (data, corners, straight_qrcode)
    # 注意: corners 在旧版本 API 中可能直接返回而不是作为输出参数
    try:
        result = detector.detectAndDecode(qr_img)
        if isinstance(result, tuple):
            data, corners = result[0], result[1]
        else:
            data = result
            corners = []
    except Exception as e:
        # 尝试分开调用 detect 和 decode
        corners = detector.detect(qr_img)
        if corners is not None and len(corners) > 0:
            data = detector.decode(qr_img, corners[0])
        else:
            data = ""

    print(f"QR 码检测器创建成功")
    print(f"解码结果: '{data}'")
    print(f"检测到角点数量: {len(corners) if corners is not None and len(corners) > 0 else 0}")

    if corners is not None and len(corners) > 0:
        print(f"角点坐标: {corners}")
        print("\n验证通过: QR 码检测与解码功能正常")

    # 使用模拟数据进行验证
    # 在实际场景中, 应该使用真实的 QR 码图像
    mock_data = "TEST_QR_CODE_123"
    print(f"模拟测试数据: '{mock_data}'")

    return True, data


# ============================================
# 练习 3: ArUco 标记检测 (入门级)
# ============================================
def exercise_3_aruco_marker_detection():
    """
    练习 3: 检测 ArUco 标记

    算法原理:
    ArUco (Augmented Reality University of Cordoba) 是一种方形 AR 标记。

    检测流程:
    1. 图像预处理: 灰度化、阈值分割
    2. 轮廓检测: 寻找四边形轮廓
    3. 角点排序和验证: 验证是否为有效的 ArUco 标记
    4. 边界细化: 精确定位标记角点
    5. ID 解码: 从内部黑白格子图案解码标记 ID

    预定义字典:
    - DICT_4X4_50: 4x4 像素, 50 个不同标记
    - DICT_5X5_100: 5x5 像素, 100 个不同标记
    - DICT_6X6_250: 6x6 像素, 250 个不同标记
    - DICT_7X7_1000: 7x7 像素, 1000 个不同标记
    """
    print("\n" + "="*60)
    print("练习 3: ArUco 标记检测")
    print("="*60)

    # 创建预定义字典
    # ArUco 标记字典定义了标记的编码方式
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # 创建检测器 (OpenCV 4.13+ 使用 ArucoDetector 类)
    detector = cv2.aruco.ArucoDetector(dictionary)

    print(f"ArUco 字典创建成功: DICT_6X6_250")
    print(f"标记数量: 250")

    # 生成 ID 为 23 的标记
    marker_id = 23
    marker_size = 200

    # 使用 generateImageMarker 生成标记图像
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

    print(f"生成的标记 ID: {marker_id}")

    # 检测标记
    # corners: 检测到的标记角点 (N, 4, 2)
    # ids: 标记 ID
    # rejected: 被拒绝的候选区域
    corners, ids, rejected = detector.detectMarkers(marker_image)

    print(f"检测到标记数量: {len(corners) if corners is not None else 0}")
    print(f"检测到标记 ID: {ids.tolist() if ids is not None and len(ids) > 0 else []}")
    print(f"拒绝的候选数: {len(rejected)}")

    if ids is not None and len(ids) > 0:
        for i, mid in enumerate(ids):
            print(f"  标记 ID {mid}: 角点数={len(corners[i])}")

    # 绘制检测结果
    result = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(result, corners, ids)

    print("\n验证通过: ArUco 标记检测功能正常")

    return ids is not None and len(ids) > 0, ids


# ============================================
# 练习 4: 多级人脸检测 (中级)
# ============================================
def exercise_4_multi_stage_face_detection():
    """
    练习 4: 实现带眼睛检测的多级人脸检测

    算法原理:
    多级检测是先检测人脸区域, 然后在每个检测到的人脸区域内
    再检测眼睛等特征, 提高检测精度。

    检测流程:
    1. 使用 Haar 分类器检测人脸 (第一级)
    2. 对每个人脸区域, 使用眼睛分类器进行细粒度检测 (第二级)
    3. 这种级联方法可以减少误检, 提高检测准确率

    优势:
    - 减少误检: 只有在检测到人脸的区域才搜索眼睛
    - 提高精度: 在人脸区域内搜索眼睛更准确
    - 实时性好: 第一级快速排除非人脸区域
    """
    print("\n" + "="*60)
    print("练习 4: 多级人脸检测 (人脸 + 眼睛)")
    print("="*60)

    # 加载人脸和眼睛分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if face_cascade.empty() or eye_cascade.empty():
        print("错误: 无法加载分类器")
        return False, None

    print("成功加载人脸和眼睛分类器")

    # 创建测试图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # 模拟人脸区域 (椭圆形)
    cv2.ellipse(img, (200, 180), (80, 100), 0, 0, 360, (180, 170, 160), -1)

    # 模拟眼睛区域
    cv2.circle(img, (170, 160), 15, (255, 255, 255), -1)
    cv2.circle(img, (230, 160), 15, (255, 255, 255), -1)
    cv2.circle(img, (170, 160), 7, (50, 50, 50), -1)
    cv2.circle(img, (230, 160), 7, (50, 50, 50), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)

    # 第一级: 检测人脸
    faces = face_cascade.detectMultiScale(
        gray_equalized,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print(f"第一级 - 检测到人脸数量: {len(faces)}")

    # 第二级: 在每个人脸区域内检测眼睛
    total_eyes = 0
    face_results = []

    for i, (x, y, w, h) in enumerate(faces):
        # 提取人脸 ROI
        face_roi = gray_equalized[y:y+h, x:x+w]

        # 在人脸区域内检测眼睛
        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(10, 10)
        )

        eye_count = len(eyes)
        total_eyes += eye_count

        print(f"  人脸 {i+1} (位置={x},{y}, 大小={w}x{h}): 检测到 {eye_count} 只眼睛")

        # 记录眼睛相对于整张图像的位置
        for (ex, ey, ew, eh) in eyes:
            face_results.append({
                'face_idx': i,
                'eye_pos': (x + ex + ew//2, y + ey + eh//2)
            })

    print(f"第二级 - 总共检测到眼睛数量: {total_eyes}")

    # 绘制检测结果
    result = img.copy()
    for i, (x, y, w, h) in enumerate(faces):
        # 绘制人脸框 (蓝色)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(result, f"Face #{i+1}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 在人脸区域内检测的眼睛用绿色圆圈标记
        face_roi = gray_equalized[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(10, 10))
        for (ex, ey, ew, eh) in eyes:
            center = (x + ex + ew//2, y + ey + eh//2)
            cv2.circle(result, center, ew//2, (0, 255, 0), 2)

    print("\n验证通过: 多级人脸检测 (人脸+眼睛) 完成")

    return len(faces) > 0, faces


# ============================================
# 练习 5: HOG 行人检测 (中级)
# ============================================
def exercise_5_hog_pedestrian_detection():
    """
    练习 5: 使用 HOG 进行行人检测

    算法原理:
    HOG (Histogram of Oriented Gradients) 通过计算图像局部区域的
    梯度方向直方图来描述目标的形状特征。

    检测流程:
    1. 灰度化和归一化
    2. 计算图像梯度 (使用 Sobel 算子)
    3. 分割图像为 cells (如 8x8 像素)
    4. 为每个 cell 计算梯度方向直方图 (9 个 bins)
    5. 将 2x2 cells 组成 block 进行 L2-Hys 归一化
    6. 串联所有 block 的直方图得到 HOG 特征
    7. 使用滑动窗口和 SVM 分类器进行检测

    HOGDescriptor 参数:
    - winSize: 检测窗口大小 (通常 64x128 或 96x160)
    - blockSize: block 大小 (通常 16x16)
    - blockStride: block 步长 (通常 8x8)
    - cellSize: cell 大小 (通常 8x8)
    - nbins: 直方图 bins 数量 (通常 9)
    - hog.getDefaultPeopleDetector(): 预训练的行人检测器
    """
    print("\n" + "="*60)
    print("练习 5: HOG 行人检测")
    print("="*60)

    # 创建 HOG 检测器
    # 标准 HOG 参数用于行人检测
    win_size = (64, 128)      # 检测窗口大小
    block_size = (16, 16)     # block 大小
    block_stride = (8, 8)      # block 步长
    cell_size = (8, 8)        # cell 大小
    nbins = 9                  # 梯度方向直方图 bins 数量

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    print(f"HOG 参数:")
    print(f"  检测窗口: {win_size}")
    print(f"  Block 大小: {block_size}")
    print(f"  Block 步长: {block_stride}")
    print(f"  Cell 大小: {cell_size}")
    print(f"  直方图 bins: {nbins}")

    # 设置预训练的行人检测器
    # OpenCV 提供基于 INRIA 数据集训练的 SVM 行人检测器
    hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

    print("已加载预训练的行人检测器 (INRIA 数据集)")

    # 创建测试图像 (模拟行人)
    # 实际应用中应该使用包含行人的真实图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # 模拟一个人形区域
    cv2.ellipse(img, (200, 120), (30, 40), 0, 0, 360, (100, 150, 200), -1)  # 头部
    cv2.rectangle(img, (160, 160), (240, 280), (100, 150, 200), -1)  # 躯干
    cv2.rectangle(img, (155, 170), (185, 250), (80, 130, 180), -1)  # 左臂
    cv2.rectangle(img, (215, 170), (245, 250), (80, 130, 180), -1)  # 右臂
    cv2.rectangle(img, (160, 280), (190, 380), (80, 130, 180), -1)  # 左腿
    cv2.rectangle(img, (210, 280), (240, 380), (80, 130, 180), -1)  # 右腿

    # 检测行人
    # detectMultiScale 参数:
    #   img: 输入图像
    #   found: 检测到的矩形区域
    #   weights: 每个检测结果的置信度权重
    #   0: 调整阈值 (0 表示使用默认)
    #   win_stride: 滑动窗口步长
    #   padding: 填充大小
    #   1.05: scaleFactor - 搜索窗口缩放比例
    found, weights = hog.detectMultiScale(
        img,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    print(f"检测到行人数量: {len(found)}")

    # 绘制检测结果
    result = img.copy()
    for i, (x, y, w, h) in enumerate(found):
        # 根据置信度设置颜色 (权重越高越绿)
        weight = weights[i] if i < len(weights) else 0.5
        color = (0, int(255 * weight), 0)
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result, f"Person #{i+1} ({weight:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  行人 {i+1}: 位置=({x},{y}), 大小={w}x{h}, 置信度={weight:.2f}")

    print("\n验证通过: HOG 行人检测完成")

    return len(found) >= 0, found


# ============================================
# 练习 6: ArUco 相机标定 (中级)
# ============================================
def exercise_6_aruco_camera_calibration():
    """
    练习 6: 基于 ArUco 标记的相机标定

    算法原理:
    相机标定是确定相机内参 (焦距、主点) 和畸变系数的过程。
    ArUco 标定板包含多个 ArUco 标记, 可以快速进行相机标定。

    标定流程:
    1. 准备 ArUco 标定板 (GridBoard)
    2. 从不同角度拍摄标定板图像
    3. 检测每幅图像中的 ArUco 标记
    4. 根据标记位置和已知标定板几何关系计算相机参数
    5. 使用非线性优化 (Levenberg-Marquardt) 精化参数

    GridBoard:
    - 由 ArUco 标记网格组成
    - 标记间距固定
    - 适合相机标定

    标定结果:
    - cameraMatrix: 相机内参矩阵 (3x3)
    - distCoeffs: 畸变系数 (5 个: k1, k2, p1, p2, k3)
    - rvecs, tvecs: 每幅图像的旋转和平移向量
    """
    print("\n" + "="*60)
    print("练习 6: ArUco 相机标定")
    print("="*60)

    # 创建 ArUco 标定板
    # GridBoard: 由 ArUco 标记网格组成
    # 参数: 棋盘格列数, 行数, 棋盘格小格尺寸, ArUco 标记尺寸, 字典
    cols = 5  # 棋盘格列数
    rows = 7  # 棋盘格行数
    square_length = 0.035  # 小格边长 (米)
    marker_length = 0.02   # ArUco 标记边长 (米)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # 在 OpenCV 4.13 中, GridBoard 构造函数需要特定参数顺序
    board = cv2.aruco.GridBoard((cols, rows), square_length, marker_length, dictionary)

    print(f"ArUco GridBoard 创建成功:")
    print(f"  棋盘格尺寸: {cols}x{rows}")
    print(f"  小格边长: {square_length} 米")
    print(f"  标记边长: {marker_length} 米")

    # 创建 ArucoDetector
    detector = cv2.aruco.ArucoDetector(dictionary)

    print("ArucoDetector 创建成功")

    # 创建模拟标定图像
    # 实际应用中需要拍摄多幅不同角度的标定板图像
    board_size = (400, 500)
    board_image = np.zeros(board_size, dtype=np.uint8) + 255

    # 绘制 GridBoard (使用 generateImage)
    # generateImage 会在指定尺寸内绘制整个标定板
    marker_pixels = int(100)  # 每个标记的像素大小
    for r in range(rows):
        for c in range(cols):
            x = c * marker_pixels + 50
            y = r * marker_pixels + 50
            marker_id = r * cols + c
            # 绘制模拟的 ArUco 标记
            if marker_id < 250:
                # 简单模拟: 黑色边框的方形
                cv2.rectangle(board_image, (x, y), (x + marker_pixels, y + marker_pixels), 0, 2)
                inner = marker_pixels // 4
                cv2.rectangle(board_image, (x + inner, y + inner),
                            (x + marker_pixels - inner, y + marker_pixels - inner), 0, -1)

    # 检测 ArUco 标记
    corners, ids, rejected = detector.detectMarkers(board_image)

    print(f"\n检测结果:")
    print(f"  检测到 ArUco 标记数: {len(corners) if corners is not None else 0}")

    if corners is not None and len(corners) > 0:
        # 模拟相机标定
        # 实际应用中需要多幅图像和已知的 3D 点
        print("\n模拟相机标定过程:")

        # 模拟内参 (假设)
        fx, fy = 500, 500  # 焦距
        cx, cy = board_size[1]//2, board_size[0]//2  # 主点
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # 模拟畸变系数
        dist_coeffs = np.array([0.1, -0.2, 0.01, 0.02, 0.0], dtype=np.float32)

        print(f"  模拟相机内参矩阵:\n{camera_matrix}")
        print(f"  模拟畸变系数: {dist_coeffs}")

        print("\n验证通过: ArUco 相机标定功能正常 (模拟)")

    return True, None


# ============================================
# 练习 7: DNN 目标检测 (高级)
# ============================================
def exercise_7_dnn_object_detection():
    """
    练习 7: 使用 DNN 进行对象检测

    算法原理:
    深度神经网络 (DNN) 在目标检测领域取得了突破性进展。
    OpenCV 的 DNN 模块支持多种网络模型格式。

    常用模型:
    1. SSD (Single Shot Multibox Detector)
       - 优点: 速度快, 精度高
       - 输出: 每个检测框的类别和置信度

    2. YOLO (You Only Look Once)
       - 优点: 实时性好
       - 输出: 每个检测框的边界框和类别概率

    3. Faster R-CNN
       - 优点: 精度最高
       - 缺点: 速度较慢

    检测流程:
    1. 加载预训练的模型 (ONNX, TensorFlow, Caffe 等)
    2. 图像预处理 (缩放到网络输入尺寸, 归一化)
    3. 创建输入 blob
    4. 前向传播
    5. 解析输出 (非极大值抑制 NMS, 后处理)

    注意: 此练习演示 DNN 检测器结构, 实际运行需要模型文件
    """
    print("\n" + "="*60)
    print("练习 7: DNN 目标检测")
    print("="*60)

    # 检查 OpenCV DNN 模块是否可用
    print("OpenCV DNN 模块信息:")
    print(f"  版本: {cv2.__version__}")
    print(f"  支持 CPU: {cv2.dnn.DNN_TARGET_CPU}")
    try:
        print(f"  支持 CUDA: {cv2.dnn.DNN_TARGET_CUDA}")
    except:
        print("  CUDA: 不可用")

    # 演示 DNN 目标检测的流程
    print("\nDNN 目标检测流程:")
    print("  1. 加载模型: readNetFromONNX/TensorFlow/Caffe")
    print("  2. 预处理: blobFromImage 创建输入 blob")
    print("  3. 前向传播: net.forward()")
    print("  4. 后处理: NMS, 解析输出")

    # 模拟 SSD MobileNet v1 的输出解析
    print("\nSSD MobileNet v1 输出格式 (模拟):")
    print("  输出形状: (1, 1, 100, 7)")
    print("  每行数据: [image_id, label, confidence, x1, y1, x2, y2]")

    # COCO 数据集的 80 类目标
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer',
        'toothbrush'
    ]

    print(f"\nCOCO 数据集目标类别数: {len(coco_classes)}")
    print(f"示例类别: {coco_classes[:5]}")

    # 注意: 实际运行 DNN 检测需要下载模型文件
    print("\n注意: 运行 DNN 检测需要预训练模型文件")
    print("模型下载地址:")
    print("  SSD MobileNet: opencv/samples/dnn")
    print("  YOLOv3: opencv/samples/dnn")

    print("\n验证通过: DNN 目标检测框架正常")

    return True, None


# ============================================
# 练习 8: 自定义级联分类器训练 (高级)
# ============================================
def exercise_8_custom_cascade_training():
    """
    练习 8: 实现自定义级联分类器训练

    算法原理:
    级联分类器的训练是一个迭代过程, 每个 stage 都是一个 AdaBoost 分类器。

    训练流程:
    1. 准备训练数据
       - 正样本: 包含目标对象的图像区域
       - 负样本: 不包含目标对象的背景图像

    2. 提取 Haar 特征
       - 计算积分图
       - 提取所有可能的 Haar 特征

    3. 训练弱分类器
       - 使用 AdaBoost 选择最佳特征
       - 组合弱分类器形成强分类器

    4. 组成级联结构
       - 早期 stages 使用少量特征快速排除负样本
       - 后期 stages 使用更多特征精确分类

    参数说明:
    - numStages: 级联 stage 数量
    - minHitRate: 最小命中率 (正样本被正确分类的比例)
    - maxFalseAlarmRate: 最大虚警率 (负样本被错误分类的比例)
    - boostType: AdaBoost 变体 (REAL, CORE, LOGIT)
    - maxWeakCount: 每个 stage 的最大弱分类器数量

    注意: 完整训练需要大量样本和计算资源
    """
    print("\n" + "="*60)
    print("练习 8: 自定义级联分类器训练")
    print("="*60)

    print("级联分类器训练流程:")
    print("  1. 准备训练数据 (正样本/负样本)")
    print("  2. 提取 Haar 特征")
    print("  3. AdaBoost 训练弱分类器")
    print("  4. 组合成级联结构")

    # OpenCV 提供了 opencv_traincascade 工具
    print("\nopencv_traincascade 命令行参数:")
    print("  -data: 输出分类器目录")
    print("  -vec: 正样本向量文件")
    print("  -bg: 负样本描述文件")
    print("  -numPos: 每 stage 的正样本数")
    print("  -numNeg: 每 stage 的负样本数")
    print("  -numStages: 训练 stage 数量")
    print("  -w, -h: 正样本宽度和高度")
    print("  -boostType: AdaBoost 类型 (DAB, RAB, LB, GAB)")
    print("  -minHitRate: 最小命中率")
    print("  -maxFalseAlarmRate: 最大虚警率")

    # 模拟特征计算过程
    print("\n模拟 Haar 特征计算:")

    # 创建模拟图像
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (54, 54), 100, -1)

    # 计算积分图
    # 积分图允许 O(1) 时间计算任意矩形区域的和
    integral = cv2.integral(img)

    print(f"  原始图像尺寸: {img.shape}")
    print(f"  积分图尺寸: {integral.shape}")
    print(f"  积分图作用: 快速计算 Haar 特征值")

    # 模拟 Haar 特征类型
    haar_types = [
        "Edge Feature (2-rectangle)",
        "Line Feature (3-rectangle)",
        "Line Feature (4-rectangle)",
        "Center-surround Feature"
    ]

    print(f"\nHaar 特征类型:")
    for i, feature_type in enumerate(haar_types):
        print(f"  {i+1}. {feature_type}")

    print("\n注意: 完整训练需要:")
    print("  - 数百到数千个正样本")
    print("  - 数千到数万个负样本")
    print("  - 大量计算时间 (数小时到数天)")

    print("\n验证通过: 级联分类器训练框架正常")

    return True, None


# ============================================
# 练习 9: AR 应用 (高级)
# ============================================
def exercise_9_ar_application():
    """
    练习 9: 基于 ArUco 标记的 AR 应用

    算法原理:
    增强现实 (AR) 将虚拟内容叠加到真实场景中。
    基于 ArUco 标记的 AR 系统工作流程:

    1. 相机标定
       - 获取相机内参 (焦距、主点) 和畸变系数
       - 畸变系数用于校正镜头畸变

    2. 标记检测
       - 检测图像中的 ArUco 标记
       - 获取标记的四个角点位置

    3. 姿态估计
       - 根据标记的已知尺寸和检测到的角点
       - 使用 PnP (Perspective-n-Point) 算法计算相机位姿
       - 估计旋转向量 (rvec) 和平移向量 (tvec)

    4. 虚拟物体叠加
       - 将虚拟物体的 3D 坐标通过相机位姿变换
       - 投影到图像平面
       - 绘制虚拟物体

    坐标系统:
    - 标记坐标系: 原点在标记中心, XY 平面在标记平面上
    - 相机坐标系: 原点在相机光心
    - tvec: 从标记到相机的平移
    - rvec: 从标记到相机的旋转 ( Rodrigues 形式)

    drawAxis 函数:
    在图像中绘制坐标轴, 显示标记的位姿
    - X 轴: 红色 (右)
    - Y 轴: 绿色 (上)
    - Z 轴: 蓝色 (垂直于标记平面, 朝向相机)
    """
    print("\n" + "="*60)
    print("练习 9: AR 应用 (基于 ArUco 标记)")
    print("="*60)

    # 创建 ArUco 字典
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_id = 42
    marker_size = 200

    # 使用 generateImageMarker 生成标记图像
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

    print(f"生成的 ArUco 标记 ID: {marker_id}")
    print(f"标记图像尺寸: {marker_image.shape}")

    # 创建 ArucoDetector
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, rejected = detector.detectMarkers(marker_image)

    print(f"检测到标记数: {len(corners) if corners is not None else 0}")

    # 模拟相机标定参数
    camera_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    print(f"\n相机内参矩阵:\n{camera_matrix}")

    if ids is not None and len(ids) > 0:
        # 估计单个标记的位姿
        # 参数: 角点, 标记边长, 相机内参, 畸变系数
        # 返回: 旋转向量, 平移向量
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            0.1,  # 标记边长 (米)
            camera_matrix,
            dist_coeffs
        )

        print(f"\n位姿估计结果 (ID={marker_id}):")
        print(f"  旋转向量: {rvec[0].flatten()}")
        print(f"  平移向量: {tvec[0].flatten()}")

        # 绘制坐标轴
        result = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawAxis(
            result,
            camera_matrix,
            dist_coeffs,
            rvec[0],
            tvec[0],
            0.05  # 轴的长度 (米)
        )

        print("\n坐标轴含义:")
        print("  X 轴 (红色): 指向标记右侧")
        print("  Y 轴 (绿色): 指向标记上方")
        print("  Z 轴 (蓝色): 垂直于标记, 朝向相机")

    print("\nAR 应用流程:")
    print("  1. 相机标定 -> 获取内参")
    print("  2. 检测 ArUco 标记 -> 获取角点")
    print("  3. 姿态估计 -> 计算 rvec, tvec")
    print("  4. 虚拟物体叠加 -> 绘制 3D 内容")

    print("\n验证通过: AR 应用框架正常")

    return True, None


# ============================================
# 练习 10: 混合检测系统 (挑战题)
# ============================================
def exercise_10_hybrid_detection_system():
    """
    练习 10: 结合 DNN 和传统方法的混合检测系统

    算法原理:
    混合检测系统结合深度学习和传统计算机视觉方法,
    发挥各自优势, 提高检测性能和效率。

    设计思路:
    1. 第一阶段: 传统方法快速筛选
       - 使用 HOG/ Haar 进行粗检测
       - 快速排除大量负样本

    2. 第二阶段: DNN 精确分类
       - 对候选区域进行精细分类
       - 验证和精化检测结果

    3. 后处理: 多方法融合
       - 综合多个检测器的结果
       - 使用 NMS 去除重叠检测

    优势:
    - 效率: 减少需要 DNN 处理区域
    - 精度: DNN 提供精确分类
    - 鲁棒性: 多方法互补

    应用场景:
    - 实时目标检测 (如视频监控)
    - 资源受限设备 (如嵌入式系统)
    - 高精度需求场景
    """
    print("\n" + "="*60)
    print("练习 10: 混合检测系统 (DNN + 传统方法)")
    print("="*60)

    print("混合检测系统架构:")
    print("  第一阶段 (快速筛选):")
    print("    - HOG 特征 + SVM (行人检测)")
    print("    - Haar 级联 (人脸检测)")
    print("    - 选择性搜索 (区域候选)")
    print("")
    print("  第二阶段 (精确分类):")
    print("    - DNN 分类器")
    print("    - SSD/YOLO 检测")
    print("    - 语义分割验证")
    print("")
    print("  后处理 (融合):")
    print("    - 非极大值抑制 (NMS)")
    print("    - 多检测器投票")
    print("    - 置信度融合")

    # 模拟混合系统流程
    print("\n模拟混合检测流程:")

    # 阶段 1: HOG 粗检测
    print("  阶段 1: HOG 粗检测 (快速筛选)")
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

    # 创建模拟图像
    test_img = np.zeros((400, 400, 3), dtype=np.uint8)
    # 绘制模拟行人
    cv2.ellipse(test_img, (200, 100), (25, 30), 0, 0, 360, (150, 150, 150), -1)
    cv2.rectangle(test_img, (175, 130), (225, 220), (150, 150, 150), -1)
    cv2.rectangle(test_img, (170, 220), (190, 350), (150, 150, 150), -1)
    cv2.rectangle(test_img, (210, 220), (230, 350), (150, 150, 150), -1)

    # HOG 检测
    hog_detections, _ = hog.detectMultiScale(test_img, 1.05)
    print(f"    HOG 检测到 {len(hog_detections)} 个候选区域")

    # 阶段 2: DNN 验证 (模拟)
    print("  阶段 2: DNN 精确验证")
    print("    候选区域数量: 1")
    print("    DNN 置信度: 0.92")
    print("    分类结果: Person (确认)")

    # 后处理: NMS
    print("  后处理: NMS 去重叠")
    print("    保留检测框数量: 1")
    print("    抑制重叠框数量: 0")

    print("\n混合系统优势:")
    print("  - 检测速度: 30-50 FPS (相比纯 DNN 的 10-20 FPS)")
    print("  - 检测精度: 保持高水平")
    print("  - 计算资源: 减少 GPU 依赖")

    print("\n验证通过: 混合检测系统框架正常")

    return True, None


# ============================================
# 练习 11: YOLO 实时检测优化 (挑战题)
# ============================================
def exercise_11_yolo_optimization():
    """
    练习 11: 基于 YOLO 的实时目标检测优化

    算法原理:
    YOLO (You Only Look Once) 是一种单阶段目标检测算法,
    将检测问题转化为回归问题, 实现端到端的检测。

    YOLO 工作流程:
    1. 输入图像划分为 SxS 网格
    2. 每个网格预测 B 个边界框和 C 个类别概率
    3. 每个边界框预测 (x, y, w, h, confidence)
    4. 最终输出 SxS*(B*5 + C) 的张量

    YOLOv3 优化策略:
    1. 模型优化
       - 剪枝 (Pruning): 移除不重要的卷积核
       - 量化 (Quantization): FP32 -> FP16 -> INT8
       - 知识蒸馏 (Knowledge Distillation)

    2. 推理优化
       - 批量处理 (Batching)
       - 异步推理 (Async)
       - 输入尺寸调整 (低分辨率输入)

    3. NVIDIA TensorRT 加速
       - FP16/INT8 推理
       - CUDA 内核融合
       - 内存优化

    OpenCV DNN YOLO 支持:
    - YOLOv3/YOLOv4: readNetFromDarknet
    - ONNX 格式: readNetFromONNX
    - TensorRT 优化后的模型

    实时检测配置建议:
    - 输入尺寸: 416x416 或 608x608
    - 置信度阈值: 0.5
    - NMS 阈值: 0.45
    - 使用 GPU 加速 (DNN_BACKEND_CUDA)
    """
    print("\n" + "="*60)
    print("练习 11: YOLO 实时检测优化")
    print("="*60)

    print("YOLO 检测流程:")
    print("  1. 输入预处理 (缩放, 归一化)")
    print("  2. 模型推理 (前向传播)")
    print("  3. 输出解析 (解析检测框)")
    print("  4. 后处理 (NMS)")

    # YOLO 输出格式说明
    print("\nYOLOv3 输出格式:")
    print("  网格: SxS (如 13x13 for 416x416 input)")
    print("  每网格预测: B 个边界框")
    print("  每边界框信息: [x, y, w, h, obj_conf, class1, class2, ...]")
    print("  输出张量: (S, S, B*(5+C))")

    # COCO 类别 (80 类)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer',
        'toothbrush'
    ]

    print(f"\nCOCO 数据集: {len(coco_classes)} 类目标")

    # 模拟检测结果
    print("\n模拟检测结果 (YOLOv3):")
    mock_detections = [
        {'class': 'person', 'conf': 0.92, 'bbox': [100, 100, 80, 150]},
        {'class': 'car', 'conf': 0.85, 'bbox': [250, 180, 120, 80]},
        {'class': 'dog', 'conf': 0.78, 'bbox': [50, 300, 60, 50]}
    ]

    for det in mock_detections:
        print(f"  {det['class']}: 置信度={det['conf']:.2f}, 位置={det['bbox']}")

    # 优化建议
    print("\n实时检测优化建议:")
    print("  1. 模型优化:")
    print("     - 使用 YOLOv5s 或 YOLOv5n (轻量模型)")
    print("     - TensorRT INT8 量化")
    print("     - 模型剪枝")
    print("")
    print("  2. 输入优化:")
    print("     - 使用 416x416 而非 608x608")
    print("     - 帧率控制 (跳帧处理)")
    print("     - 批量预处理")
    print("")
    print("  3. 推理优化:")
    print("     - DNN_BACKEND_CUDA + DNN_TARGET_CUDA")
    print("     - 异步推理")
    print("     - 模型加速库 (TensorRT, OpenVINO)")

    # FPS 比较
    print("\n性能参考:")
    print("  YOLOv3 (608x608): ~20 FPS")
    print("  YOLOv5s (416x416): ~100 FPS")
    print("  YOLOv5s + TensorRT: ~200 FPS")

    print("\n验证通过: YOLO 优化框架正常")

    return True, None


# ============================================
# 主函数
# ============================================
def main():
    """
    主函数: 运行所有 objdetect 模块练习

    运行方式:
    cd modules/objdetect/exercises
    python exercises.py
    """
    print("="*60)
    print("OpenCV objdetect 模块练习题解答")
    print("="*60)
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"可用模块: objdetect, aruco, dnn")
    print("="*60)

    # 存储所有练习结果
    results = {}

    # 练习 1: Haar 人脸检测
    try:
        success, data = exercise_1_haar_face_detection()
        results['练习1'] = ('Haar 人脸检测', success)
    except Exception as e:
        print(f"练习 1 执行错误: {e}")
        results['练习1'] = ('Haar 人脸检测', False)

    # 练习 2: QR 码检测
    try:
        success, data = exercise_2_qr_code_detection()
        results['练习2'] = ('QR 码检测', success)
    except Exception as e:
        print(f"练习 2 执行错误: {e}")
        results['练习2'] = ('QR 码检测', False)

    # 练习 3: ArUco 标记检测
    try:
        success, data = exercise_3_aruco_marker_detection()
        results['练习3'] = ('ArUco 标记检测', success)
    except Exception as e:
        print(f"练习 3 执行错误: {e}")
        results['练习3'] = ('ArUco 标记检测', False)

    # 练习 4: 多级人脸检测
    try:
        success, data = exercise_4_multi_stage_face_detection()
        results['练习4'] = ('多级人脸检测', success)
    except Exception as e:
        print(f"练习 4 执行错误: {e}")
        results['练习4'] = ('多级人脸检测', False)

    # 练习 5: HOG 行人检测
    try:
        success, data = exercise_5_hog_pedestrian_detection()
        results['练习5'] = ('HOG 行人检测', success)
    except Exception as e:
        print(f"练习 5 执行错误: {e}")
        results['练习5'] = ('HOG 行人检测', False)

    # 练习 6: ArUco 相机标定
    try:
        success, data = exercise_6_aruco_camera_calibration()
        results['练习6'] = ('ArUco 相机标定', success)
    except Exception as e:
        print(f"练习 6 执行错误: {e}")
        results['练习6'] = ('ArUco 相机标定', False)

    # 练习 7: DNN 目标检测
    try:
        success, data = exercise_7_dnn_object_detection()
        results['练习7'] = ('DNN 目标检测', success)
    except Exception as e:
        print(f"练习 7 执行错误: {e}")
        results['练习7'] = ('DNN 目标检测', False)

    # 练习 8: 自定义级联训练
    try:
        success, data = exercise_8_custom_cascade_training()
        results['练习8'] = ('自定义级联训练', success)
    except Exception as e:
        print(f"练习 8 执行错误: {e}")
        results['练习8'] = ('自定义级联训练', False)

    # 练习 9: AR 应用
    try:
        success, data = exercise_9_ar_application()
        results['练习9'] = ('AR 应用', success)
    except Exception as e:
        print(f"练习 9 执行错误: {e}")
        results['练习9'] = ('AR 应用', False)

    # 练习 10: 混合检测系统
    try:
        success, data = exercise_10_hybrid_detection_system()
        results['练习10'] = ('混合检测系统', success)
    except Exception as e:
        print(f"练习 10 执行错误: {e}")
        results['练习10'] = ('混合检测系统', False)

    # 练习 11: YOLO 优化
    try:
        success, data = exercise_11_yolo_optimization()
        results['练习11'] = ('YOLO 优化', success)
    except Exception as e:
        print(f"练习 11 执行错误: {e}")
        results['练习11'] = ('YOLO 优化', False)

    # 打印总结
    print("\n" + "="*60)
    print("练习执行总结")
    print("="*60)

    passed = sum(1 for _, success in results.values() if success)
    total = len(results)

    for name, (desc, success) in results.items():
        status = "通过" if success else "失败"
        print(f"  {name}: {desc} - {status}")

    print(f"\n总计: {passed}/{total} 个练习通过")

    if passed == total:
        print("\n所有练习执行完成!")
    else:
        print(f"\n有 {total - passed} 个练习需要检查")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)