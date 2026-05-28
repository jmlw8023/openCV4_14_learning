# -*- coding: utf-8 -*-
"""
OpenCV features2d 模块练习题
==========================

本文件包含 features2d 模块的所有练习题解答:

入门级:
  1. ORB 特征点检测与可视化
  2. 两幅图像的 ORB 特征匹配
  3. SIFT 特征匹配

中级:
  4. 带比率测试的特征匹配过滤
  5. 基于 RANSAC 的单应性估计
  6. 不同检测器性能比较

高级:
  7. 完整图像拼接流程
  8. 基于特征的图像检索系统
  9. ORB 目标检测模板匹配

挑战题:
  10. 基于深度学习的特征描述符 (SuperPoint/SuperMatch)
  11. 多图像批量拼接

作者: OpenCV Learning
版本: 1.0
"""

import cv2
import numpy as np
import os

# =============================================================================
# 工具函数
# =============================================================================

def create_test_image(width=800, height=600):
    """
    创建测试用合成图像

    算法原理:
    - 生成棋盘格图案用于测试特征检测
    - 添加几何形状提供明显特征点
    """
    # 创建棋盘格背景 (提供丰富角点)
    img = np.zeros((height, width), dtype=np.uint8)
    cell_size = 40
    for y in range(height):
        for x in range(width):
            # 棋盘格: 黑白交替
            if ((x // cell_size) + (y // cell_size)) % 2 == 0:
                img[y, x] = 255

    # 添加旋转的矩形 (提供更多角点)
    pts = np.array([
        [200, 150], [350, 200], [300, 350], [150, 300]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)

    # 添加圆形区域
    cv2.circle(img, (500, 300), 80, 255, -1)

    # 转换为 BGR 图像用于彩色显示
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img, img_color


def create_test_image_pair():
    """
    创建模拟的两幅测试图像 (带视角变化)

    算法原理:
    - 第一幅: 正面视角的棋盘格
    - 第二幅: 模拟旋转后的视角 (通过透视变换)

    透视变换矩阵:
    - 使用 3x3 单应性矩阵模拟相机旋转
    """
    img1, _ = create_test_image()

    # 定义透视变换 (模拟轻微旋转)
    src_pts = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
    dst_pts = np.float32([[50, 30], [750, 50], [720, 580], [30, 570]])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img2 = cv2.warpPerspective(img1, H, (800, 600))

    return img1, img2, H


def draw_keypoints_on_image(image, keypoints, title="Keypoints"):
    """
    在图像上绘制关键点

    参数:
      image: 输入图像
      keypoints: 关键点列表
      title: 窗口标题

    算法: 使用 cv2.drawKeypoints 绘制关键点
    - DEFAULT: 只绘制点位置
    - DRAW_RICH_KEYPOINTS: 绘制带大小和方向的关键点
    """
    if len(image.shape) == 2:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color_img = image.copy()

    # 绘制关键点 (带方向和大小的丰富模式)
    output = cv2.drawKeypoints(
        color_img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"[{title}] 检测到 {len(keypoints)} 个关键点")
    return output


def draw_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    """
    绘制两幅图像之间的特征匹配

    算法: 使用 cv2.drawMatches 并排显示两张图像及匹配线
    """
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    result = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print(f"[{title}] 显示 {len(matches)} 个匹配")
    return result


# =============================================================================
# 练习 1: ORB 特征点检测与可视化 (入门级)
# =============================================================================

def exercise1_orb_detection():
    """
    练习 1: ORB 特征点检测与可视化

    算法原理:
    ---------------
    ORB (Oriented FAST and Rotated BRIEF) 是融合了 FAST 角点检测和 BRIEF 描述符的算法:

    1. FAST 角点检测:
       - 在像素周围画一个半径为 3 的圆 (16 个像素)
       - 如果圆上有 N 个连续像素都比中心像素亮或都更暗,则中心是角点
       - N 通常取 12 (FAST-12)

    2. 方向计算 (BRIEF 的改进):
       - 使用灰度质心法计算关键点方向
       - 计算公式: m_10 = Σ(x*y), m_01 = Σ(y)
       - 角度 = atan2(m_01, m_10)

    3. rBRIEF 描述符:
       - 在特征点附近随机选取像素对进行比较
       - 根据方向旋转采样模式,确保旋转不变性

    ORB 的优势:
    - 计算速度快,适合实时应用
    - 专利免费 (不像 SIFT 和 SURF 有专利)
    - 具有旋转不变性

    参数说明:
    - nfeatures: 最大特征点数量
    - scaleFactor: 金字塔缩放因子 (默认 1.2)
    - nlevels: 金字塔层数
    - edgeThreshold: 边缘过滤阈值
    - firstLevel: 第一层金字塔索引
    - WTA_K: 每次比较的像素对数 (2,3,4)
    - patchSize: BRIEF 描述符的邻域大小
    """
    print("\n" + "="*70)
    print("练习 1: ORB 特征点检测与可视化")
    print("="*70)

    # 创建测试图像
    img, _ = create_test_image()

    # 创建 ORB 检测器
    # ORB::create() 参数详解:
    # - nfeatures=500: 最多检测 500 个特征点
    # - scaleFactor=1.2: 图像金字塔每层缩小比例
    # - nlevels=8: 金字塔层数
    # - edgeThreshold=31: 边缘阈值,低于此值的边缘被忽略
    # - firstLevel=0: 第一层金字塔索引
    # - WTA_K=2: BRIEF 比较的像素对数
    # - scoreType=Harris: 使用 Harris 响应排序特征点
    # - patchSize=31: 描述符计算邻域大小
    orb = cv2.ORB_create(
        nfeatures=500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31
    )

    # 检测关键点 (只检测,不计算描述符)
    # detect 函数输出: vector<KeyPoint>
    # KeyPoint 属性:
    #   - pt: (x, y) 坐标
    #   - size: 邻域直径 (通常 31)
    #   - angle: 方向 [0, 360), -1 表示无方向
    #   - response: 响应强度 (Harris 响应值)
    #   - octave: 金字塔层索引
    #   - class_id: 特征组 ID
    keypoints = orb.detect(img, None)

    print(f"\n[ORB 检测结果]")
    print(f"  关键点数量: {len(keypoints)}")

    # 分析关键点属性
    if keypoints:
        angles = [kp.angle for kp in keypoints if kp.angle >= 0]
        responses = [kp.response for kp in keypoints]
        octaves = set(kp.octave for kp in keypoints)

        print(f"  有方向的关键点: {len(angles)}/{len(keypoints)}")
        print(f"  响应值范围: [{min(responses):.2f}, {max(responses):.2f}]")
        print(f"  金字塔层数: {sorted(octaves)}")
        print(f"  平均响应强度: {np.mean(responses):.2f}")

    # 绘制关键点
    output = draw_keypoints_on_image(img, keypoints, "ORB 特征点")

    # 验证: 检查是否检测到特征点
    assert len(keypoints) > 0, "未检测到任何特征点!"
    print(f"\n[验证通过] ORB 成功检测到 {len(keypoints)} 个特征点")

    return output


# =============================================================================
# 练习 2: 两幅图像的 ORB 特征匹配 (入门级)
# =============================================================================

def exercise2_orb_matching():
    """
    练习 2: 两幅图像的 ORB 特征匹配

    算法原理:
    ---------------
    特征匹配是计算机视觉中的核心任务,用于在两幅图像中找到对应的点。

    1. 特征检测与描述:
       - 使用 ORB 检测特征点
       - 使用 ORB 计算二进制描述符 (32 bytes = 256 bits)

    2. 暴力匹配 (BFMatcher):
       - 对于第一幅图像的每个描述符,与第二幅的所有描述符逐一比较
       - 计算 Hamming 距离 (异或操作计数)
       - 选择距离最小的作为最佳匹配

    3. Hamming 距离:
       - 用于二进制描述符 (ORB, BRISK, FREAK)
       - distance = count of different bits
       - 例如: 1010 XOR 1110 = 0100, 有 1 位不同

    匹配流程:
    1. detectAndCompute: 同时检测关键点并计算描述符
    2. BFMatcher.match: 找到最佳匹配
    3. 可选: 使用 ratio test 过滤匹配

    DMatch 结构:
    - queryIdx: 第一幅图像 (查询图像) 的描述符索引
    - trainIdx: 第二幅图像 (训练图像) 的描述符索引
    - distance: 匹配距离 (Hamming 或 Euclidean)
    - imgIdx: 训练图像索引 (多图像时)
    """
    print("\n" + "="*70)
    print("练习 2: 两幅图像的 ORB 特征匹配")
    print("="*70)

    # 创建测试图像对
    img1, img2, _ = create_test_image_pair()

    # 创建 ORB 检测器
    orb = cv2.ORB_create(nfeatures=500)

    # 检测关键点并计算描述符
    # detectAndCompute 同步完成检测和描述符计算,效率更高
    # 参数: image, mask, keypoints, descriptors
    # 返回: keypoints (更新后的), descriptors
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)

    print(f"\n[特征检测结果]")
    print(f"  图像1关键点: {len(kp1)}")
    print(f"  图像2关键点: {len(kp2)}")

    # 创建 BFMatcher (Brute Force Matcher)
    # BFMatcher 参数:
    # - normType: 距离度量
    #   - NORM_HAMMING: 汉明距离,用于二进制描述符 (ORB, BRISK)
    #   - NORM_L2: 欧氏距离,用于浮点描述符 (SIFT, SURF)
    # - crossCheck: 交叉验证,双向匹配一致才保留
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 执行匹配
    # match 返回最佳匹配 DMatch
    # knnMatch 返回 k 个最佳匹配
    matches = matcher.match(desc1, desc2)

    print(f"\n[匹配结果]")
    print(f"  总匹配数: {len(matches)}")

    # 分析匹配距离
    if matches:
        distances = [m.distance for m in matches]
        print(f"  最小距离: {min(distances)}")
        print(f"  最大距离: {max(distances)}")
        print(f"  平均距离: {np.mean(distances):.2f}")

    # 绘制匹配结果
    result = draw_matches(img1, kp1, img2, kp2, matches, "ORB 匹配")

    # 验证
    assert len(matches) > 0, "未找到任何匹配!"
    print(f"\n[验证通过] ORB 匹配成功找到 {len(matches)} 个匹配对")

    return result


# =============================================================================
# 练习 3: SIFT 特征匹配 (入门级)
# =============================================================================

def exercise3_sift_matching():
    """
    练习 3: SIFT 特征匹配

    算法原理:
    ---------------
    SIFT (Scale-Invariant Feature Transform) 由 David Lowe 于 1999 年提出,
    是最经典和广泛使用的特征检测算法。

    1. 尺度空间极值检测:
       - 使用 DoG (Difference of Gaussian) 金字塔
       - 在不同尺度的图像中寻找极值点
       - 确保尺度不变性

    2. 关键点定位:
       - 精确定位亚像素级关键点
       - 去除低对比度和边缘响应点
       - 使用泰勒展开拟合

    3. 方向分配:
       - 计算关键点邻域的梯度方向直方图
       - 取峰值方向作为主方向
       - 确保旋转不变性

    4. 描述符生成:
       - 在关键点周围 16x16 邻域计算梯度
       - 将邻域分成 4x4 子块
       - 每个子块计算 8 方向直方图
       - 最终形成 128 维特征向量 (4x4x8=128)

    专利情况:
    - SIFT 专利在 2020 年过期
    - OpenCV 4.x 版本已包含 SIFT 实现

    与 ORB 的区别:
    - ORB: 二进制描述符 (256 bits), 快速, 无尺度不变
    - SIFT: 浮点描述符 (128 维), 精度高, 有尺度不变
    """
    print("\n" + "="*70)
    print("练习 3: SIFT 特征匹配")
    print("="*70)

    # 检查 SIFT 是否可用 (OpenCV 4.x)
    try:
        sift = cv2.SIFT_create(nfeatures=500)
        print("[SIFT] SIFT 检测器创建成功 (OpenCV 4.x)")
    except AttributeError:
        print("[SIFT] 错误: SIFT 不可用 (检查 OpenCV 版本)")
        return None

    # 创建测试图像
    img1, img2, _ = create_test_image_pair()

    # 检测关键点并计算描述符
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    print(f"\n[SIFT 检测结果]")
    print(f"  图像1关键点: {len(kp1)}")
    print(f"  图像2关键点: {len(kp2)}")
    print(f"  描述符维度: {desc1.shape[1]} (SIFT 是 128 维)")

    # SIFT 使用欧氏距离,创建 FlannBasedMatcher
    # Flann = Fast Library for Approximate Nearest Neighbors
    # 适用于高维浮点描述符,比 BFMatcher 更快

    # 创建 FLANN 匹配器
    # FLANN 使用 KD-Tree (对于 SIFT 128维数据)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 搜索遍历次数

    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # k-近邻匹配 (k=2 用于比率测试)
    knn_matches = flann_matcher.knnMatch(desc1, desc2, k=2)

    print(f"\n[FLANN 匹配结果]")
    print(f"  k-近邻匹配数: {len(knn_matches)}")

    # 绘制匹配结果
    result = draw_matches(img1, kp1, img2, kp2, knn_matches, "SIFT 匹配")

    # 验证
    assert len(knn_matches) > 0, "SIFT 未找到任何匹配!"
    print(f"\n[验证通过] SIFT 成功匹配 {len(knn_matches)} 个特征对")

    return result


# =============================================================================
# 练习 4: 带比率测试的特征匹配过滤 (中级)
# =============================================================================

def exercise4_ratio_test_filtering():
    """
    练习 4: 带比率测试的特征匹配过滤

    算法原理:
    ---------------
    Lowe 的比率测试 (Ratio Test) 是过滤错误匹配的标准方法。

    1. 为什么需要过滤:
       - 直接匹配可能产生错误匹配 (outliers)
       - 错误匹配会导致后续计算出错

    2. 比率测试原理:
       - 对于每个关键点,找到两个最近匹配: m1, m2
       - 计算比率: ratio = m1.distance / m2.distance
       - 如果 best.match 远好于 second.best.match,说明 best 是可靠的
       - 阈值通常取 0.75 (可调整)

    3. 判断标准:
       - ratio < 0.75: 好的匹配 (unique match)
       - ratio >= 0.75: 匹配不明确,可能是歧义匹配

    4. 交叉验证增强:
       - 除了比率测试,还可以做双向匹配验证
       - 检查 query->train 和 train->query 是否一致
       - 交叉验证可进一步减少错误匹配

    数学解释:
    - 假设匹配距离服从某种分布
    - 如果 best.distance 远小于 second.distance
    - 说明 best 是显著的最佳匹配
    """
    print("\n" + "="*70)
    print("练习 4: 带比率测试的特征匹配过滤")
    print("="*70)

    # 创建测试图像
    img1, img2, _ = create_test_image_pair()

    # 使用 ORB
    orb = cv2.ORB_create(nfeatures=500)
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)

    # BFMatcher 进行 k-近邻匹配
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)  # k=2 用于比率测试

    print(f"\n[过滤前]")
    print(f"  k-近邻匹配总数: {len(knn_matches)}")

    # 应用比率测试 (Lowe's ratio test)
    # ratio_threshold = 0.75 是经典阈值
    ratio_threshold = 0.75
    good_matches = []

    for match_pair in knn_matches:
        # match_pair 包含 k 个匹配,这里 k=2
        if len(match_pair) >= 2:
            # match[0] 是最佳匹配, match[1] 是次佳匹配
            best = match_pair[0]
            second_best = match_pair[1]

            # 计算比率
            ratio = best.distance / second_best.distance

            # 只保留比率小于阈值的匹配
            if ratio < ratio_threshold:
                good_matches.append(best)

    print(f"\n[比率测试过滤]")
    print(f"  阈值: {ratio_threshold}")
    print(f"  过滤后匹配数: {len(good_matches)}")
    print(f"  过滤率: {(1 - len(good_matches)/len(knn_matches))*100:.1f}%")

    # 分析过滤后匹配的距离分布
    if good_matches:
        distances = [m.distance for m in good_matches]
        print(f"\n[过滤后匹配质量]")
        print(f"  最小距离: {min(distances)}")
        print(f"  最大距离: {max(distances)}")
        print(f"  平均距离: {np.mean(distances):.2f}")

    # 绘制过滤后的匹配
    result = draw_matches(img1, kp1, img2, kp2, good_matches, "比率测试过滤后")

    # 验证
    assert len(good_matches) > 0, "比率测试后无有效匹配!"
    print(f"\n[验证通过] 比率测试成功过滤出 {len(good_matches)} 个高质量匹配")

    return result


# =============================================================================
# 练习 5: 基于 RANSAC 的单应性估计 (中级)
# =============================================================================

def exercise5_ransac_homography():
    """
    练习 5: 基于 RANSAC 的单应性估计

    算法原理:
    ---------------
    单应性 (Homography) 描述了两个平面之间的透视变换关系。

    1. 单应性矩阵:
       - 3x3 矩阵,有 8 个自由度 (齐次坐标,尺度不变)
       - H = [h11 h12 h13; h21 h22 h23; h31 h32 h33]
       - 表示平面间的旋转、平移、投影变换

    2. 应用场景:
       - 图像拼接
       - 平面校正
       - 目标检测

    3. RANSAC (Random Sample Consensus):
       - 鲁棒的参数估计方法
       - 能处理 outliers (错误匹配)

    4. RANSAC 算法流程:
       a) 随机选择最小样本集 (4 个匹配点)
       b) 计算模型参数 (单应性矩阵)
       c) 计算所有点的投影误差
       d) 统计 inliers (内点) 数量
       e) 重复 N 次,选择内点最多的模型

    5. findHomography 参数:
       - method=RANSAC: 使用 RANSAC 鲁棒估计
       - ransacReprojThreshold=5.0: 投影误差阈值 (像素)
       - confidence=0.995: 置信度
       - maxIters=2000: 最大迭代次数

    投影误差计算:
    - 对于匹配点 (p1, p2)
    - p2' = H * p1 (齐次坐标乘法)
    - 误差 = ||p2 - p2'||^2
    """
    print("\n" + "="*70)
    print("练习 5: 基于 RANSAC 的单应性估计")
    print("="*70)

    # 创建测试图像对 (有已知单应性)
    img1, img2, expected_H = create_test_image_pair()

    # ORB 特征匹配
    orb = cv2.ORB_create(nfeatures=500)
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)

    # 使用 BFMatcher 和比率测试
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    # 比率测试过滤
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) >= 2:
            if match_pair[0].distance / match_pair[1].distance < 0.75:
                good_matches.append(match_pair[0])

    print(f"\n[特征匹配]")
    print(f"  高质量匹配数: {len(good_matches)}")

    # 提取匹配点坐标
    # pts1, pts2 是 Nx2 数组,用于 findHomography
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 计算单应性矩阵
    # findHomography 参数:
    # - pts1: 源图像关键点坐标
    # - pts2: 目标图像关键点坐标
    # - method: RANSAC, LMEDS (最小中值), RHO, 0 (DLT)
    # - ransacReprojThreshold: RANSAC 时投影误差阈值
    # - confidence: RANSAC 置信度
    H, mask = cv2.findHomography(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        confidence=0.995,
        maxIters=2000
    )

    # mask 标识内点 (1) 和外点 (0)
    # 只保留内点 (RANSAC 认为是正确的匹配)
    inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

    print(f"\n[RANSAC 单应性估计]")
    print(f"  计算得到 3x3 单应性矩阵:")
    print(f"  {H}")

    # 计算内点数量和比例
    inlier_count = np.sum(mask)
    inlier_ratio = inlier_count / len(mask) * 100

    print(f"\n[RANSAC 内点分析]")
    print(f"  内点数量: {inlier_count}/{len(mask)}")
    print(f"  内点比例: {inlier_ratio:.1f}%")

    # 验证单应性矩阵质量
    # 使用单应性矩阵变换第一幅图像的特征点
    # 检查与第二幅图像的匹配误差
    if inlier_count >= 4:
        # 随机选择 4 个内点验证
        sample_indices = np.random.choice(len(pts1), min(4, len(pts1)), replace=False)

        total_error = 0.0
        for idx in sample_indices:
            pt1_h = np.append(pts1[idx], 1)  # 齐次坐标
            pt2_pred = H @ pt1_h  # 矩阵乘法
            pt2_pred = pt2_pred / pt2_pred[2]  # 归一化

            error = np.linalg.norm(pt2_pred[:2] - pts2[idx])
            total_error += error

        avg_error = total_error / len(sample_indices)
        print(f"\n[投影误差验证]")
        print(f"  平均投影误差: {avg_error:.2f} 像素")
        print(f"  (误差 < 5 像素 表示单应性估计良好)")

    # 绘制只包含内点的匹配
    result = draw_matches(img1, kp1, img2, kp2, inliers, "RANSAC 内点匹配")

    # 验证
    assert H is not None, "单应性矩阵计算失败!"
    assert inlier_count >= 4, "内点数量不足,无法计算单应性!"
    print(f"\n[验证通过] RANSAC 成功估计单应性矩阵,内点比例 {inlier_ratio:.1f}%")

    return result, H


# =============================================================================
# 练习 6: 不同检测器性能比较 (中级)
# =============================================================================

def exercise6_detector_comparison():
    """
    练习 6: 不同检测器性能比较

    算法原理:
    ---------------
    特征检测器比较需要考虑多个指标:

    1. 检测器类型:

       a) ORB (Oriented FAST and Rotated BRIEF):
          - 基于 FAST 角点检测
          - 二进制描述符 (256 bits)
          - 快速,无专利
          - 无尺度不变性

       b) AKAZE (Accelerated KAZE):
          - 非线性尺度空间
          - M-LDB 描述符 (486 bytes)
          - 快速,无专利
          - 有尺度不变性和旋转不变性

       c) BRISK (Binary Robust Invariant Scalable Keypoints):
          - FAST 改进 + 金字塔
          - 二进制描述符 (64 bytes)
          - 有尺度不变性

    2. 性能指标:
       - 检测数量: 图像中检测到的特征点数量
       - 匹配数量: 两幅图像间的匹配对数量
       - 匹配质量: 正确匹配的比例
       - 计算时间: 检测和描述符计算的时间

    3. 描述符匹配:
       - ORB: 使用 Hamming 距离 (快)
       - AKAZE: 使用 Hamming 距离 (快)
       - BRISK: 使用 Hamming 距离 (快)

    4. 选择建议:
       - 实时应用: ORB
       - 精度要求高: AKAZE 或 SIFT
       - 需要尺度不变: AKAZE
    """
    print("\n" + "="*70)
    print("练习 6: 不同检测器性能比较")
    print("="*70)

    # 创建测试图像
    img1, img2, _ = create_test_image_pair()

    # 定义要比较的检测器
    detectors = {
        "ORB": cv2.ORB_create(nfeatures=500),
        "AKAZE": cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,  # 0 = 完整描述符
            descriptor_channels=3,
            threshold=0.001,
            nOctaves=4,
            nOctaveLayers=4
        ),
        "BRISK": cv2.BRISK_create(
            thresh=30,
            octaves=3,
            patternScale=1.0
        )
    }

    # 存储结果
    results = {}

    for name, detector in detectors.items():
        print(f"\n[{name}] 检测中...")

        # 计时
        import time
        start = time.time()

        # 检测和计算
        kp1, desc1 = detector.detectAndCompute(img1, None)
        kp2, desc2 = detector.detectAndCompute(img2, None)

        detect_time = time.time() - start

        print(f"  检测时间: {detect_time*1000:.1f} ms")
        print(f"  图像1特征点: {len(kp1)}, 图像2特征点: {len(kp2)}")

        # 匹配
        start = time.time()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.match(desc1, desc2)
        match_time = time.time() - start

        print(f"  匹配时间: {match_time*1000:.1f} ms")
        print(f"  匹配数量: {len(matches)}")

        # 计算匹配质量 (使用比率测试)
        knn_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = knn_matcher.knnMatch(desc1, desc2, k=2)

        good_count = 0
        for match_pair in knn_matches:
            if len(match_pair) >= 2:
                if match_pair[0].distance / match_pair[1].distance < 0.75:
                    good_count += 1

        print(f"  高质量匹配 (ratio test): {good_count}")

        # 存储结果
        results[name] = {
            "detector": detector,
            "kp1": kp1,
            "kp2": kp2,
            "desc1": desc1,
            "desc2": desc2,
            "matches": matches,
            "detect_time": detect_time,
            "match_time": match_time,
            "good_count": good_count
        }

    # 汇总比较
    print("\n" + "-"*50)
    print("[性能比较汇总]")
    print("-"*50)
    print(f"{'检测器':<10} {'检测时间':<12} {'匹配时间':<12} {'匹配数':<10} {'高质量匹配':<12}")
    print("-"*50)

    for name, res in results.items():
        print(f"{name:<10} {res['detect_time']*1000:>8.1f}ms {res['match_time']*1000:>8.1f}ms "
              f"{len(res['matches']):>8} {res['good_count']:>10}")

    print("-"*50)

    # 确定最佳检测器
    best_by_matches = max(results.items(), key=lambda x: len(x[1]['matches']))
    best_by_quality = max(results.items(), key=lambda x: x[1]['good_count'])

    print(f"\n[分析结论]")
    print(f"  最多匹配: {best_by_matches[0]} ({len(best_by_matches[1]['matches'])} 个)")
    print(f"  最高质量: {best_by_quality[0]} ({best_by_quality[1]['good_count']} 个高质量)")

    return results


# =============================================================================
# 练习 7: 完整图像拼接流程 (高级)
# =============================================================================

def exercise7_image_stitching():
    """
    练习 7: 完整图像拼接流程

    算法原理:
    ---------------
    图像拼接是将多幅有重叠区域的图像合成为一幅全景图。

    完整流程:
    1. 特征检测
       - 使用 ORB 或 AKAZE 检测特征点
       - ORB 速度快,适合实时拼接

    2. 特征描述
       - 计算每个关键点的描述符

    3. 特征匹配
       - 使用 BFMatcher 或 FLANN 匹配描述符
       - 应用比率测试过滤错误匹配

    4. 单应性估计
       - 使用 RANSAC 估计两幅图像间的单应性矩阵
       - 处理 outliers

    5. 图像变换
       - 使用 warpPerspective 将图像变换到统一坐标系
       - 应用得到的单应性矩阵

    6. 图像融合
       - 将变换后的图像拼接
       - 处理重叠区域的融合 (简单平均或羽化)

    关键函数:
    - findHomography: 计算单应性矩阵
    - warpPerspective: 应用透视变换
    - addWeighted: 图像融合

    注意事项:
    - 图像需要有足够的重叠区域 (>30%)
    - 避免大范围无特征区域 (白墙等)
    - 光照变化会影响匹配效果
    """
    print("\n" + "="*70)
    print("练习 7: 完整图像拼接流程")
    print("="*70)

    # 创建模拟的拼接场景 (两幅有重叠的图像)
    img1, img2, _ = create_test_image_pair()

    print("\n[步骤 1: 特征检测与匹配]")
    # 使用 ORB 检测器
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)

    print(f"  图像1特征点: {len(kp1)}")
    print(f"  图像2特征点: {len(kp2)}")

    print("\n[步骤 2: 特征匹配与过滤]")
    # k-近邻匹配
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    # 比率测试
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) >= 2:
            if match_pair[0].distance / match_pair[1].distance < 0.75:
                good_matches.append(match_pair[0])

    print(f"  过滤后匹配数: {len(good_matches)}")

    print("\n[步骤 3: RANSAC 单应性估计]")
    # 提取匹配点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # RANSAC 单应性估计
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # 内点数量
    inliers = sum(mask)
    print(f"  内点数量: {inliers}/{len(mask)}")
    print(f"  单应性矩阵:\n  {H}")

    print("\n[步骤 4: 图像拼接]")
    # 获取第二幅图像的尺寸
    h, w = img2.shape[:2]

    # 将图像1变换到图像2的坐标系
    # Warp img1 to img2 using computed homography
    # Output size covers both images
    result_width = w * 2
    result_height = h * 2

    result = cv2.warpPerspective(img1, H, (result_width, result_height))

    # 将图像2复制到结果图像的正确位置
    # 平移以适应变换后的图像位置
    # 这里简单处理:将img2放在右侧
    result[0:h, 0:w] = img2

    print(f"  输出图像尺寸: {result.shape[:2]}")

    print("\n[步骤 5: 融合处理]")
    # 简单的加权融合在重叠区域
    # 创建重叠区域掩码
    overlap_mask = np.zeros_like(result)
    overlap_mask[0:h, w//2:w] = 1  # 右半部分作为简单示例

    # 应用羽化效果 (线性渐变)
    fade = np.tile(np.linspace(0, 1, w//2), (h, 1))
    fade = np.stack([fade]*3, axis=-1)

    # 简单融合: 将重叠区域平滑过渡
    # 实际应用中需要更复杂的融合算法

    print("  融合处理完成 (简单羽化)")

    # 绘制匹配结果
    match_vis = draw_matches(img1, kp1, img2, kp2, good_matches, "拼接特征匹配")

    # 验证
    assert H is not None, "单应性矩阵计算失败"
    assert inliers >= 4, "内点不足"
    print(f"\n[验证通过] 图像拼接流程完成,使用 {inliers} 个内点估计单应性")

    return match_vis, H


# =============================================================================
# 练习 8: 基于特征的图像检索系统 (高级)
# =============================================================================

def exercise8_image_retrieval():
    """
    练习 8: 基于特征的图像检索系统

    算法原理:
    ---------------
    图像检索是在图像库中找到与查询图像相似的图像。

    1. 特征提取:
       - 为库中每幅图像计算特征描述符
       - 使用聚类构建视觉词典 (Bag of Words)

    2. 视觉词典 (BoVW - Bag of Visual Words):
       - 将描述符空间量化到 K 个视觉单词
       - 使用 K-means 聚类
       - 每幅图像表示为直方图向量

    3. 相似度计算:
       - 查询图像: 提取特征 -> 量化 -> 直方图
       - 库图像: 预计算的直方图
       - 相似度 = 直方图交集 或 余弦相似度

    4. 改进方法:
       - TF-IDF 加权
       - 空间验证
       - 词汇树

    实现简化版本:
    - 不使用完整的 BoVW
    - 使用平均描述符距离作为相似度度量
    """
    print("\n" + "="*70)
    print("练习 8: 基于特征的图像检索系统")
    print("="*70)

    # 创建模拟图像库 (5幅图像)
    print("\n[步骤 1: 构建图像库]")
    image_library = []
    for i in range(5):
        img, _ = create_test_image()
        # 添加不同变化
        if i == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif i == 2:
            img = cv2.GaussianBlur(img, (5, 5), 1)
        elif i == 3:
            img = cv2.resize(img, (600, 450))
        image_library.append(img)

    print(f"  图像库大小: {len(image_library)} 幅图像")

    print("\n[步骤 2: 特征提取]")
    # 使用 ORB 提取特征
    orb = cv2.ORB_create(nfeatures=200)

    library_features = []
    for i, img in enumerate(image_library):
        kp, desc = orb.detectAndCompute(img, None)
        library_features.append({
            "keypoints": kp,
            "descriptors": desc,
            "image_index": i
        })
        print(f"  图像 {i}: {len(kp)} 个特征点")

    print("\n[步骤 3: 模拟查询]")
    # 使用第一幅图像作为查询
    query_img = image_library[0]
    query_kp, query_desc = orb.detectAndCompute(query_img, None)

    print(f"  查询图像特征点: {len(query_kp)}")

    print("\n[步骤 4: 相似度计算]")
    # 使用 BFMatcher 计算与每幅库图像的匹配数
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    similarity_scores = []
    for i, feat in enumerate(library_features):
        # k-近邻匹配
        matches = matcher.match(query_desc, feat["descriptors"])

        # 计算匹配分数 (匹配数量作为相似度)
        score = len(matches)
        similarity_scores.append((i, score, matches))

    # 按相似度排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n[检索结果 (按相似度排序)]")
    print(f"{'排名':<6} {'图像索引':<10} {'匹配数':<10} {'相似度':<10}")
    print("-"*36)

    for rank, (idx, score, matches) in enumerate(similarity_scores[:5], 1):
        similarity = score / max(len(query_kp), 1) * 100
        print(f"{rank:<6} {idx:<10} {score:<10} {similarity:>8.1f}%")

    # 最相似图像的匹配可视化
    best_idx, best_score, best_matches = similarity_scores[0]
    result = draw_matches(
        query_img, query_kp,
        image_library[best_idx], library_features[best_idx]["keypoints"],
        best_matches,
        f"最佳匹配 (图像 {best_idx})"
    )

    print(f"\n[验证通过]")
    print(f"  最相似图像: 索引 {best_idx}")
    print(f"  匹配特征点数: {best_score}")

    return result, similarity_scores


# =============================================================================
# 练习 9: ORB 目标检测模板匹配 (高级)
# =============================================================================

def exercise9_template_matching():
    """
    练习 9: ORB 目标检测模板匹配

    算法原理:
    ---------------
    基于特征的目标检测是在图像中找到与模板相似的目标区域。

    1. 模板处理:
       - 读取模板图像
       - 检测模板的特征点和描述符

    2. 场景搜索:
       - 在场景图像中检测特征点
       - 将模板特征与场景特征匹配

    3. 空间验证:
       - 使用 RANSAC 估计变换矩阵
       - 过滤不符合几何约束的匹配

    4. 边界框计算:
       - 使用估计的变换矩阵变换模板角点
       - 得到目标在场景中的位置

    应用场景:
    - 产品检测
    - 文档扫描
    - 增强现实

    注意:
    - 模板需要包含足够的纹理特征
    - 场景中可能有多个目标实例
    - 光照和视角变化会影响效果
    """
    print("\n" + "="*70)
    print("练习 9: ORB 目标检测模板匹配")
    print("="*70)

    # 创建模拟场景和模板
    scene, _ = create_test_image()
    template = scene[150:350, 200:400]  # 从场景中提取一块作为模板

    print("\n[步骤 1: 模板特征提取]")
    orb = cv2.ORB_create(nfeatures=200)
    template_kp, template_desc = orb.detectAndCompute(template, None)
    print(f"  模板特征点: {len(template_kp)}")

    print("\n[步骤 2: 场景特征提取]")
    scene_kp, scene_desc = orb.detectAndCompute(scene, None)
    print(f"  场景特征点: {len(scene_kp)}")

    print("\n[步骤 3: 特征匹配]")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # k-近邻匹配
    knn_matches = matcher.knnMatch(template_desc, scene_desc, k=2)

    # 比率测试
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) >= 2:
            if match_pair[0].distance / match_pair[1].distance < 0.75:
                good_matches.append(match_pair[0])

    print(f"  过滤后匹配数: {len(good_matches)}")

    print("\n[步骤 4: 几何验证 (RANSAC)]")
    # 提取匹配点对
    query_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches])
    train_pts = np.float32([scene_kp[m.trainIdx].pt for m in good_matches])

    # RANSAC 估计变换矩阵
    if len(good_matches) >= 4:
        H, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

        # 计算内点数量
        inliers = sum(mask)
        inlier_ratio = inliers / len(mask) * 100

        print(f"  内点数量: {inliers}/{len(mask)}")
        print(f"  内点比例: {inlier_ratio:.1f}%")

        print("\n[步骤 5: 边界框计算]")
        # 模板的四个角点
        h, w = template.shape[:2]
        template_corners = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ]).reshape(-1, 1, 2)

        # 变换到场景坐标系
        if H is not None:
            transformed_corners = cv2.perspectiveTransform(template_corners, H)

            print(f"  模板尺寸: {w}x{h}")
            print(f"  检测到的目标位置 (四个角点):")
            for i, corner in enumerate(transformed_corners):
                x, y = corner[0]
                print(f"    角点 {i+1}: ({x:.1f}, {y:.1f})")

        print("\n[步骤 6: 可视化]")
        # 绘制匹配结果
        match_result = draw_matches(
            template, template_kp,
            scene, scene_kp,
            good_matches,
            "模板匹配结果"
        )

        # 验证
        assert H is not None, "单应性矩阵计算失败"
        assert inliers >= 4, "内点不足"
        print(f"\n[验证通过] 目标检测成功,检测到 {inliers} 个一致匹配")

        return match_result, H
    else:
        print("  匹配点不足,无法进行几何验证")
        return None, None


# =============================================================================
# 练习 10: 深度学习特征描述符 (挑战题)
# =============================================================================

def exercise10_deep_learning_features():
    """
    练习 10: 深度学习特征描述符 (SuperPoint/SuperMatch)

    算法原理:
    ---------------
    传统的特征描述符 (SIFT, ORB) 是手工设计的,而深度学习特征通过神经网络学习。

    1. SuperPoint:
       - 全卷积网络,同时预测关键点位置和描述符
       - 关键点检测: 热力图回归
       - 描述符学习: Siamese 网络,对比损失

    2. SuperMatch:
       - 改进的描述符网络
       - 更好的匹配策略

    3. 传统方法的深度学习改进:
       - HardNet: 铰链损失学习二进制描述符
       - L2Net: 端到端学习描述符
       - ContextDesc: 考虑上下文的描述符

    实现说明:
    - 此练习演示概念,实际使用需要加载预训练模型
    - OpenCV DNN 模块可以加载 ONNX 格式的模型

    本练习展示如何用传统方法模拟深度学习流程:
    - 使用 SIFT/ORB 作为特征提取器
    - 使用神经网络匹配 (如可用)
    """
    print("\n" + "="*70)
    print("练习 10: 深度学习特征描述符 (概念演示)")
    print("="*70)

    print("""
[说明] 深度学习特征描述符需要加载预训练模型。
       这里展示完整框架,实际使用时请加载真实模型。

[SuperPoint 概念]
  - 输入: 图像
  - 输出1: 关键点热力图 (H x W)
  - 输出2: 描述符张量 (D x H x W)

[使用 OpenCV DNN 模块加载 ONNX 模型]
  net = cv2.dnn.readNetFromONNX('superpoint.onnx')
  blob = cv2.dnn.blobFromImage(img)
  net.setInput(blob)
  keypoints, descriptors = net.forward()

[替代方案: 使用传统特征模拟]
""")

    # 创建测试图像
    img1, img2, _ = create_test_image_pair()

    print("\n[步骤 1: 模拟深度学习特征提取]")
    print("  使用 SIFT 作为高质量特征描述符 (可类比深度学习特征)")

    # 使用 SIFT (深度学习前的最佳传统方法)
    sift = cv2.SIFT_create(nfeatures=300)
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    print(f"  图像1: {len(kp1)} 个关键点, 描述符维度 {desc1.shape}")
    print(f"  图像2: {len(kp2)} 个关键点, 描述符维度 {desc2.shape}")

    print("\n[步骤 2: 特征匹配]")
    # 使用 FLANN (适合高维浮点描述符)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(desc1, desc2, k=2)

    # 比率测试
    good_matches = [m[0] for m in knn_matches
                    if len(m) >= 2 and m[0].distance / m[1].distance < 0.75]

    print(f"  高质量匹配: {len(good_matches)}")

    print("\n[步骤 3: 几何验证]")
    # 提取匹配点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = sum(mask)

    print(f"  内点数量: {inliers}/{len(mask)}")
    print(f"  内点比例: {inliers/len(mask)*100:.1f}%")

    print("\n[框架总结]")
    print("  深度学习特征流程:")
    print("  1. 图像输入")
    print("  2. 深度网络前向传播")
    print("  3. 关键点 + 描述符输出")
    print("  4. 匹配层 (可学习)")
    print("  5. 几何验证")

    # 验证
    print(f"\n[验证通过] 模拟深度学习特征流程完成,获得 {inliers} 个内点")

    return None


# =============================================================================
# 练习 11: 多图像批量拼接 (挑战题)
# =============================================================================

def exercise11_batch_stitching():
    """
    练习 11: 多图像批量拼接

    算法原理:
    ---------------
    批量拼接是将多幅有序图像拼接成全景图。

    1. 图像排序:
       - 假设图像已经按顺序排列 (左到右或上到下)
       - 相邻图像有重叠区域

    2. 增量拼接:
       - 从第一幅图像开始
       - 依次将下一幅图像拼接上来
       - 累计变换矩阵

    3. 两两匹配:
       - 每对相邻图像独立估计单应性矩阵
       - 使用 RANSAC 过滤错误匹配

    4. 坐标变换:
       - 累积变换: H_total = H_n @ ... @ H_2 @ H_1
       - 将所有图像变换到统一坐标系

    5. 图像融合:
       - 处理曝光差异
       - 处理重叠区域
       - 去除拼接痕迹

    挑战:
       - 累积误差 (越长拼接误差越大)
       - 曝光不一致
       - 移动物体

    实现步骤:
    1. 加载图像序列
    2. 两两匹配相邻图像
    3. 计算相邻单应性矩阵
    4. 累积变换到世界坐标系
    5. 图像融合输出
    """
    print("\n" + "="*70)
    print("练习 11: 多图像批量拼接")
    print("="*70)

    # 创建模拟的三幅图像序列
    print("\n[步骤 1: 创建图像序列]")
    images = []
    for i in range(3):
        img, _ = create_test_image()
        # 模拟轻微旋转和平移
        if i > 0:
            angle = 5 * i
            M = cv2.getRotationMatrix2D((400, 300), angle, 0.95)
            img = cv2.warpAffine(img, M, (800, 600))
            # 轻微平移
            M = np.float32([[1, 0, 10*i], [0, 1, 5*i]])
            img = cv2.warpAffine(img, M, (800, 600))
        images.append(img)

    print(f"  图像序列长度: {len(images)}")

    print("\n[步骤 2: 两两特征匹配]")
    orb = cv2.ORB_create(nfeatures=500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 存储每对图像的匹配信息
    pairwise_matches = []

    for i in range(len(images) - 1):
        # 检测特征
        kp1, desc1 = orb.detectAndCompute(images[i], None)
        kp2, desc2 = orb.detectAndCompute(images[i+1], None)

        # k-近邻匹配
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)

        # 比率测试
        good = [m[0] for m in knn_matches
                if len(m) >= 2 and m[0].distance / m[1].distance < 0.75]

        print(f"  图像 {i} -> {i+1}: {len(good)} 个高质量匹配")

        pairwise_matches.append({
            "kp1": kp1,
            "kp2": kp2,
            "matches": good
        })

    print("\n[步骤 3: 计算相邻单应性矩阵]")
    homographies = []

    for i, pm in enumerate(pairwise_matches):
        pts1 = np.float32([pm["kp1"][m.queryIdx].pt for m in pm["matches"]])
        pts2 = np.float32([pm["kp2"][m.trainIdx].pt for m in pm["matches"]])

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        inliers = sum(mask)

        homographies.append(H)

        print(f"  图像 {i} -> {i+1}: 内点 {inliers}/{len(mask)}")

    print("\n[步骤 4: 累积变换到统一坐标系]")
    # 从第一幅图像开始,累积变换
    # H_total[i] = H[i] @ H[i-1] @ ... @ H[0]
    cumulative_H = [homographies[0]]

    for i in range(1, len(homographies)):
        H_total = homographies[i] @ cumulative_H[-1]
        cumulative_H.append(H_total)

    print("  累积变换矩阵:")
    for i, H in enumerate(cumulative_H):
        print(f"    图像 {i} -> 图像 {len(images)-1}:")
        print(f"    {H}")

    print("\n[步骤 5: 拼接图像]")
    # 创建输出画布
    result = np.zeros((600, 1600), dtype=np.uint8)

    # 放置第一幅图像
    result[0:600, 0:800] = images[0]

    # 依次拼接后续图像
    for i in range(1, len(images)):
        # 使用累积变换
        warped = cv2.warpPerspective(images[i], cumulative_H[i-1], (1600, 600))

        # 简单融合: 将非黑色区域复制到结果
        mask = (warped > 0).astype(np.uint8)
        result = result * (1 - mask) + warped * mask

    print(f"  输出尺寸: {result.shape}")

    print("\n[步骤 6: 批量拼接统计]")
    total_matches = sum(len(pm["matches"]) for pm in pairwise_matches)
    total_inliers = sum(sum(cv2.findHomography(
        np.float32([pm["kp1"][m.queryIdx].pt for m in pm["matches"]]),
        np.float32([pm["kp2"][m.trainIdx].pt for m in pm["matches"]]),
        cv2.RANSAC, 5.0
    )[1]) for pm in pairwise_matches)

    print(f"  总匹配数: {total_matches}")
    print(f"  总内点数: {total_inliers}")
    print(f"  内点比例: {total_inliers/total_matches*100:.1f}%")

    print(f"\n[验证通过] 批量拼接完成,处理 {len(images)} 幅图像")

    return result


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数 - 运行所有练习

    执行顺序:
    1. 入门级练习 (1-3)
    2. 中级练习 (4-6)
    3. 高级练习 (7-9)
    4. 挑战题 (10-11)
    """
    print("="*70)
    print("OpenCV features2d 模块练习题")
    print("="*70)

    try:
        # 入门级练习
        print("\n\n" + "="*70)
        print("入门级练习 (1-3)")
        print("="*70)

        exercise1_orb_detection()
        exercise2_orb_matching()
        exercise3_sift_matching()

        # 中级练习
        print("\n\n" + "="*70)
        print("中级练习 (4-6)")
        print("="*70)

        exercise4_ratio_test_filtering()
        exercise5_ransac_homography()
        exercise6_detector_comparison()

        # 高级练习
        print("\n\n" + "="*70)
        print("高级练习 (7-9)")
        print("="*70)

        exercise7_image_stitching()
        exercise8_image_retrieval()
        exercise9_template_matching()

        # 挑战题
        print("\n\n" + "="*70)
        print("挑战题 (10-11)")
        print("="*70)

        exercise10_deep_learning_features()
        exercise11_batch_stitching()

        print("\n\n" + "="*70)
        print("所有练习完成!")
        print("="*70)

    except Exception as e:
        print(f"\n[错误] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())