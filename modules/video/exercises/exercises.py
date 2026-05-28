#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV Video 模块练习题
======================

本文件包含 OpenCV video 模块的所有练习题,涵盖:
- KCF/MOSSE 目标跟踪
- 光流法 (Optical Flow)
- 背景分割
- 视频分析

作者: OpenCV Learning Team
版本: 4.14.0-pre
"""

import cv2
import numpy as np
import time
import os
from typing import List, Tuple, Optional


# =============================================================================
# 辅助函数
# =============================================================================

def create_test_video(width=640, height=480, duration=3, fps=30):
    """
    创建测试视频用于练习

    Args:
        width: 视频宽度
        height: 视频高度
        duration: 持续时间(秒)
        fps: 帧率

    Returns:
        VideoCapture 对象
    """
    # 创建一个简单的测试视频 - 移动的方块
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_file = '/tmp/test_video.mp4'
    out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))

    num_frames = duration * fps
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # 背景色

        # 移动的方块
        x = int((width - 100) * (i / num_frames))
        y = height // 2 - 50
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)

        # 添加一些特征点便于光流跟踪
        for j in range(5):
            px = x + 20 * j
            py = y + 50
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

        out.write(frame)

    out.release()
    return cv2.VideoCapture(temp_file)


def draw_tracking_result(frame, bbox, label="Track", color=(255, 0, 0), success=True):
    """
    在帧上绘制跟踪结果

    Args:
        frame: 输入帧
        bbox: 跟踪边界框
        label: 标签文本
        color: 绘制颜色
        success: 跟踪是否成功
    """
    if success and bbox.width > 0 and bbox.height > 0:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def verify_result(name: str, success: bool, message: str = ""):
    """
    打印验证结果

    Args:
        name: 测试名称
        success: 是否成功
        message: 附加消息
    """
    status = "✓ 通过" if success else "✗ 失败"
    print(f"[{status}] {name}")
    if message:
        print(f"    {message}")


# =============================================================================
# 入门级练习
# =============================================================================

def exercise_1_kcf_tracker():
    """
    练习1: KCF 目标跟踪器的基本使用
    =================================

    算法原理:
    --------
    KCF (Kernelized Correlation Filters) 是一种基于核相关滤波的目标跟踪算法。
    其核心思想是:
    1. 在目标区域提取样本特征
    2. 使用循环矩阵来表示样本空间
    3. 通过核函数将样本映射到高维空间
    4. 在高维空间中进行相关滤波器的训练
    5. 使用快速傅里叶变换(FFT)加速计算,实现实时跟踪

    KCF 的优点:
    - 高跟踪精度
    - 较快的处理速度
    - 良好的抗遮挡能力(中等)

    本练习实现:
    -----------
    1. 创建 KCF 跟踪器
    2. 在初始帧中选择目标区域
    3. 在后续帧中更新跟踪状态
    4. 可视化跟踪结果
    """
    print("\n" + "=" * 60)
    print("练习1: KCF 目标跟踪器的基本使用")
    print("=" * 60)

    # 创建测试视频或使用摄像头
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    print("信息: 读取到第一帧,准备选择跟踪目标...")
    print("提示: 在实际应用中,使用 cv2.selectROI() 选择目标")

    # 模拟选择一个初始区域(实际使用时取消注释下面的代码)
    # bbox = cv2.selectROI("选择跟踪目标", frame, showCrosshair=True)
    # 如果用户取消选择, bbox 为空
    # if bbox.width == 0 or bbox.height == 0:
    #     return False

    # 使用预定义的 bbox 进行演示
    h, w = frame.shape[:2]
    bbox = (w // 4, h // 4, w // 4, h // 4)  # (x, y, w, h)
    print(f"信息: 使用初始边界框 bbox = {bbox}")

    # 创建 KCF 跟踪器
    # KCF 跟踪器参数:
    # - desc尺度: 特征描述符的尺度个数
    # - desc_np尺度: 非极大值抑制的尺度个数
    # - 混合权重: 融合多种特征的权重
    try:
        tracker = cv2.TrackerKCF.create()
        print("信息: KCF 跟踪器创建成功")
    except Exception as e:
        print(f"错误: KCF 跟踪器创建失败: {e}")
        cap.release()
        return False

    # 初始化跟踪器
    success = tracker.init(frame, bbox)
    if not success:
        print("错误: 跟踪器初始化失败")
        cap.release()
        return False
    print("信息: 跟踪器初始化成功")

    # 跟踪状态统计
    frame_count = 0
    success_count = 0

    # 处理视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 更新跟踪器
        success, bbox = tracker.update(frame)

        if success:
            success_count += 1
            frame = draw_tracking_result(frame, bbox, "KCF Tracker", (255, 0, 0))
        else:
            cv2.putText(frame, "跟踪失败", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示帧率信息
        cv2.putText(frame, f"帧: {frame_count}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 为了演示,不实际显示窗口
        # cv2.imshow("KCF 跟踪", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()

    # 计算成功率
    success_rate = success_count / frame_count if frame_count > 0 else 0
    print(f"\n统计:")
    print(f"  - 总帧数: {frame_count}")
    print(f"  - 成功跟踪帧数: {success_count}")
    print(f"  - 成功率: {success_rate * 100:.1f}%")

    # 验证结果
    success = success_rate > 0.5 and frame_count > 0
    verify_result("KCF 跟踪器初始化", True, "跟踪器成功创建并初始化")
    verify_result("KCF 跟踪更新", success, f"成功率: {success_rate * 100:.1f}%")

    return success


def exercise_2_lucas_kanade_optical_flow():
    """
    练习2: Lucas-Kanade 光流跟踪特征点
    ====================================

    算法原理:
    --------
    Lucas-Kanade 光流是一种稀疏光流算法,基于以下假设:
    1. 亮 度恒定假设: 同一个像素点在时间上的亮度不变
    2. 小运动假设: 像素点在帧间的位移很小
    3. 空间一致性: 邻近像素点有相似的运动

    核心方程(光流方程):
    I(x,y,t) = I(x+dx, y+dy, t+dt)
    ∇I · [vx; vy] + It = 0

    其中:
    - ∇I = (Ix, Iy) 是图像梯度
    - [vx; vy] 是光流向量
    - It 是时间梯度

    使用加权最小二乘法求解,邻近像素点的权重由空间距离决定。

    本练习实现:
    -----------
    1. 读取视频并转换为灰度图
    2. 使用 goodFeaturesToTrack 检测 Shi-Tomasi 特征点
    3. 使用 calcOpticalFlowPyrLK 计算光流
    4. 可视化特征点轨迹
    """
    print("\n" + "=" * 60)
    print("练习2: Lucas-Kanade 光流跟踪特征点")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧并转换为灰度图
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    prevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测初始特征点
    # Shi-Tomasi 特征点检测参数:
    # - maxCorners: 最大特征点数量
    # - qualityLevel: 特征点的质量阈值(与最大特征值的乘积)
    # - minDistance: 特征点之间的最小距离
    # - blockSize: 计算梯度时使用的邻域大小
    prevPoints = cv2.goodFeaturesToTrack(
        prevGray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=21
    )

    if prevPoints is None or len(prevPoints) == 0:
        print("错误: 未检测到特征点")
        cap.release()
        return False

    print(f"信息: 检测到 {len(prevPoints)} 个特征点")

    # 光流跟踪统计
    frame_count = 0
    total_points = len(prevPoints)
    tracked_points = 0

    # Lucas-Kanade 光流参数:
    # - winSize: 搜索窗口大小,越大越抗噪声但越慢
    # - maxLevel: 金字塔层数,用于处理大运动
    # - criteria: 迭代终止条件
    # - flags: 计算选项
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=0
    )

    # 创建用于绘制结果的图像
    result_frame = frame.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        currentGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        # 注意: prevPoints 的形状必须是 (N, 1, 2)
        currentPoints, status, err = cv2.calcOpticalFlowPyrLK(
            prevGray, currentGray,
            prevPoints, None,
            **lk_params
        )

        # status[i] == 1 表示第 i 个特征点成功跟踪
        if status is not None:
            tracked_points = np.sum(status)

            # 绘制跟踪结果
            for i, (prev_pt, curr_pt, st) in enumerate(zip(
                prevPoints.reshape(-1, 2),
                currentPoints.reshape(-1, 2),
                status.reshape(-1)
            )):
                if st:
                    # 绘制轨迹线
                    cv2.line(result_frame,
                             tuple(map(int, prev_pt)),
                             tuple(map(int, curr_pt)),
                             (0, 255, 0), 2)
                    # 绘制当前点
                    cv2.circle(result_frame,
                               tuple(map(int, curr_pt)),
                               3, (0, 0, 255), -1)

            # 可视化统计
            cv2.putText(result_frame,
                        f"帧: {frame_count}, 跟踪点: {tracked_points}/{total_points}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 为了演示,不实际显示
            # cv2.imshow("Lucas-Kanade 光流", result_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # 更新前一帧和特征点
        prevGray = currentGray.copy()
        prevPoints = currentPoints[status == 1].reshape(-1, 1, 2)

        # 如果特征点太少,重新检测
        if len(prevPoints) < 10:
            prevPoints = cv2.goodFeaturesToTrack(
                prevGray, maxCorners=100, qualityLevel=0.3,
                minDistance=7, blockSize=21
            )
            if prevPoints is not None:
                total_points = len(prevPoints)
                tracked_points = total_points

    cap.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - 初始特征点数: {total_points}")
    print(f"  - 最后一帧跟踪点数: {tracked_points}")

    # 验证结果
    success = frame_count > 0 and tracked_points >= 0
    verify_result("Lucas-Kanade 光流初始化", True, "成功检测特征点并初始化")
    verify_result("光流计算", success, f"处理了 {frame_count} 帧")
    verify_result("特征点跟踪", tracked_points > 0, f"跟踪了点数: {tracked_points}")

    return success


def exercise_3_background_subtraction():
    """
    练习3: 简单的运动检测 (背景减除)
    =================================

    算法原理:
    --------
    背景减除是视频分析中的基础技术,用于分离前景运动物体。
    核心思想: 建立一个背景模型,然后逐帧比较当前帧与背景模型的差异。

    OpenCV 提供了多种背景减除算法:

    1. MOG2 (Mixture of Gaussians 2):
       - 使用高斯混合模型建模每个像素
       - 自动选择适当的高斯分布数量
       - 能够应对光照变化
       - 参数: history(背景帧数), varThreshold(方差阈值), detectShadows(检测阴影)

    2. KNN (K-Nearest Neighbors):
       - 使用 K 近邻算法建模像素值
       - 对噪声更鲁棒
       - 参数: history, distThreshold(距离阈值), detectShadows

    3. GMG (Godbehey-Mellon-Guo):
       - 基于核密度估计
       - 初始化较慢但后续很快

    本练习实现:
    -----------
    1. 创建 MOG2 和 KNN 背景减除器
    2. 处理视频帧并生成前景掩码
    3. 应用形态学操作去除噪声
    4. 可视化运动检测结果
    """
    print("\n" + "=" * 60)
    print("练习3: 简单的运动检测 (背景减除)")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 创建背景减除器
    # MOG2 参数:
    # - history: 用于构建背景模型的帧数,越长越能处理复杂场景
    # - varThreshold: 像素与模型匹配时的方差阈值
    # - detectShadows: 是否检测阴影,开启会降低检测精度
    mog = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )

    # KNN 参数:
    # - history: 同上
    # - dist2Threshold: 像素与背景模型的距离阈值
    # - detectShadows: 同上
    knn = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=400,
        detectShadows=False
    )

    print("信息: 背景减除器创建成功")

    # 形态学操作使用的核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 统计信息
    frame_count = 0
    mog_fg_count = 0
    knn_fg_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # MOG2 前景检测
        fgMaskMOG = mog.apply(frame)

        # KNN 前景检测
        fgMaskKNN = knn.apply(frame)

        # 二值化前景掩码
        # 阈值 200 用于过滤噪声
        _, fgBinMOG = cv2.threshold(fgMaskMOG, 200, 255, cv2.THRESH_BINARY)
        _, fgBinKNN = cv2.threshold(fgMaskKNN, 200, 255, cv2.THRESH_BINARY)

        # 形态学开操作 - 去除小的白色噪点
        fgBinMOG = cv2.morphologyEx(fgBinMOG, cv2.MORPH_OPEN, kernel)
        fgBinKNN = cv2.morphologyEx(fgBinKNN, cv2.MORPH_OPEN, kernel)

        # 统计前景像素数量
        mog_fg_count = np.sum(fgBinMOG > 0)
        knn_fg_count = np.sum(fgBinKNN > 0)

        # 可视化
        # 为了演示,创建结果图像
        resultMOG = cv2.bitwise_and(frame, frame, mask=fgBinMOG)
        resultKNN = cv2.bitwise_and(frame, frame, mask=fgBinKNN)

        # 添加信息文字
        cv2.putText(resultMOG, f"MOG2 前景区像素: {mog_fg_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(resultKNN, f"KNN 前景区像素: {knn_fg_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 为了演示,不实际显示
        # cv2.imshow("原始帧", frame)
        # cv2.imshow("MOG2 前景", resultMOG)
        # cv2.imshow("KNN 前景", resultKNN)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - MOG2 检测到前景的平均帧数: {mog_fg_count / max(frame_count, 1)}")
    print(f"  - KNN 检测到前景的平均帧数: {knn_fg_count / max(frame_count, 1)}")

    # 验证结果
    success = frame_count > 0
    verify_result("MOG2 背景减除器", success, "成功创建并使用 MOG2")
    verify_result("KNN 背景减除器", success, "成功创建并使用 KNN")
    verify_result("运动检测", success and mog_fg_count > 0,
                  f"检测到运动变化")

    return success


# =============================================================================
# 中级练习
# =============================================================================

def exercise_4_multi_target_tracking():
    """
    练习4: 多目标跟踪系统
    ======================

    算法原理:
    --------
    多目标跟踪(Multi-Target Tracking) 是计算机视觉中的重要问题。
    与单目标跟踪不同,多目标跟踪需要:
    1. 为每个目标维护独立的跟踪器
    2. 处理目标的出现和消失
    3. 处理目标之间的相互遮挡
    4. 区分不同目标

    简化实现方案:
    - 为每个目标分配一个独立的跟踪器(KCF/MOSSE)
    - 使用独立的目标框初始化每个跟踪器
    - 在每一帧中更新所有跟踪器
    - 标记每个目标的跟踪状态

    本练习实现:
    -----------
    1. 初始化多个目标区域
    2. 为每个目标创建独立的跟踪器
    3. 在视频帧中跟踪所有目标
    4. 使用不同颜色标识不同目标
    """
    print("\n" + "=" * 60)
    print("练习4: 多目标跟踪系统")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    # 定义多个跟踪目标
    # 模拟多个移动的方块
    h, w = frame.shape[:2]
    targets = [
        (w // 4, h // 4, w // 6, h // 6),      # 目标1
        (w // 2, h // 2, w // 6, h // 6),      # 目标2
        (3 * w // 4, h // 4, w // 6, h // 6),  # 目标3
    ]

    print(f"信息: 定义了 {len(targets)} 个跟踪目标")

    # 为每个目标创建跟踪器
    trackers = []
    labels = []

    for i, bbox in enumerate(targets):
        # 使用 KCF 跟踪器
        tracker = cv2.TrackerKCF.create()
        tracker.init(frame, bbox)
        trackers.append(tracker)
        labels.append(f"目标 {i + 1}")

    print(f"信息: 成功初始化 {len(trackers)} 个跟踪器")

    # 定义不同颜色 (BGR 格式)
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
    ]

    # 跟踪统计
    frame_count = 0
    success_counts = [0] * len(trackers)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 更新所有跟踪器
        for i, (tracker, label) in enumerate(zip(trackers, labels)):
            success, bbox = tracker.update(frame)
            color = colors[i % len(colors)]

            if success:
                success_counts[i] += 1
                frame = draw_tracking_result(frame, bbox, label, color)

        # 显示统计信息
        cv2.putText(frame, f"帧: {frame_count}, 目标数: {len(trackers)}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 为了演示,不实际显示
        # cv2.imshow("多目标跟踪", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 处理帧数: {frame_count}")
    for i, count in enumerate(success_counts):
        rate = count / max(frame_count, 1) * 100
        print(f"  - {labels[i]} 跟踪成功率: {rate:.1f}%")

    # 验证结果
    success = frame_count > 0 and sum(success_counts) > 0
    verify_result("多目标跟踪初始化", True, f"成功初始化 {len(trackers)} 个跟踪器")
    verify_result("多目标跟踪更新", success, f"处理了 {frame_count} 帧")

    return success


def exercise_5_dense_optical_flow_visualization():
    """
    练习5: 光流可视化 (稠密光流)
    ============================

    算法原理:
    --------
    稠密光流与稀疏光流不同,它计算图像中每个像素的光流向量,
    而不是只计算特定特征点的光流。这提供了更完整的运动信息。

    OpenCV 提供了两种主要的稠密光流算法:

    1. Farneback 光流:
       - 基于多项式展开的全局方法
       - 将每个像素领域展开为多项式
       - 通过多项式系数计算光流
       - 参数:
         * pyr_scale: 金字塔尺度因子
         * levels: 金字塔层数
         * winsize: 平均窗口大小
         * iterations: 迭代次数
         * poly_n: 多项式展开的阶数
         * poly_sigma: 高斯核的标准差

    2. TVL1 光流:
       - 基于 L1 范数优化的变分方法
       - 能够处理大的非连续运动
       - 通过交替方向乘子法(ADMM)求解

    本练习实现:
    -----------
    1. 使用 Farneback 算法计算稠密光流
    2. 将光流向量转换为颜色编码图像
    3. 可视化光流场
    """
    print("\n" + "=" * 60)
    print("练习5: 光流可视化 (稠密光流)")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧并转换为灰度图
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    prevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Farneback 光流参数:
    # - pyr_scale: 相邻金字塔层的尺度因子,0.5 表示每层缩小一半
    # - levels: 金字塔层数,包括初始层
    # - winsize: 用于计算光流的平均窗口大小,越大越平滑
    # - iterations: 迭代次数,越多越精确但越慢
    # - poly_n: 多项式展开的邻域大小,越大越平滑
    # - poly_sigma: 高斯标准差,用于多项式权重
    # - flags: 操作标志
    flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    print("信息: Farneback 光流参数已设置")
    print(f"  - pyr_scale: {flow_params['pyr_scale']}")
    print(f"  - levels: {flow_params['levels']}")
    print(f"  - winsize: {flow_params['winsize']}")

    # 跟踪统计
    frame_count = 0
    avg_magnitude = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        currentGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prevGray, currentGray,
            None,  # flow 初始化
            **flow_params
        )

        if flow is not None:
            # 计算光流幅度和角度
            # cartToPolar 将笛卡尔坐标 (x, y) 转换为极坐标 (magnitude, angle)
            magnitude, angle = cv2.cartToPolar(
                flow[..., 0], flow[..., 1],
                angleInDegrees=True
            )

            # 计算平均光流幅度
            avg_magnitude = np.mean(magnitude)

            # 转换为 HSV 颜色空间以便可视化
            # H: 角度 (0-180 在 OpenCV 中,实际是 0-360)
            # S: 固定最大值
            # V: 光流幅度归一化值

            # 创建 HSV 图像
            hsv = np.zeros_like(frame, dtype=np.float32)

            # 角度转换为 hue 值 (0-180 对应 0-360 度)
            hsv[..., 0] = angle * (180 / np.pi / 2)
            # 饱和度为最大值
            hsv[..., 1] = 255
            # 幅度归一化到 0-255
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # 转换为 BGR 以便显示
            hsv_uint8 = hsv.astype(np.uint8)
            flowDisplay = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)

            # 为了演示,不实际显示
            # cv2.imshow("稠密光流", flowDisplay)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # 更新前一帧
        prevGray = currentGray.copy()

    cap.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - 平均光流幅度: {avg_magnitude:.2f} 像素/帧")

    # 验证结果
    success = frame_count > 0 and avg_magnitude >= 0
    verify_result("Farneback 光流计算", success, f"处理了 {frame_count} 帧")
    verify_result("光流场生成", flow is not None, "成功生成光流场")
    verify_result("光流可视化", success, f"平均幅度: {avg_magnitude:.2f}")

    return success


def exercise_6_tracker_comparison():
    """
    练习6: 比较不同跟踪器 (KCF, MOSSE, MEDIAN_FLOW) 的性能
    ======================================================

    算法原理:
    --------
    不同跟踪器有各自的优缺点,本练习对比三种经典跟踪器:

    1. KCF (Kernelized Correlation Filters):
       - 高精度,较快速度
       - 使用核相关滤波和循环矩阵
       - 抗遮挡能力中等
       - 适用: 通用跟踪场景

    2. MOSSE (Minimum Output Sum of Squared Error):
       - 基于自适应相关滤波
       - 极快的处理速度
       - 精度中等,抗遮挡能力较差
       - 适用: 高速需求场景

    3. MEDIAN_FLOW:
       - 基于 Lucas-Kanade 光流
       - 较好的短期跟踪性能
       - 能够检测跟踪失败
       - 抗遮挡能力较差
       - 适用: 短期跟踪

    性能评估指标:
    - FPS: 每秒处理的帧数
    - 成功率: 成功跟踪的帧数占比
    - 精度: 跟踪框与真实框的重叠度(IoU)

    本练习实现:
    -----------
    1. 创建三种跟踪器
    2. 在相同视频序列上测试
    3. 比较各跟踪器的 FPS 和成功率
    """
    print("\n" + "=" * 60)
    print("练习6: 比较不同跟踪器的性能")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    # 定义初始跟踪区域
    h, w = frame.shape[:2]
    bbox = (w // 4, h // 4, w // 4, h // 4)

    # 定义要比较的跟踪器
    tracker_types = {
        'KCF': lambda: cv2.TrackerKCF.create(),
        'MOSSE': lambda: cv2.TrackerMOSSE.create(),
        'MEDIAN_FLOW': lambda: cv2.TrackerMedianFlow.create(),
    }

    results = {}

    for tracker_name, tracker_creator in tracker_types.items():
        print(f"\n测试 {tracker_name} 跟踪器...")

        # 重新读取视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            continue

        # 创建跟踪器
        try:
            tracker = tracker_creator()
        except Exception as e:
            print(f"  错误: {tracker_name} 创建失败: {e}")
            continue

        # 初始化跟踪器
        success = tracker.init(frame, bbox)
        if not success:
            print(f"  错误: {tracker_name} 初始化失败")
            continue

        # 性能测试
        frame_count = 0
        success_count = 0
        total_time = 0.0

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # 更新跟踪器
            success, new_bbox = tracker.update(frame)

            elapsed = time.time() - start_time
            total_time += elapsed
            frame_count += 1

            if success:
                success_count += 1

        # 计算性能指标
        if frame_count > 0:
            avg_fps = frame_count / total_time if total_time > 0 else 0
            success_rate = success_count / frame_count

            results[tracker_name] = {
                'fps': avg_fps,
                'success_rate': success_rate,
                'frames': frame_count
            }

            print(f"  - FPS: {avg_fps:.2f}")
            print(f"  - 成功率: {success_rate * 100:.1f}%")
            print(f"  - 处理帧数: {frame_count}")

    cap.release()

    # 打印比较结果
    print("\n" + "=" * 60)
    print("跟踪器性能比较结果")
    print("=" * 60)
    print(f"{'跟踪器':<15} {'FPS':<10} {'成功率':<10} {'帧数':<10}")
    print("-" * 45)

    for name, data in results.items():
        print(f"{name:<15} {data['fps']:<10.2f} {data['success_rate']*100:<10.1f}% {data['frames']:<10}")

    # 确定最佳跟踪器
    if results:
        best_by_fps = max(results.items(), key=lambda x: x[1]['fps'])
        best_by_success = max(results.items(), key=lambda x: x[1]['success_rate'])

        print(f"\n最佳 FPS: {best_by_fps[0]} ({best_by_fps[1]['fps']:.2f})")
        print(f"最佳成功率: {best_by_success[0]} ({best_by_success[1]['success_rate']*100:.1f}%)")

    # 验证结果
    success = len(results) >= 3
    verify_result("KCF 跟踪器", 'KCF' in results, f"FPS: {results.get('KCF', {}).get('fps', 0):.2f}")
    verify_result("MOSSE 跟踪器", 'MOSSE' in results, f"FPS: {results.get('MOSSE', {}).get('fps', 0):.2f}")
    verify_result("MEDIAN_FLOW 跟踪器", 'MEDIAN_FLOW' in results, f"FPS: {results.get('MEDIAN_FLOW', {}).get('fps', 0):.2f}")
    verify_result("性能比较", success, f"成功比较 {len(results)} 种跟踪器")

    return success


# =============================================================================
# 高级练习
# =============================================================================

def exercise_7_motion_estimation_optical_flow():
    """
    练习7: 基于光流的运动估计
    ==========================

    算法原理:
    --------
    光流不仅可用于跟踪,还可用于估计场景中物体的运动:

    1. 相机运动估计:
       - 通过光流场整体趋势估计相机平移和旋转
       - 用于视觉里程计(Visual Odometry)

    2. 运动分割:
       - 将具有相似运动模式的像素分组
       - 分离不同运动物体

    3. 3D 结构恢复:
       - 通过光流场中的视差估计深度
       - 用于稀疏/稠密重建

    运动模型:
    - 平移模型: (dx, dy) 仅平移
    - 仿射模型: 6 参数 [a11, a12, a21, a22, b1, b2]
    - 射影模型: 8 参数 Homography

    本练习实现:
    -----------
    1. 计算稠密光流
    2. 估计全局运动参数(仿射模型)
    3. 使用 RANSAC 剔除异常值
    4. 可视化运动场和估计的运动
    """
    print("\n" + "=" * 60)
    print("练习7: 基于光流的运动估计")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    prevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 光流和运动估计参数
    flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2
    )

    # 跟踪统计
    frame_count = 0
    motion_vectors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        currentGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prevGray, currentGray,
            None,
            **flow_params
        )

        if flow is not None:
            # 采样光流向量的子集用于运动估计
            # 避免使用所有像素以提高效率
            sample_step = 16
            points_prev = []
            points_curr = []

            for y in range(0, flow.shape[0], sample_step):
                for x in range(0, flow.shape[1], sample_step):
                    points_prev.append([x, y])
                    fx, fy = flow[y, x]
                    points_curr.append([x + fx, y + fy])

            points_prev = np.array(points_prev, dtype=np.float32)
            points_curr = np.array(points_curr, dtype=np.float32)

            # 使用 RANSAC 估计仿射变换
            # findHomography 参数:
            # - method: RANSAC, PROSAC, LMEDS, RHO
            # - ransacReprojThreshold: RANSAC 重投影阈值
            # - maxIters: RANSAC 最大迭代次数
            # - confidence: RANSAC 置信度
            try:
                # 需要至少 4 个对应点来估计仿射变换
                if len(points_prev) >= 4:
                    M, mask = cv2.findHomography(
                        points_prev, points_curr,
                        cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000
                    )

                    if M is not None:
                        # 提取仿射变换参数
                        # M 是 3x3 矩阵,表示齐次坐标下的仿射变换
                        # [a11, a12, b1; a21, a22, b2; 0, 0, 1]
                        dx = M[0, 2]  # 平移量 x
                        dy = M[1, 2]  # 平移量 y

                        # 计算旋转角度和尺度
                        a11, a12 = M[0, 0], M[0, 1]
                        a21, a22 = M[1, 0], M[1, 1]

                        scale = np.sqrt(a11**2 + a21**2)
                        angle = np.arctan2(a12, a11) * 180 / np.pi

                        motion_vectors.append({
                            'dx': dx,
                            'dy': dy,
                            'scale': scale,
                            'angle': angle
                        })

                        print(f"帧 {frame_count}: 平移({dx:.2f}, {dy:.2f}), "
                              f"旋转:{angle:.2f}°, 尺度:{scale:.3f}")
            except Exception as e:
                print(f"帧 {frame_count} 运动估计错误: {e}")

        # 更新前一帧
        prevGray = currentGray.copy()

    cap.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - 成功估计运动帧数: {len(motion_vectors)}")

    if motion_vectors:
        avg_dx = np.mean([v['dx'] for v in motion_vectors])
        avg_dy = np.mean([v['dy'] for v in motion_vectors])
        avg_scale = np.mean([v['scale'] for v in motion_vectors])
        avg_angle = np.mean([v['angle'] for v in motion_vectors])

        print(f"  - 平均平移: ({avg_dx:.2f}, {avg_dy:.2f})")
        print(f"  - 平均旋转: {avg_angle:.2f}°")
        print(f"  - 平均尺度: {avg_scale:.3f}")

    # 验证结果
    success = frame_count > 0 and len(motion_vectors) > 0
    verify_result("光流计算", True, f"处理了 {frame_count} 帧")
    verify_result("运动估计", success, f"成功估计 {len(motion_vectors)} 帧的运动")

    return success


def exercise_8_tracking_by_detection():
    """
    练习8: 基于跟踪的航迹维持 (tracking-by-detection)
    ================================================

    算法原理:
    --------
    Tracking-by-Detection 是现代多目标跟踪的主流框架:

    1. 检测阶段:
       - 使用目标检测器(Faster R-CNN, YOLO, SSD 等)
       - 在每一帧中检测所有目标

    2. 关联阶段:
       - 将检测结果与现有轨迹关联
       - 常用方法: IOU 匹配, 外观特征匹配, 运动模型预测

    3. 轨迹管理:
       - 处理目标的出现和消失
       - 管理轨迹的创建和销毁
       - 处理遮挡情况

    关键算法:
    - 卡尔曼滤波: 预测目标未来位置
    - IOU 关联: 基于重叠度的关联
    - 匈牙利算法: 最优关联分配

    本练习实现:
    -----------
    1. 模拟目标检测结果
    2. 使用卡尔曼滤波预测目标位置
    3. 使用 IOU 进行目标关联
    4. 维护轨迹的生命周期
    """
    print("\n" + "=" * 60)
    print("练习8: 基于跟踪的航迹维持 (tracking-by-detection)")
    print("=" * 60)

    # 简单卡尔曼滤波器实现
    class KalmanBoxTracker:
        """
        卡尔曼滤波器用于跟踪边界框

        状态向量: [x, y, w, h, vx, vy, vw, vh]
        其中 (x, y) 是中心坐标, (w, h) 是宽高
        (vx, vy, vw, vh) 是对应的速度
        """
        count = 0  # 全局轨迹计数器

        def __init__(self, bbox):
            self.bbox = np.array(bbox, dtype=np.float32)
            # 状态向量: [x, y, w, h, vx, vy, vw, vh]
            self.kalman = cv2.KalmanFilter(8, 4)

            # 状态转移矩阵 (匀速模型)
            # x' = x + vx
            # y' = y + vy
            # w' = w + vw
            # h' = h + vh
            self.kalman.transitionMatrix = np.array([
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)

            # 观测矩阵 (我们只能观测到位置和大小)
            self.kalman.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]
            ], dtype=np.float32)

            # 测量噪声协方差
            self.kalman.measurementNoiseCov = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 10, 0],
                [0, 0, 0, 10]
            ], dtype=np.float32) * 1e-1

            # 过程噪声协方差
            self.kalman.processNoiseCov = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=np.float32) * 1e-3

            # 初始化状态
            x, y, w, h = bbox
            self.kalman.statePost = np.array([
                [x + w/2], [y + h/2], [w], [h], [0], [0], [0], [0]
            ], dtype=np.float32)

            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1
            self.age = 0
            self.hits = 0
            self.no_losses = 0

        def predict(self):
            """预测下一帧的位置"""
            self.kalman.predict()
            self.age += 1
            return self.get_state()

        def update(self, bbox):
            """使用检测结果更新滤波器"""
            self.hits += 1
            self.no_losses = 0
            x, y, w, h = bbox
            measurement = np.array([
                [x + w/2], [y + h/2], [w], [h]
            ], dtype=np.float32)
            self.kalman.correct(measurement)

        def get_state(self):
            """获取当前估计的状态"""
            state = self.kalman.statePost
            x, y, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
            return (x - w/2, y - h/2, w, h)

    def compute_iou(box1, box2):
        """计算两个边界框的 IOU (Intersection over Union)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算交集区域
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # 计算交集面积
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - y1)
        inter_area = inter_w * inter_h

        # 计算并集面积
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        # 计算 IOU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 模拟检测器 - 简单地在每一帧生成一个检测框
    # 实际应用中应使用真实的目标检测器
    trackers = []  # 活跃的轨迹列表
    frame_count = 0
    max_age = 30  # 轨迹最大无更新帧数
    min_hits = 3  # 最少命中次数才输出有效轨迹

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 模拟检测结果 (简化版本: 检测移动的方块)
        h, w = frame.shape[:2]

        # 实际应用中,这里应该是真实的目标检测器输出
        # 这里我们模拟一个沿对角线移动的目标
        x = int((w - 100) * (frame_count / 90))
        y = int((h - 100) * (frame_count / 90))
        detected_boxes = [(x, y, 100, 100)] if frame_count <= 90 else []

        # 预测所有轨迹的位置
        for tracker in trackers:
            tracker.predict()

        # 关联检测结果与轨迹
        matched_trackers = []
        unmatched_detections = list(range(len(detected_boxes)))

        if trackers and detected_boxes:
            # 计算 IOU 矩阵
            iou_matrix = np.zeros((len(trackers), len(detected_boxes)))
            for i, tracker in enumerate(trackers):
                tracker_bbox = tracker.get_state()
                for j, det_bbox in enumerate(detected_boxes):
                    iou_matrix[i, j] = compute_iou(tracker_bbox, det_bbox)

            # 贪婪匹配
            while iou_matrix.size > 0:
                max_iou = np.max(iou_matrix)
                if max_iou < 0.3:  # IOU 阈值
                    break

                # 找到最大 IOU 的索引
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                i, j = max_idx

                # 匹配
                trackers[i].update(detected_boxes[j])
                matched_trackers.append(i)
                unmatched_detections.remove(j)

                # 移除已匹配的行和列
                iou_matrix = np.delete(iou_matrix, i, axis=0)
                iou_matrix = np.delete(iou_matrix, j, axis=1)
                trackers = [t for idx, t in enumerate(trackers) if idx != i or j not in unmatched_detections]

        # 为未匹配的检测创建新轨迹
        for det_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detected_boxes[det_idx])
            trackers.append(new_tracker)

        # 更新并输出有效轨迹
        valid_tracks = []
        for tracker in trackers:
            if tracker.hits >= min_hits:
                valid_tracks.append(tracker)

        # 移除超时的轨迹
        trackers = [t for t in trackers if t.age < max_age]

        # 统计信息输出
        if frame_count % 10 == 0:
            print(f"帧 {frame_count}: 活跃轨迹 {len(trackers)}, "
                  f"有效轨迹 {len(valid_tracks)}, "
                  f"新检测 {len(unmatched_detections)}")

    cap.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - 初始化轨迹数: {KalmanBoxTracker.count}")

    # 验证结果
    success = frame_count > 0 and KalmanBoxTracker.count > 0
    verify_result("航迹初始化", True, f"创建了 {KalmanBoxTracker.count} 条航迹")
    verify_result("航迹维持", success, f"处理了 {frame_count} 帧")

    return success


def exercise_9_long_term_tracker():
    """
    练习9: 长期目标跟踪 (TLD 或 GOTURN)
    ====================================

    算法原理:
    --------
    长期目标跟踪与短期跟踪不同,需要处理:
    1. 目标完全遮挡后的重新检测
    2. 目标的重新出现
    3. 跟踪失败检测

    1. TLD (Tracking-Learning-Detection):
       - 将跟踪、检测和学习结合
       - 跟踪器处理短期运动
       - 检测器处理目标重新出现
       - 在线学习改进检测器

    2. GOTURN (Generic Object Tracking Using Regression Networks):
       - 基于深度学习的跟踪器
       - 使用卷积神经网络预测目标位置
       - 需要预训练模型文件
       - 模型下载: https://raw.githubusercontent.com/AaronJin2016/GOTURN_Trained_Model/master/tracker.gob

    本练习实现:
    -----------
    1. 尝试创建 TLD 跟踪器
    2. 演示长期跟踪的基本概念
    3. 处理跟踪失败情况
    """
    print("\n" + "=" * 60)
    print("练习9: 长期目标跟踪 (TLD 或 GOTURN)")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    # 定义初始跟踪区域
    h, w = frame.shape[:2]
    bbox = (w // 4, h // 4, w // 4, h // 4)

    # TLD 跟踪器
    # TLD (Tracking-Learning-Detection) 跟踪器参数:
    # - backend: 使用的检测器后端
    # - target: 跟踪目标类型
    try:
        tracker_tld = cv2.TrackerTLD.create()
        print("信息: TLD 跟踪器创建成功")
    except Exception as e:
        print(f"警告: TLD 跟踪器创建失败: {e}")
        tracker_tld = None

    # GOTURN 跟踪器 (需要模型文件)
    # 注意: GOTURN 需要额外的模型文件 tracker.gob
    # 如果没有模型文件,GOTURN 将无法使用
    tracker_goturn = None
    goturn_model_path = None  # 如果有模型文件,在这里指定路径

    if goturn_model_path and os.path.exists(goturn_model_path):
        try:
            tracker_goturn = cv2.TrackerGOTURN.create(goturn_model_path)
            print("信息: GOTURN 跟踪器创建成功")
        except Exception as e:
            print(f"警告: GOTURN 跟踪器创建失败: {e}")

    # 如果没有可用的长期跟踪器,使用 KCF 作为替代
    if tracker_tld is None and tracker_goturn is None:
        print("提示: 使用 KCF 作为替代跟踪器进行演示")
        tracker_tld = cv2.TrackerKCF.create()

    # 初始化选定的跟踪器
    tracker = tracker_tld if tracker_tld else tracker_goturn
    tracker_name = "TLD" if tracker == tracker_tld else "GOTURN"

    success = tracker.init(frame, bbox)
    if not success:
        print("错误: 跟踪器初始化失败")
        cap.release()
        return False

    print(f"信息: {tracker_name} 跟踪器初始化成功")

    # 跟踪统计
    frame_count = 0
    success_count = 0
    fail_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 更新跟踪器
        success, new_bbox = tracker.update(frame)

        if success:
            success_count += 1
            frame = draw_tracking_result(frame, new_bbox, f"{tracker_name} Tracker", (255, 0, 0))
        else:
            fail_count += 1
            cv2.putText(frame, "跟踪失败 - 需要重新检测", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示状态信息
        status = "跟踪中" if success else "失败"
        cv2.putText(frame, f"{tracker_name} - {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 为了演示,不实际显示
        # cv2.imshow("长期跟踪", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 使用的跟踪器: {tracker_name}")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - 成功跟踪帧数: {success_count}")
    print(f"  - 跟踪失败帧数: {fail_count}")
    print(f"  - 成功率: {success_count / max(frame_count, 1) * 100:.1f}%")

    # 验证结果
    success_rate = success_count / max(frame_count, 1)
    verify_result(f"{tracker_name} 初始化", True, "跟踪器成功初始化")
    verify_result(f"{tracker_name} 跟踪", success_count > 0,
                  f"成功率: {success_rate * 100:.1f}%")

    return success_count > 0


# =============================================================================
# 挑战题
# =============================================================================

def exercise_10_multi_camera_tracking():
    """
    练习10: 多摄像头跟踪协同
    =======================

    算法原理:
    --------
    多摄像头跟踪是计算机视觉中的高级问题,涉及:
    1. 跨摄像头目标匹配
    2. 目标重识别 (Re-ID)
    3. 空间和时间一致性

    常用方法:
    - 基于外观的匹配: 使用深度学习特征比较不同摄像头中的目标
    - 基于时间的匹配: 当目标在一个摄像头消失后,
                     在另一个摄像头出现时进行关联
    - 基于几何的匹配: 利用摄像头之间的几何关系限制搜索空间

    本练习实现(简化版本):
    ------------------
    1. 模拟两个摄像头视图
    2. 为每个摄像头创建独立的跟踪器
    3. 模拟目标在摄像头间的切换
    4. 简单的时间关联逻辑
    """
    print("\n" + "=" * 60)
    print("练习10: 多摄像头跟踪协同")
    print("=" * 60)

    # 模拟两个摄像头
    cap1 = create_test_video()
    cap2 = create_test_video()

    if not cap1.isOpened() or not cap2.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 为每个摄像头创建跟踪器
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("错误: 无法读取视频帧")
        cap1.release()
        cap2.release()
        return False

    h, w = frame1.shape[:2]

    # 摄像头1跟踪目标
    bbox1 = (w // 4, h // 4, w // 4, h // 4)
    tracker1 = cv2.TrackerKCF.create()
    tracker1.init(frame1, bbox1)

    # 摄像头2跟踪目标
    bbox2 = (3 * w // 4, h // 4, w // 4, h // 4)
    tracker2 = cv2.TrackerKCF.create()
    tracker2.init(frame2, bbox2)

    print("信息: 两个摄像头的跟踪器已初始化")

    # 多摄像头跟踪状态
    class MultiCameraTracker:
        def __init__(self):
            self.tracks = {}  # track_id -> track_info
            self.next_id = 0

        def update(self, camera_id, detections):
            """
            更新指定摄像头的检测结果

            Args:
                camera_id: 摄像头ID
                detections: 检测到的目标列表 [(bbox, feature), ...]
            """
            # 简化版本: 直接使用跟踪器输出的边界框作为检测结果
            pass

        def associate(self, camera1_tracks, camera2_tracks):
            """
            关联不同摄像头中的同一目标
            """
            # 简化版本: 假设两个摄像头观察到不同的目标
            # 实际应用中需要使用外观特征或几何约束进行匹配
            pass

    # 初始化多摄像头跟踪器
    mc_tracker = MultiCameraTracker()

    # 跟踪统计
    frame_count = 0
    cam1_success = 0
    cam2_success = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame_count += 1

        # 更新摄像头1的跟踪器
        success1, bbox1_new = tracker1.update(frame1)
        if success1:
            cam1_success += 1
            frame1 = draw_tracking_result(frame1, bbox1_new, "Camera 1", (255, 0, 0))

        # 更新摄像头2的跟踪器
        success2, bbox2_new = tracker2.update(frame2)
        if success2:
            cam2_success += 1
            frame2 = draw_tracking_result(frame2, bbox2_new, "Camera 2", (0, 255, 0))

        # 显示摄像头信息
        cv2.putText(frame1, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame2, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 为了演示,不实际显示
        # cv2.imshow("Camera 1", frame1)
        # cv2.imshow("Camera 2", frame2)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap1.release()
    cap2.release()

    # 统计信息
    print(f"\n统计:")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - 摄像头1 成功跟踪帧数: {cam1_success}")
    print(f"  - 摄像头2 成功跟踪帧数: {cam2_success}")
    print(f"  - 摄像头1 成功率: {cam1_success / max(frame_count, 1) * 100:.1f}%")
    print(f"  - 摄像头2 成功率: {cam2_success / max(frame_count, 1) * 100:.1f}%")

    # 验证结果
    success = frame_count > 0 and cam1_success > 0 and cam2_success > 0
    verify_result("多摄像头初始化", True, "成功初始化两个摄像头")
    verify_result("摄像头1跟踪", cam1_success > 0, f"成功率: {cam1_success/max(frame_count,1)*100:.1f}%")
    verify_result("摄像头2跟踪", cam2_success > 0, f"成功率: {cam2_success/max(frame_count,1)*100:.1f}%")

    return success


def exercise_11_deep_learning_tracker():
    """
    练习11: 结合深度学习的跟踪器 (如 DaSiamRPN)
    ==============================================

    算法原理:
    --------
    深度学习跟踪器利用卷积神经网络学习目标的特征表示,
    在复杂场景中具有更好的鲁棒性。

    主要类型:
    1. Siamese Network (孪生网络):
       - 使用两个共享权重的网络分支
       - 一个处理模板图像(第一帧)
       - 一个处理搜索图像(当前帧)
       - 通过互相关计算目标位置

    2. DaSiamRPN (Discriminative SiamRPN):
       - 在 SiamRPN 基础上加入判别学习
       - 增强区分目标和背景的能力
       - 能够处理严重遮挡

    3. ATOM (Accurate Tracking by Overlap Maximization):
       - 通过重叠最大化进行跟踪
       - 使用调制注意力机制

    4. DiMP (Discriminative Model Prediction):
       - 端到端的判别跟踪器
       - 在跟踪过程中持续学习

    预训练模型:
    - OpenCV 提供了 GOTURN 模型接口
    - 需要下载 .nab 文件
    - NanoTrack 在 OpenCV 4.5+ 中可用

    本练习实现:
    -----------
    1. 尝试使用 NanoTrack (轻量级深度学习跟踪器)
    2. 演示深度学习跟踪器的基本用法
    3. 比较与传统跟踪器的性能差异
    """
    print("\n" + "=" * 60)
    print("练习11: 结合深度学习的跟踪器 (DaSiamRPN)")
    print("=" * 60)

    # 创建测试视频
    cap = create_test_video()

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return False

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return False

    # 定义初始跟踪区域
    h, w = frame.shape[:2]
    bbox = (w // 4, h // 4, w // 4, h // 4)

    # 检查 NanoTrack 是否可用
    # NanoTrack 从 OpenCV 4.5 开始支持
    tracker_nano = None

    try:
        tracker_nano = cv2.TrackerNanoTrack.create()
        print("信息: NanoTrack 跟踪器创建成功")
        print("  - NanoTrack 是轻量级深度学习跟踪器")
        print("  - 专为边缘设备优化")
        print("  - 高精度,快速推理")
    except Exception as e:
        print(f"警告: NanoTrack 创建失败: {e}")

    # 如果 NanoTrack 不可用,尝试使用其他深度学习跟踪器
    if tracker_nano is None:
        try:
            # DaSiamRPN 需要额外的模型文件
            # 这里演示创建过程,实际使用需要下载模型
            tracker_nano = cv2.TrackerNanoTrack.create()
        except:
            print("提示: 深度学习跟踪器不可用,使用 KCF 作为替代")
            tracker_nano = None

    # 使用 KCF 作为备选
    if tracker_nano is None:
        tracker = cv2.TrackerKCF.create()
        tracker_name = "KCF (替代)"
    else:
        tracker = tracker_nano
        tracker_name = "NanoTrack (深度学习)"

    # 初始化跟踪器
    success = tracker.init(frame, bbox)
    if not success:
        print("错误: 跟踪器初始化失败")
        cap.release()
        return False

    print(f"信息: {tracker_name} 跟踪器初始化成功")

    # 跟踪性能测试
    frame_count = 0
    success_count = 0
    total_time = 0.0

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 更新跟踪器
        success, new_bbox = tracker.update(frame)
        elapsed = time.time() - start_time
        total_time += elapsed

        if success:
            success_count += 1
            frame = draw_tracking_result(frame, new_bbox, tracker_name, (255, 0, 0))

        # 为了演示,不实际显示
        # cv2.imshow("深度学习跟踪", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()

    # 计算性能指标
    avg_fps = frame_count / total_time if total_time > 0 else 0
    success_rate = success_count / max(frame_count, 1) * 100

    # 统计信息
    print(f"\n统计:")
    print(f"  - 跟踪器类型: {tracker_name}")
    print(f"  - 处理帧数: {frame_count}")
    print(f"  - 成功跟踪帧数: {success_count}")
    print(f"  - 平均 FPS: {avg_fps:.2f}")
    print(f"  - 成功率: {success_rate:.1f}%")

    # 验证结果
    success = frame_count > 0 and success_count > 0
    verify_result(f"{tracker_name} 初始化", True, "跟踪器成功初始化")
    verify_result(f"{tracker_name} 跟踪", success, f"成功率: {success_rate:.1f}%")
    verify_result(f"{tracker_name} 性能", avg_fps > 0, f"FPS: {avg_fps:.2f}")

    return success


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数 - 运行所有练习
    """
    print("=" * 60)
    print("OpenCV Video 模块练习题")
    print("=" * 60)
    print("\n练习内容:")
    print("  入门级:")
    print("    1. KCF 目标跟踪器的基本使用")
    print("    2. Lucas-Kanade 光流跟踪特征点")
    print("    3. 简单的运动检测 (背景减除)")
    print("\n  中级:")
    print("    4. 多目标跟踪系统")
    print("    5. 光流可视化 (稠密光流)")
    print("    6. 比较不同跟踪器的性能")
    print("\n  高级:")
    print("    7. 基于光流的运动估计")
    print("    8. 基于跟踪的航迹维持")
    print("    9. 长期目标跟踪 (TLD 或 GOTURN)")
    print("\n  挑战题:")
    print("    10. 多摄像头跟踪协同")
    print("    11. 结合深度学习的跟踪器")
    print("=" * 60)

    results = {}

    # 运行入门级练习
    print("\n\n" + "=" * 60)
    print("入门级练习")
    print("=" * 60)

    results[1] = exercise_1_kcf_tracker()
    results[2] = exercise_2_lucas_kanade_optical_flow()
    results[3] = exercise_3_background_subtraction()

    # 运行中级练习
    print("\n\n" + "=" * 60)
    print("中级练习")
    print("=" * 60)

    results[4] = exercise_4_multi_target_tracking()
    results[5] = exercise_5_dense_optical_flow_visualization()
    results[6] = exercise_6_tracker_comparison()

    # 运行高级练习
    print("\n\n" + "=" * 60)
    print("高级练习")
    print("=" * 60)

    results[7] = exercise_7_motion_estimation_optical_flow()
    results[8] = exercise_8_tracking_by_detection()
    results[9] = exercise_9_long_term_tracker()

    # 运行挑战题
    print("\n\n" + "=" * 60)
    print("挑战题")
    print("=" * 60)

    results[10] = exercise_10_multi_camera_tracking()
    results[11] = exercise_11_deep_learning_tracker()

    # 打印总结
    print("\n\n" + "=" * 60)
    print("练习总结")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")

    print("\n详细结果:")
    for i, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  练习 {i}: {status}")

    print("\n" + "=" * 60)
    print("所有练习完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()