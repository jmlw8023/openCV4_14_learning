#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV Videoio 模块练习题
=========================

本文件包含 OpenCV videoio 模块的所有练习题解答,
涵盖 VideoCapture、VideoWriter、视频帧处理、摄像头捕获和视频属性读取等功能。

练习列表:
--------
入门级:
    1. 读取视频文件并逐帧显示
    2. 打开摄像头并实时显示
    3. 从摄像头捕获视频并保存到文件

中级:
    4. 实现视频播放器的暂停/继续功能
    5. 实现视频的时间跳转功能 (百分比/秒)
    6. 实现批量视频格式转换 (MP4 → AVI)

高级:
    7. 实现多摄像头同步捕获
    8. 实现视频处理流水线 (如: 边缘检测输出)
    9. 实现视频水印添加功能

挑战题:
    10. 实现自定义视频编码器后端
    11. 实现 RTSP 流媒体服务器推送

作者: OpenCV Learning Team
版本: 1.0
OpenCV 版本要求: 4.14.0-pre
"""

import cv2
import numpy as np
import os
import sys
import time
from typing import Optional, Tuple, List


# =============================================================================
# 辅助函数
# =============================================================================

def print_section(title: str) -> None:
    """打印分节标题,用于分隔不同的练习输出"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_video_info(cap: cv2.VideoCapture, source_name: str = "视频") -> None:
    """
    打印视频/摄像头的基本信息

    参数:
        cap: VideoCapture 对象
        source_name: 源名称,用于显示
    """
    # CAP_PROP_FRAME_WIDTH: 帧宽度(像素)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # CAP_PROP_FRAME_HEIGHT: 帧高度(像素)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # CAP_PROP_FPS: 帧率(每秒帧数)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # CAP_PROP_FRAME_COUNT: 总帧数(对实时流无效,返回-1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # CAP_PROP_POS_MSEC: 当前播放位置(毫秒)
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    # CAP_PROP_POS_FRAMES: 当前帧编号
    pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # CAP_PROP_CURRENT_BACKEND: 当前使用的后端
    backend = int(cap.get(cv2.CAP_PROP_CURRENT_BACKEND))

    print(f"\n[{source_name} 信息]")
    print(f"  分辨率: {width} x {height} 像素")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {frame_count}")
    print(f"  当前位置: {pos_msec:.2f} ms (第 {pos_frames} 帧)")
    print(f"  后端ID: {backend}")


def create_test_video(output_path: str, fps: float = 30.0, duration: float = 2.0,
                      width: int = 640, height: int = 480) -> bool:
    """
    创建测试用的小视频文件,用于练习需要输入视频的场景

    算法原理:
        1. 使用 VideoWriter 创建指定格式的视频文件
        2. 生成随时间变化的彩色帧(使用正弦波产生颜色变化效果)
        3. 每帧包含时间戳文字和进度条,便于验证视频播放

    参数:
        output_path: 输出视频文件路径
        fps: 帧率
        duration: 时长(秒)
        width: 宽度
        height: 高度

    返回:
        是否创建成功
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 Part 2 编码
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"错误: 无法创建视频写入器 {output_path}")
        return False

    total_frames = int(fps * duration)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(total_frames):
        # 创建基于时间的彩色图像(使用正弦波产生平滑颜色过渡)
        # 三个通道使用不同频率的正弦波,产生颜色循环效果
        t = i / fps  # 当前时间(秒)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 计算颜色分量: R 和 B 用余弦,G 用正弦,产生色彩变化
        frame[:, :, 2] = np.abs(np.sin(t * 2 + np.arange(width) * 0.01) * 255).astype(np.uint8)
        frame[:, :, 1] = np.abs(np.cos(t * 3 + np.arange(width) * 0.01) * 255).astype(np.uint8)
        frame[:, :, 0] = np.abs(np.sin(t * 1.5 + np.arange(width) * 0.01) * 255).astype(np.uint8)

        # 添加时间戳文字
        timestamp = f"Time: {t:.2f}s"
        cv2.putText(frame, timestamp, (20, 50), font, 1.2, (255, 255, 255), 2)

        # 添加帧号
        frame_num = f"Frame: {i+1}/{total_frames}"
        cv2.putText(frame, frame_num, (20, 100), font, 1.0, (255, 255, 255), 2)

        # 添加进度条
        progress = i / total_frames
        bar_width = int(width * 0.8 * progress)
        cv2.rectangle(frame, (int(width * 0.1), height - 50),
                      (int(width * 0.1) + bar_width, height - 30),
                      (0, 255, 0), -1)

        writer.write(frame)

    writer.release()
    print(f"测试视频已创建: {output_path} ({total_frames} 帧, {duration}s)")
    return True


# =============================================================================
# 练习 1: 读取视频文件并逐帧显示
# =============================================================================

def exercise_1_read_video_file(video_path: Optional[str] = None) -> None:
    """
    练习 1: 读取视频文件并逐帧显示

    算法原理:
        1. 使用 VideoCapture 打开视频文件
        2. 通过循环调用 read() 方法逐帧读取
        3. 每读取一帧进行检查:如果为空(Frame empty),表示已到达视频末尾
        4. 使用 imshow 显示每一帧
        5. 使用 waitKey 控制播放速度(毫秒为单位)

    VideoCapture 核心机制:
        - read() 方法返回 (success, frame) 元组
        - 当返回 frame 为空时,表示视频播放完毕或读取失败
        - 循环读取时要在每次迭代中检查帧是否为空
    """
    print_section("练习 1: 读取视频文件并逐帧显示")

    # 如果没有提供视频文件,创建测试视频
    if video_path is None or not os.path.exists(video_path):
        video_path = "test_video_ex1.mp4"
        if not create_test_video(video_path, fps=10, duration=3):
            return

    # 创建 VideoCapture 对象并打开视频文件
    # 构造函数可以接受文件路径,OpenCV 会自动检测后端(FFmpeg/MSMF等)
    cap = cv2.VideoCapture(video_path)

    # 重要: 检查视频是否成功打开
    # isOpened() 方法返回布尔值,表示后端是否成功初始化
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return

    # 打印视频基本信息
    print_video_info(cap, os.path.basename(video_path))

    # 获取总帧数计算视频时长
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    print(f"  预计时长: {duration:.2f} 秒")

    frame_num = 0
    start_time = time.time()

    print("\n开始逐帧读取...")
    print("按 'q' 键退出播放")

    # 创建窗口用于显示
    window_name = "练习1 - 视频播放"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 主循环: 逐帧读取并显示
    while True:
        # read() 方法从视频源读取下一帧
        # 返回值 success 表示是否成功读取(可能返回False但仍有frame数据)
        success, frame = cap.read()

        # 检查帧是否为空
        # empty frame 表示: 1)视频结束  2)读取错误  3)损坏的帧
        if frame is None or frame.size == 0:
            print(f"\n视频播放完毕! (共读取 {frame_num} 帧)")
            break

        frame_num += 1

        # 在帧上叠加播放信息
        display_frame = frame.copy()
        info_text = f"Frame {frame_num}/{frame_count}"
        cv2.putText(display_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示帧
        cv2.imshow(window_name, display_frame)

        # waitKey 参数:
        #   - 返回值: 按键的 ASCII 码(如果超时不带按键,返回 -1)
        #   - 控制播放速度: waitKey(100) 表示每帧间隔 100ms,约 10fps
        #   - 如果设为 1ms,会尽可能快地播放
        key = cv2.waitKey(int(1000/fps))  # 按帧率控制播放速度

        if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
            print(f"用户退出 (读取了 {frame_num} 帧)")
            break

    # 计算实际播放时间
    elapsed = time.time() - start_time
    print(f"实际播放时间: {elapsed:.2f} 秒")
    print(f"平均播放帧率: {frame_num/elapsed:.2f} FPS")

    # 清理资源
    cap.release()
    cv2.destroyWindow(window_name)


# =============================================================================
# 练习 2: 打开摄像头并实时显示
# =============================================================================

def exercise_2_camera_realtime(camera_index: int = 0) -> None:
    """
    练习 2: 打开摄像头并实时显示

    算法原理:
        1. 使用 VideoCapture(device_index) 打开摄像头
        2. device_index: 0 表示第一个摄像头,1 表示第二个,以此类推
        3. 通过循环实时读取帧并显示
        4. 使用 waitKey(1) 实现最小延迟的实时显示
        5. 检测摄像头是否可用,如果不可用则报错

    摄像头后端选择(按平台):
        - Windows: DirectShow (CAP_DSHOW) 或 MSMF (CAP_MSMF)
        - Linux: V4L2 (CAP_V4L2)
        - macOS: AVFoundation (CAP_AVFOUNDATION)

    性能优化:
        - waitKey(1) 提供最小延迟,但CPU占用高
        - waitKey(30) 降低CPU占用,但延迟增加
        - 实时显示场景通常使用 waitKey(1) 或 waitKey(5)
    """
    print_section("练习 2: 打开摄像头并实时显示")

    # 打印可用后端信息
    backend = cv2.CAP_ANY  # 自动选择后端
    print(f"尝试打开摄像头索引: {camera_index}")
    print(f"使用后端: ", end="")

    # 根据平台选择后端
    if sys.platform == 'win32':
        print("DirectShow/MSMF (Windows)")
    elif sys.platform == 'linux':
        print("V4L2 (Linux)")
    else:
        print("AVFoundation (macOS)")

    # 打开摄像头
    # 方式1: 直接在构造函数中指定设备索引
    cap = cv2.VideoCapture(camera_index, backend)

    # 方式2: 使用 open() 方法
    # cap = cv2.VideoCapture()
    # cap.open(camera_index, backend)

    # 检查是否成功打开
    if not cap.isOpened():
        print(f"\n错误: 无法打开摄像头 {camera_index}")
        print("可能的原因:")
        print("  - 摄像头被其他程序占用")
        print("  - 摄像头索引不正确(尝试 0, 1, 2...)")
        print("  - 缺少摄像头驱动")
        return

    # 获取摄像头支持的分辨率
    # 注意: 并非所有分辨率都被支持,设置不支持的分辨率可能会失败
    print("\n尝试设置摄像头分辨率...")

    # 常见的摄像头设置
    # CAP_PROP_FRAME_WIDTH/CAP_PROP_FRAME_HEIGHT: 设置帧分辨率
    target_width = 640
    target_height = 480

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    # 读取实际设置的分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"目标分辨率: {target_width}x{target_height}")
    print(f"实际分辨率: {actual_width}x{actual_height}")
    print(f"实际帧率: {actual_fps:.2f} FPS")

    # 创建显示窗口
    window_name = "练习2 - 摄像头实时显示"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    print("\n开始实时显示...")
    print("按 'q' 键退出")

    # 实时显示循环
    while True:
        # 读取一帧
        success, frame = cap.read()

        if not success or frame is None:
            print("警告: 读取摄像头帧失败")
            break

        frame_count += 1

        # 计算帧率
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0

        # 在帧上叠加信息
        display_frame = frame.copy()
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示帧
        cv2.imshow(window_name, display_frame)

        # waitKey(1) 实现最小延迟的实时显示
        # 返回按下的键值,-1 表示超时没有按键
        key = cv2.waitKey(1)

        if key == ord('q') or key == 27:
            print(f"用户退出 (共显示 {frame_count} 帧)")
            break

    # 清理
    elapsed = time.time() - start_time
    print(f"平均帧率: {frame_count/elapsed:.2f} FPS")

    cap.release()
    cv2.destroyWindow(window_name)


# =============================================================================
# 练习 3: 从摄像头捕获视频并保存到文件
# =============================================================================

def exercise_3_camera_capture_and_save(output_path: str = "camera_capture.mp4",
                                       camera_index: int = 0,
                                       duration: float = 5.0) -> None:
    """
    练习 3: 从摄像头捕获视频并保存到文件

    算法原理:
        1. 打开摄像头 VideoCapture
        2. 创建 VideoWriter 用于写入视频
        3. 配置 FourCC 编码格式、帧率、分辨率
        4. 循环读取摄像头帧并写入 VideoWriter
        5. 完成后释放资源

    VideoWriter 配置参数:
        - fourcc: 4字符编码标识符,确定视频编解码器
          常见格式: 'MJPG' (Motion JPEG), 'H264' (H.264), 'XVID' (Xvid)
        - fps: 目标帧率,应与源帧率匹配
        - frameSize: 帧尺寸,必须与 VideoCapture 设置一致
        - isColor: 是否为彩色视频

    关键注意事项:
        - VideoWriter 必须指定正确的 fourcc,否则保存失败
        - frameSize 必须与实际帧尺寸完全匹配
        - 写入前检查 writer.isOpened()
    """
    print_section("练习 3: 从摄像头捕获视频并保存到文件")

    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_index}")
        return

    # 获取摄像头原始属性
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"摄像头分辨率: {cap_width}x{cap_height}")
    print(f"摄像头帧率: {cap_fps:.2f} FPS")

    # 配置 VideoWriter
    # FourCC (4字符代码) 选择:
    #   'mp4v' - MPEG-4 Part 2 (通用,兼容性最好)
    #   'H264' - H.264 (高压缩率,但需要编码器支持)
    #   'MJPG' - Motion JPEG (无损,但文件大)
    #   'XVID' - Xvid (开源 MPEG-4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter
    writer = cv2.VideoWriter(output_path, fourcc, cap_fps, (cap_width, cap_height))

    # 验证 VideoWriter 是否成功创建
    if not writer.isOpened():
        print(f"错误: 无法创建视频写入器: {output_path}")
        print("可能的原因:")
        print("  - 指定的编码格式不被支持")
        print("  - 磁盘空间不足")
        print("  - 文件路径无效")
        cap.release()
        return

    print(f"\n开始录制视频到: {output_path}")
    print(f"录制时长: {duration} 秒 (或按 'q' 提前结束)")

    window_name = "练习3 - 摄像头录制"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    recording = True

    # 录制循环
    while recording:
        # 读取摄像头帧
        success, frame = cap.read()
        if not success or frame is None:
            print("警告: 摄像头帧读取失败")
            break

        frame_count += 1

        # 写入视频文件
        # write() 方法接受一个 Mat/UMat 帧
        # 也可以使用 operator<<: writer << frame
        writer.write(frame)

        # 计算已录制时长
        elapsed = time.time() - start_time
        remaining = duration - elapsed

        # 在预览帧上叠加录制信息
        display_frame = frame.copy()
        rec_text = f"REC - Time: {elapsed:.1f}s / {duration:.1f}s"
        cv2.putText(display_frame, rec_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示录制状态红点
        cv2.circle(display_frame, (cap_width - 30, 30), 10, (0, 0, 255), -1)

        cv2.imshow(window_name, display_frame)

        # 检查是否达到指定时长或用户按键退出
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or elapsed >= duration:
            recording = False
            print(f"录制结束 (录制了 {frame_count} 帧)")

    # 计算实际录制帧率
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"实际录制帧率: {actual_fps:.2f} FPS")

    # 释放资源
    cap.release()
    writer.release()
    cv2.destroyWindow(window_name)

    # 验证输出文件
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n视频已保存: {output_path}")
        print(f"文件大小: {file_size / 1024:.2f} KB")
        print(f"录制帧数: {frame_count}")
    else:
        print(f"\n错误: 视频文件未生成: {output_path}")


# =============================================================================
# 练习 4: 实现视频播放器的暂停/继续功能
# =============================================================================

def exercise_4_pause_resume(video_path: Optional[str] = None) -> None:
    """
    练习 4: 实现视频播放器的暂停/继续功能

    算法原理:
        1. 使用状态变量控制播放/暂停状态
        2. 暂停时保持当前帧画面,停止调用 read()
        3. 继续时恢复读取下一帧
        4. 使用 waitKey 检测空格键切换状态
        5. 可选: 支持逐帧前进(按 'f' 键)

    状态机设计:
        - PLAYING: 正常播放状态,持续读取帧
        - PAUSED: 暂停状态,显示当前帧,不读取新帧
        - 切换条件: 按空格键(ASCII 32)

    逐帧播放实现:
        - 按 'f' 键手动调用一次 read()
        - 暂停状态下支持逐帧审查
    """
    print_section("练习 4: 实现视频播放器的暂停/继续功能")

    # 创建测试视频
    if video_path is None or not os.path.exists(video_path):
        video_path = "test_video_ex4.mp4"
        if not create_test_video(video_path, fps=10, duration=5):
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {video_path}")
        return

    print_video_info(cap, os.path.basename(video_path))

    # 播放状态常量
    STATE_PLAYING = "playing"
    STATE_PAUSED = "paused"

    state = STATE_PLAYING  # 初始状态为播放
    current_frame = None
    frame_count = 0
    paused_frame_count = 0

    window_name = "练习4 - 暂停/继续播放器"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n播放控制:")
    print("  空格键: 暂停/继续")
    print("  'f' 键: 逐帧前进(暂停时有效)")
    print("  'q' 或 ESC: 退出")

    # 主播放循环
    while True:
        if state == STATE_PLAYING:
            # 播放状态: 读取新帧
            success, frame = cap.read()

            if frame is None or frame.size == 0:
                print(f"\n视频播放完毕! (共播放 {frame_count} 帧)")
                break

            current_frame = frame
            frame_count += 1

        elif state == STATE_PAUSED:
            # 暂停状态: 逐帧计数(用于统计暂停期间观看的帧数)
            paused_frame_count += 1

        # 在帧上叠加状态信息
        if current_frame is not None:
            display_frame = current_frame.copy()

            # 显示播放状态
            state_text = "▶ 播放中" if state == STATE_PLAYING else "⏸ 暂停"
            color = (0, 255, 0) if state == STATE_PLAYING else (0, 200, 255)
            cv2.putText(display_frame, state_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 显示帧号
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, display_frame)

        # 根据状态调整等待时间
        # 播放状态: 每帧等待 1000/fps 毫秒
        # 暂停状态: 等待 30ms,响应按键但保持画面
        wait_time = 1000 // 10 if state == STATE_PLAYING else 30
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == 27:  # 退出
            print(f"\n用户退出 (播放了 {frame_count} 帧, 暂停浏览 {paused_frame_count} 帧)")
            break
        elif key == 32:  # 空格键 - 切换暂停/继续
            state = STATE_PAUSED if state == STATE_PLAYING else STATE_PLAYING
            status = "暂停" if state == STATE_PAUSED else "继续播放"
            print(f"状态切换: {status}")
        elif key == ord('f') and state == STATE_PAUSED:
            # 逐帧前进: 在暂停状态下手动读取一帧
            print("逐帧前进")
            success, frame = cap.read()
            if frame is not None and frame.size > 0:
                current_frame = frame
                frame_count += 1

    cap.release()
    cv2.destroyWindow(window_name)


# =============================================================================
# 练习 5: 实现视频的时间跳转功能
# =============================================================================

def exercise_5_seek_video(video_path: Optional[str] = None) -> None:
    """
    练习 5: 实现视频的时间跳转功能

    算法原理:
        1. 使用 CAP_PROP_POS_MSEC 设置播放位置(毫秒为单位)
        2. 使用 CAP_PROP_POS_FRAMES 设置播放位置(帧号为单位)
        3. 使用 CAP_PROP_POS_AVI_RATIO 设置播放位置(总长的比例 [0,1])
        4. 设置后调用 read() 读取该位置对应的帧
        5. 注意: 不是所有后端都支持精确跳转

    跳转方式:
        - 按时间(毫秒): set(CAP_PROP_POS_MSEC, 5000) 跳到第5秒
        - 按帧号: set(CAP_PROP_POS_FRAMES, 100) 跳到第100帧
        - 按比例: set(CAP_PROP_POS_AVI_RATIO, 0.5) 跳到50%位置

    后端限制:
        - 本地文件: 通常支持精确跳转
        - RTSP 流: 可能只支持关键帧跳转
        - 摄像头: 不支持跳转
    """
    print_section("练习 5: 实现视频的时间跳转功能")

    # 创建测试视频
    if video_path is None or not os.path.exists(video_path):
        video_path = "test_video_ex5.mp4"
        if not create_test_video(video_path, fps=10, duration=10):
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {video_path}")
        return

    # 获取视频总信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = frame_count / fps if fps > 0 else 0

    print(f"视频总帧数: {frame_count}")
    print(f"视频帧率: {fps:.2f} FPS")
    print(f"视频时长: {total_duration:.2f} 秒")

    window_name = "练习5 - 时间跳转"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_pos = 0  # 当前播放位置(秒)

    print("\n跳转控制:")
    print("  输入数字 + 's': 跳转到指定秒数 (如: 3s)")
    print("  输入数字 + 'f': 跳转到指定帧号 (如: 50f)")
    print("  输入数字 + '%': 跳转到指定百分比 (如: 50%)")
    print("  'q' 或 ESC: 退出")

    # 时间跳转演示
    test_seeks = [
        (2.0, "按秒跳转: 2秒"),
        (5.0, "按秒跳转: 5秒"),
        (0.5, "按比例跳转: 50%"),
    ]

    print("\n--- 自动跳转演示 ---")
    for pos, description in test_seeks:
        print(f"\n跳转到: {description}")

        if isinstance(pos, float) and pos <= 1.0:
            # 按比例跳转 (0.0 - 1.0)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, pos)
            print(f"  设置 POS_AVI_RATIO = {pos}")
        else:
            # 按时间跳转 (秒)
            msec = pos * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, msec)
            print(f"  设置 POS_MSEC = {msec:.0f} ms")

        # 读取跳转后的帧
        success, frame = cap.read()
        if not success or frame is None:
            print(f"  跳转后无法读取帧")
            continue

        # 获取实际跳转位置
        actual_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        actual_ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)

        print(f"  实际位置: {actual_msec:.0f} ms (第 {actual_frame} 帧, {actual_ratio:.1%})")

        # 显示帧
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Jumped to: {description}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Time: {actual_msec/1000:.2f}s",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Ratio: {actual_ratio:.1%}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)
        cv2.waitKey(1500)  # 显示1.5秒

        # 按 'q' 可提前退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 手动跳转交互
    print("\n--- 手动跳转 ---")
    print("输入跳转命令(输入 'q' 退出):")

    while True:
        user_input = input("\n> ")
        if user_input.lower() == 'q':
            break

        try:
            if user_input.endswith('s'):
                # 按秒跳转
                seconds = float(user_input[:-1])
                msec = seconds * 1000
                cap.set(cv2.CAP_PROP_POS_MSEC, msec)
                print(f"  跳转至: {seconds:.2f} 秒")

            elif user_input.endswith('f'):
                # 按帧号跳转
                frame_num = int(user_input[:-1])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                print(f"  跳转至: 第 {frame_num} 帧")

            elif user_input.endswith('%'):
                # 按百分比跳转
                ratio = float(user_input[:-1]) / 100.0
                cap.set(cv2.CAP_PROP_POS_AVI_RATIO, ratio)
                print(f"  跳转至: {ratio*100:.1f}%")

            else:
                print("  无效输入格式!")
                continue

            # 读取跳转后的帧并显示
            success, frame = cap.read()
            if success and frame is not None and frame.size > 0:
                actual_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                display_frame = frame.copy()
                cv2.putText(display_frame, f"Time: {actual_msec/1000:.2f}s  Frame: {actual_frame}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(100)
            else:
                print("  跳转后无法读取帧(可能超出视频范围)")

        except ValueError:
            print(f"  无法解析输入: {user_input}")

    cap.release()
    cv2.destroyWindow(window_name)


# =============================================================================
# 练习 6: 实现批量视频格式转换
# =============================================================================

def exercise_6_batch_convert(input_dir: Optional[str] = None,
                               output_dir: Optional[str] = None,
                               input_ext: str = "mp4",
                               output_ext: str = "avi") -> None:
    """
    练习 6: 实现批量视频格式转换

    算法原理:
        1. 遍历输入目录查找所有指定扩展名的视频文件
        2. 对每个视频:
           a. 打开源视频获取属性(fourcc, fps, 分辨率)
           b. 创建目标格式的 VideoWriter
           c. 逐帧读取并转换编码写入新文件
        3. 保持相同的 fps 和分辨率,只改变编码格式

    格式转换流程:
        输入视频 → VideoCapture → 帧数据 → VideoWriter → 输出视频

    编码格式选择建议:
        - MP4 → AVI: 使用 'MJPG' 或 'XVID' 保持兼容性
        - AVI → MP4: 使用 'mp4v' 或 'H264'
        - 跨平台: 使用 'MJPG' (最通用)

    参数说明:
        input_dir: 输入视频目录(默认为当前目录)
        output_dir: 输出视频目录(默认为 input_dir/converted)
        input_ext: 输入文件扩展名
        output_ext: 输出文件扩展名
    """
    print_section("练习 6: 批量视频格式转换")

    # 设置目录
    if input_dir is None:
        input_dir = "."
    if output_dir is None:
        output_dir = os.path.join(input_dir, "converted")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找输入目录中的视频文件
    video_files = []
    if os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if f.lower().endswith(f'.{input_ext.lower()}'):
                video_files.append(os.path.join(input_dir, f))

    # 如果没有找到视频,创建测试视频
    if not video_files:
        test_video = os.path.join(input_dir, f"test.{input_ext}")
        print(f"目录中没有 .{input_ext} 文件,创建测试视频...")
        if not create_test_video(test_video, fps=10, duration=3):
            return
        video_files.append(test_video)

    print(f"找到 {len(video_files)} 个视频文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"转换: .{input_ext} → .{output_ext}")

    # FourCC 映射表
    fourcc_map = {
        'mp4': cv2.VideoWriter_fourcc(*'mp4v'),
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        'mkv': cv2.VideoWriter_fourcc(*'X264'),
        'mov': cv2.VideoWriter_fourcc(*'mp4v'),
    }

    # 获取输出格式的 fourcc
    output_fourcc = fourcc_map.get(output_ext.lower(), cv2.VideoWriter_fourcc(*'mp4v'))
    output_fourcc_str = ''.join([chr((output_fourcc >> 8* i) & 0xFF) for i in range(4)])
    print(f"使用编码: {output_fourcc_str}")

    # 转换统计
    success_count = 0
    fail_count = 0

    for input_path in video_files:
        filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name_without_ext}.{output_ext}")

        print(f"\n转换中: {filename}")

        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"  错误: 无法打开 {filename}")
            fail_count += 1
            continue

        # 获取输入视频属性
        in_fps = cap.get(cv2.CAP_PROP_FPS)
        in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        in_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        in_fourcc_str = ''.join([chr((in_fourcc >> 8* i) & 0xFF) for i in range(4)])

        print(f"  输入: {in_width}x{in_height}, {in_fps:.2f} FPS, {in_fourcc_str}")

        # 创建输出视频写入器
        writer = cv2.VideoWriter(output_path, output_fourcc, in_fps, (in_width, in_height))
        if not writer.isOpened():
            print(f"  错误: 无法创建输出文件 {output_path}")
            cap.release()
            fail_count += 1
            continue

        # 逐帧转换
        frame_count = 0
        start_time = time.time()

        while True:
            success, frame = cap.read()
            if frame is None or frame.size == 0:
                break

            writer.write(frame)
            frame_count += 1

        # 释放资源
        cap.release()
        writer.release()

        # 计算转换时间
        elapsed = time.time() - start_time

        # 验证输出文件
        if os.path.exists(output_path):
            out_size = os.path.getsize(output_path)
            print(f"  输出: {output_path}")
            print(f"  成功: {frame_count} 帧, {elapsed:.2f}秒, {out_size/1024:.1f}KB")
            success_count += 1
        else:
            print(f"  失败: 输出文件未生成")
            fail_count += 1

    # 打印统计
    print(f"\n{'='*40}")
    print(f"转换完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  输出目录: {output_dir}")


# =============================================================================
# 练习 7: 实现多摄像头同步捕获
# =============================================================================

def exercise_7_multi_camera_sync(max_cameras: int = 4) -> None:
    """
    练习 7: 实现多摄像头同步捕获

    算法原理:
        1. 遍历尝试打开多个摄像头(索引 0 到 max_cameras-1)
        2. 记录成功打开的摄像头列表
        3. 为每个摄像头创建独立的读取线程或同步读取
        4. 在同一窗口或多个窗口显示所有摄像头画面
        5. 处理摄像头断开等异常情况

    同步策略:
        - 简单策略: 顺序读取所有摄像头(帧率受最慢摄像头限制)
        - 高级策略: 使用独立线程为每个摄像头维护帧缓冲区
        - 时间戳策略: 为每帧添加时间戳,记录实际捕获时间

    应用场景:
        - 立体视觉: 左右摄像头同步采集
        - 多角度监控: 同时查看不同视角
        - 深度学习: 同步采集多视角数据用于3D重建
    """
    print_section("练习 7: 多摄像头同步捕获")

    # 尝试打开多个摄像头
    cameras = []
    print(f"扫描可用摄像头 (最多 {max_cameras} 个)...")

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 获取摄像头信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  摄像头 {i}: {width}x{height}, {fps:.2f} FPS ✓")
            cameras.append((i, cap))
        else:
            print(f"  摄像头 {i}: 不可用 ✗")

    if not cameras:
        print("\n错误: 没有找到可用的摄像头")
        print("提示: 如果有外接摄像头,请确保它已正确连接")
        return

    print(f"\n找到 {len(cameras)} 个可用摄像头")

    # 创建显示窗口
    window_name = "练习7 - 多摄像头同步"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 设置窗口布局(根据摄像头数量)
    n_cameras = len(cameras)
    if n_cameras == 1:
        layout = (1, 1)
    elif n_cameras == 2:
        layout = (1, 2)
    elif n_cameras <= 4:
        layout = (2, 2)
    else:
        layout = (2, (n_cameras + 1) // 2)

    frame_count = 0
    start_time = time.time()

    print("\n开始同步捕获...")
    print("按 'q' 或 ESC 退出")

    # 同步捕获循环
    while True:
        frames = []
        success_all = True

        # 同步读取所有摄像头
        for cam_idx, cap in cameras:
            success, frame = cap.read()
            if not success or frame is None:
                print(f"\n警告: 摄像头 {cam_idx} 读取失败")
                success_all = False
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                frames.append(frame)

        if not success_all:
            break

        frame_count += 1

        # 创建拼接画面
        # 计算每行每列的帧数
        rows, cols = layout
        h, w = frames[0].shape[:2]

        # 创建画布
        if n_cameras == 1:
            display = frames[0].copy()
        else:
            # 网格布局显示
            top_row = np.hstack(frames[:cols]) if n_cameras >= cols else frames[0]
            display = top_row

            if n_cameras > cols:
                bottom_frames = []
                for i in range(cols, n_cameras):
                    bottom_frames.append(frames[i])
                if bottom_frames:
                    bottom_row = np.hstack(bottom_frames)
                    display = np.vstack([display, bottom_row])

        # 调整显示大小(如果过大)
        max_display_width = 1280
        if display.shape[1] > max_display_width:
            scale = max_display_width / display.shape[1]
            display = cv2.resize(display, None, fx=scale, fy=scale)

        # 添加帧率和摄像头数量信息
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        info = f"Cameras: {n_cameras}, FPS: {current_fps:.1f}"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        # 使用短等待时间以支持更多摄像头
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            print(f"\n用户退出 (捕获了 {frame_count} 帧)")
            break

    # 释放所有摄像头
    for cam_idx, cap in cameras:
        cap.release()

    cv2.destroyWindow(window_name)

    elapsed = time.time() - start_time
    print(f"\n捕获统计:")
    print(f"  总帧数: {frame_count}")
    print(f"  总时长: {elapsed:.2f} 秒")
    print(f"  平均帧率: {frame_count/elapsed:.2f} FPS")


# =============================================================================
# 练习 8: 实现视频处理流水线
# =============================================================================

def exercise_8_processing_pipeline(video_path: Optional[str] = None,
                                    output_path: str = "processed_output.mp4") -> None:
    """
    练习 8: 实现视频处理流水线

    算法原理:
        1. 建立视频读取-处理-写入的流水线
        2. 读取原始帧后进行图像处理(如边缘检测、颜色变换等)
        3. 将处理后的帧写入输出视频
        4. 实时显示处理结果

    常用视频处理算法:
        - 边缘检测: Canny, Sobel, Laplacian
        - 颜色空间转换: cvtColor (BGR ↔ HSV, GRAY 等)
        - 图像滤波: blur, GaussianBlur, medianBlur, bilateralFilter
        - 形态学: erode, dilate, open, close
        - 轮廓检测: findContours, drawContours
        - 特征检测: ORB, SIFT, AKAZE (需付费版本)

    流水线设计模式:
        读取帧 → 预处理 → 执行算法 → 后处理 → 写入/显示

    颜色空间说明:
        - BGR: OpenCV 默认颜色空间(蓝-绿-红)
        - HSV: 色相-饱和度-明度,适合颜色检测
        - GRAY: 灰度图,单通道
        - YUV: 常用于视频编码
    """
    print_section("练习 8: 视频处理流水线")

    # 创建测试视频
    if video_path is None or not os.path.exists(video_path):
        video_path = "test_video_ex8.mp4"
        if not create_test_video(video_path, fps=10, duration=5):
            return

    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {video_path}")
        return

    # 获取输入视频属性
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"输入视频: {video_path}")
    print(f"分辨率: {in_width}x{in_height}")
    print(f"帧率: {in_fps:.2f} FPS")

    # 创建输出视频写入器
    # 注意: 输出编码可以根据需要选择
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, in_fps, (in_width, in_height))

    if not writer.isOpened():
        print(f"错误: 无法创建输出视频: {output_path}")
        cap.release()
        return

    print(f"\n输出视频: {output_path}")

    # 创建显示窗口
    window_name = "练习8 - 视频处理流水线"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    # 处理模式
    modes = {
        'edge': '边缘检测 (Canny)',
        'gray': '灰度转换',
        'blur': '高斯模糊',
        'hsv': 'HSV颜色空间',
        'original': '原始帧',
    }

    current_mode = 'edge'
    print(f"\n处理模式: {modes[current_mode]}")
    print("切换模式: '1'=边缘 '2'=灰度 '3'=模糊 '4'=HSV '0'=原始")
    print("按 'q' 或 ESC 退出")

    # 主处理循环
    while True:
        # 读取一帧
        success, frame = cap.read()
        if frame is None or frame.size == 0:
            print(f"\n视频处理完毕! (处理了 {frame_count} 帧)")
            break

        frame_count += 1
        processed = None

        # 根据当前模式处理帧
        if current_mode == 'edge':
            # 边缘检测算法 (Canny)
            # 原理:
            # 1. 使用高斯模糊降噪
            # 2. 计算梯度 (Sobel算子)
            # 3. 非极大值抑制
            # 4. 边缘连接和双阈值处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Canny 参数: 阈值1, 阈值2
            # 阈值越小,检测到的边缘越敏感(越多)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
            edges = cv2.Canny(blurred, 50, 150)
            # 将灰度边缘图转换回 BGR 以便保存和显示
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif current_mode == 'gray':
            # 灰度转换
            # 原理: 加权求和 R*0.299 + G*0.587 + B*0.114
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif current_mode == 'blur':
            # 高斯模糊
            # 原理: 使用高斯核进行卷积,平滑图像
            # 参数: 核大小, 标准差
            processed = cv2.GaussianBlur(frame, (15, 15), 0)

        elif current_mode == 'hsv':
            # HSV 颜色空间转换
            # H (0-180): 色相 - 决定颜色种类
            # S (0-255): 饱和度 - 决定颜色浓淡
            # V (0-255): 明度 - 决定颜色明暗
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 仅保留色调信息,饱和度和明度设为最大值
            # 这样可以显示颜色分类结果
            hue_channel = hsv[:, :, 0]
            hue_display = cv2.cvtColor(
                np.dstack([hue_channel, hue_channel, hue_channel]).astype(np.uint8),
                cv2.COLOR_GRAY2BGR)
            processed = hue_display

        elif current_mode == 'original':
            processed = frame

        # 写入输出视频
        writer.write(processed)

        # 叠加处理信息
        display = processed.copy()
        mode_text = f"Mode: {modes[current_mode]}"
        cv2.putText(display, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {frame_count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, display)

        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            current_mode = 'edge'
            print(f"切换模式: {modes[current_mode]}")
        elif key == ord('2'):
            current_mode = 'gray'
            print(f"切换模式: {modes[current_mode]}")
        elif key == ord('3'):
            current_mode = 'blur'
            print(f"切换模式: {modes[current_mode]}")
        elif key == ord('4'):
            current_mode = 'hsv'
            print(f"切换模式: {modes[current_mode]}")
        elif key == ord('0'):
            current_mode = 'original'
            print(f"切换模式: {modes[current_mode]}")

    # 释放资源
    elapsed = time.time() - start_time
    cap.release()
    writer.release()
    cv2.destroyWindow(window_name)

    print(f"\n处理完成:")
    print(f"  处理帧数: {frame_count}")
    print(f"  处理时间: {elapsed:.2f} 秒")
    print(f"  平均帧率: {frame_count/elapsed:.2f} FPS")
    print(f"  输出文件: {output_path}")


# =============================================================================
# 练习 9: 实现视频水印添加功能
# =============================================================================

def exercise_9_video_watermark(video_path: Optional[str] = None,
                               output_path: str = "watermarked_output.mp4",
                               watermark_text: str = "OpenCV Videoio",
                               logo_path: Optional[str] = None) -> None:
    """
    练习 9: 实现视频水印添加功能

    算法原理:
        1. 文字水印: 使用 putText 在帧上绘制文字
        2. 图片水印: 使用 addWeighted 或 cv2.rectangle + putText 叠加
        3. 半透明水印: 使用 cv2.addWeighted 实现透明度混合
        4. 动态水印: 根据帧位置计算水印参数

    水印技术对比:
        | 类型 | 方法 | 适用场景 |
        |------|------|---------|
        | 文字水印 | putText | 版权信息、时间戳 |
        | 图片水印 | addWeighted | Logo、图标 |
        | 半透明水印 | addWeighted | 可见但不遮挡主体 |
        | 动态水印 | 公式计算 | 移动水印、缩放水印 |

    addWeighted 函数原理:
        dst = src1 * alpha + src2 * beta + gamma
        用于混合两张图像或给图像添加透明度效果

    参数说明:
        watermark_text: 水印文字内容
        logo_path: 水印图片路径(可选)
    """
    print_section("练习 9: 视频水印添加功能")

    # 创建测试视频
    if video_path is None or not os.path.exists(video_path):
        video_path = "test_video_ex9.mp4"
        if not create_test_video(video_path, fps=10, duration=5):
            return

    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {video_path}")
        return

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"输入视频: {video_path}")
    print(f"分辨率: {in_width}x{in_height}")

    # 配置输出 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, in_fps, (in_width, in_height))

    if not writer.isOpened():
        print(f"错误: 无法创建输出视频: {output_path}")
        cap.release()
        return

    # 创建显示窗口
    window_name = "练习9 - 视频水印"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    # 水印配置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # 获取文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(
        watermark_text, font, font_scale, thickness)

    # 水印位置(右下角,留边距)
    margin = 20
    watermark_x = in_width - text_width - margin
    watermark_y = in_height - margin

    print(f"\n水印配置:")
    print(f"  文字: {watermark_text}")
    print(f"  位置: 右下角 ({watermark_x}, {watermark_y})")
    print(f"  尺寸: {text_width}x{text_height}")

    # 水印模式选择
    watermark_mode = 'static'  # 'static', 'dynamic', 'animated'

    print("\n水印模式: 's'=静态 'd'=动态 'a'=动画 'q'=退出")

    # 主循环
    while True:
        success, frame = cap.read()
        if frame is None or frame.size == 0:
            print(f"\n处理完毕! (处理了 {frame_count} 帧)")
            break

        frame_count += 1
        display = frame.copy()

        if watermark_mode == 'static':
            # 静态水印: 固定位置、固定样式
            # 在帧上绘制带阴影的文字水印
            position = (watermark_x, watermark_y)

            # 绘制阴影(产生立体感)
            shadow_color = (0, 0, 0)
            cv2.putText(display, watermark_text, (position[0] + 2, position[1] + 2),
                        font, font_scale, shadow_color, thickness + 1)
            # 绘制文字
            text_color = (255, 255, 255)
            cv2.putText(display, watermark_text, position,
                        font, font_scale, text_color, thickness)

        elif watermark_mode == 'dynamic':
            # 动态水印: 位置随帧变化
            # 计算水印位置(在帧四周移动)
            t = frame_count / in_fps  # 当前时间
            pos_x = int((np.sin(t * 0.5) + 1) / 2 * (in_width - text_width - 2*margin) + margin)
            pos_y = int((np.cos(t * 0.3) + 1) / 2 * (in_height - text_height - 2*margin) + margin + text_height)

            cv2.putText(display, watermark_text, (pos_x, pos_y),
                        font, font_scale, (255, 255, 255), thickness)

        elif watermark_mode == 'animated':
            # 动画水印: 透明度随时间变化
            # 使用正弦波控制透明度,产生闪烁效果
            t = frame_count / in_fps
            alpha = (np.sin(t * 3) + 1) / 2  # 0 到 1 之间变化
            alpha = max(0.3, min(1.0, alpha))  # 限制在 0.3-1.0 范围

            # 创建水印图像(白色文字)
            watermark_img = np.zeros_like(frame)
            cv2.putText(watermark_img, watermark_text, (watermark_x, watermark_y),
                        font, font_scale, (255, 255, 255), thickness)

            # 创建背景掩码(用于限定水印文字区域)
            # 使用颜色阈值提取文字区域
            watermark_gray = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(watermark_gray, 1, 255, cv2.THRESH_BINARY)

            # 使用 addWeighted 叠加水印(带透明度)
            display = cv2.addWeighted(frame, 1, watermark_img, alpha * 0.3, 0)

        # 叠加帧号和时间戳
        elapsed = frame_count / in_fps
        timestamp = f"{elapsed:.1f}s"
        info_text = f"Frame: {frame_count} | {timestamp}"
        cv2.putText(display, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示模式信息
        mode_text = f"Mode: {watermark_mode.upper()}"
        cv2.putText(display, mode_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name, display)

        # 写入输出视频
        writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            watermark_mode = 'static'
            print(f"切换水印模式: 静态")
        elif key == ord('d'):
            watermark_mode = 'dynamic'
            print(f"切换水印模式: 动态")
        elif key == ord('a'):
            watermark_mode = 'animated'
            print(f"切换水印模式: 动画")

    # 释放资源
    elapsed = time.time() - start_time
    cap.release()
    writer.release()
    cv2.destroyWindow(window_name)

    print(f"\n水印添加完成:")
    print(f"  处理帧数: {frame_count}")
    print(f"  输出文件: {output_path}")


# =============================================================================
# 练习 10: 实现自定义视频编码器后端
# =============================================================================

def exercise_10_custom_encoder(output_path: str = "custom_encoded.mp4") -> None:
    """
    练习 10: 实现自定义视频编码器后端

    算法原理:
        1. 使用 cv2.VideoWriter 时指定不同的 fourcc 编码
        2. 创建测试帧序列用于演示不同编码效果
        3. 测量不同编码的压缩率和处理速度
        4. 对比多种编码格式的输出质量

    FourCC 编码格式详解:
        'mp4v' - MPEG-4 Part 2: 通用性好,压缩率中等
        'H264' - H.264/AVC: 高压缩率,需要 x264 编码器
        'X264' - x264: 开源 H.264 编码器
        'XVID' - Xvid: 开源 MPEG-4,兼容性最好
        'MJPG' - Motion JPEG: 无压缩,质量最高但文件大
        'AVC1' - H.264 with Annex A
        'YUY2' - YUV 4:2:2 未压缩格式

    实际应用:
        - 存档: 使用 MJPG (无损)
        - 网络传输: 使用 H264 (高压缩)
        - 跨平台分享: 使用 XVID 或 mp4v
    """
    print_section("练习 10: 自定义视频编码器后端")

    # 编码格式列表
    codecs = [
        ('mp4v', 'MPEG-4 Part 2', 'mp4'),
        ('XVID', 'Xvid MPEG-4', 'avi'),
        ('MJPG', 'Motion JPEG', 'avi'),
    ]

    # 测试视频参数
    fps = 30.0
    width, height = 640, 480
    duration = 2.0  # 秒
    total_frames = int(fps * duration)

    print(f"测试参数:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  时长: {duration} 秒")
    print(f"  总帧数: {total_frames}")

    results = []

    for fourcc_str, codec_name, ext in codecs:
        output_file = f"{output_path.rsplit('.', 1)[0]}_{fourcc_str}.{ext}"

        print(f"\n{'='*40}")
        print(f"编码器: {codec_name} ({fourcc_str})")

        # 创建 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"  错误: 编码器 {fourcc_str} 不可用")
            continue

        start_time = time.time()

        # 生成测试帧
        for i in range(total_frames):
            # 创建渐变帧(不同编码会有不同的压缩效果)
            t = i / fps
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # 动态颜色变化
            frame[:, :, 0] = int((np.sin(t * 2) + 1) * 127)  # Blue
            frame[:, :, 1] = int((np.cos(t * 3) + 1) * 127)  # Green
            frame[:, :, 2] = int((np.sin(t * 5) + 1) * 127)  # Red

            # 添加帧号和编码信息
            cv2.putText(frame, f"Frame: {i+1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Codec: {fourcc_str}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            writer.write(frame)

        writer.release()
        elapsed = time.time() - start_time

        # 获取输出文件大小
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            bitrate = (file_size * 8) / (duration * 1000)  # kbps
            compression = (width * height * 3 * total_frames) / file_size

            result = {
                'codec': fourcc_str,
                'name': codec_name,
                'fps': total_frames / elapsed,
                'size': file_size,
                'bitrate': bitrate,
                'compression': compression,
            }
            results.append(result)

            print(f"  文件: {output_file}")
            print(f"  大小: {file_size / 1024:.2f} KB")
            print(f"  耗时: {elapsed:.3f} 秒")
            print(f"  实际帧率: {result['fps']:.2f} FPS")
            print(f"  码率: {bitrate:.1f} kbps")
            print(f"  压缩比: {compression:.1f}x")
        else:
            print(f"  错误: 输出文件未生成")

    # 对比结果
    if results:
        print(f"\n{'='*60}")
        print(f"编码器对比结果:")
        print(f"{'编码器':<10} {'大小(KB)':<12} {'码率(kbps)':<12} {'压缩比':<10}")
        print("-" * 44)

        for r in sorted(results, key=lambda x: x['size']):
            print(f"{r['codec']:<10} {r['size']/1024:<12.2f} {r['bitrate']:<12.1f} {r['compression']:<10.1f}")


# =============================================================================
# 练习 11: 实现 RTSP 流媒体服务器推送
# =============================================================================

def exercise_11_rtsp_push(rtsp_url: Optional[str] = None) -> None:
    """
    练习 11: 实现 RTSP 流媒体服务器推送

    算法原理:
        1. 使用 VideoWriter 打开 RTSP URL(需要 FFmpeg 支持)
        2. 构建 RTSP 推送命令(基于 FFmpeg 或 gstreamer)
        3. 循环写入帧到流媒体服务器
        4. 处理连接断开等异常情况

    RTSP 推送协议说明:
        - RTSP (Real Time Streaming Protocol): 实时流协议
        - RTP (Real-time Transport Protocol): 实时传输协议
        - 通常配合使用: RTSP/RTP

    常见流媒体服务器:
        - FFmpeg server: ffmpeg -i video.mp4 -c:v copy -f rtsp rtsp://localhost:8554/stream
        - VLC: 支持 RTSP 服务器模式
        - MediaMTX (gstreamer): 开源 RTSP 服务器
        - Wowza, Red5: Java 实现的流媒体服务器

    RTSP URL 格式:
        rtsp://[用户名:密码@]服务器地址:端口/流名称

    重要提示:
        - RTSP 推送需要流媒体服务器支持
        - Windows 上 OpenCV 需要编译时选择 FFmpeg 后端
        - 需要使用与环境匹配的 FFmpeg 库

    本练习演示:
        - 本地模拟 RTSP 推送(创建测试流)
        - 使用 cv2.VideoCapture 检查后端支持
        - 显示可用的视频后端信息
    """
    print_section("练习 11: RTSP 流媒体服务器推送")

    # 注意事项
    print("注意: RTSP 推送需要:")
    print("  1. OpenCV 编译时支持 FFmpeg")
    print("  2. 有可用的 RTSP 流媒体服务器")
    print("  3. 网络配置允许 UDP 传输")

    # 检查可用的后端
    print("\n检查 OpenCV 支持的视频后端...")
    backends = [
        (cv2.CAP_ANY, "自动选择"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_MSMF, "MSMF (Windows)"),
        (cv2.CAP_V4L2, "V4L2 (Linux)"),
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)"),
    ]

    available_backends = []
    for backend_id, backend_name in backends:
        # 尝试使用每个后端创建 VideoCapture
        test_cap = cv2.VideoCapture(0, backend_id)
        if test_cap.isOpened():
            available_backends.append((backend_id, backend_name))
            test_cap.release()

    print(f"可用的后端: {len(available_backends)}")
    for backend_id, backend_name in available_backends:
        print(f"  - {backend_name} (ID: {backend_id})")

    # RTSP 服务器配置
    if rtsp_url is None:
        # 默认测试 URL(需要替换为实际的 RTSP 服务器地址)
        rtsp_url = "rtsp://localhost:8554/teststream"
        print(f"\n默认 RTSP URL: {rtsp_url}")
        print("(请根据实际流媒体服务器修改)")

    # 演示本地视频流处理流程
    print("\n" + "="*40)
    print("演示: 本地视频流处理")

    # 创建测试视频
    test_video = "test_stream.mp4"
    if not create_test_video(test_video, fps=10, duration=3):
        return

    print(f"\n创建测试视频流: {test_video}")

    # 打开视频(模拟从文件读取后进行流推送)
    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print("错误: 无法打开测试视频")
        return

    print("视频流读取成功!")
    print("\n要推送到 RTSP 服务器,需要:")
    print("  1. 启动流媒体服务器(如 MediaMTX)")
    print("  2. 使用 FFmpeg 建立推流:")
    print("     ffmpeg -i input.mp4 -c:v copy -f rtsp rtsp://localhost:8554/stream")
    print("  3. 或者使用 Python 的 ffmpeg-python/av 库")

    # 显示可用的流处理库
    print("\n可用的流处理库:")
    print("  - ffmpeg-python: pip install ffmpeg-python")
    print("  - PyAV: pip install av")
    print("  - imageio-ffmpeg: pip install imageio-ffmpeg")

    # 示例代码(概念)
    print("\n概念示例代码:")
    print("-" * 40)
    print("""
    # 使用 FFmpeg C API (需要编译 OpenCV with FFmpeg)
    import subprocess

    # 启动 FFmpeg 进程进行 RTSP 推送
    ffmpeg_cmd = [
        'ffmpeg',
        '-re',                          # 以原始帧率读取
        '-i', 'input.mp4',              # 输入文件
        '-c:v', 'libx264',             # H.264 编码
        '-preset', 'fast',              # 编码预设
        '-f', 'rtsp',                  # 输出格式
        'rtsp://server/stream'          # RTSP 地址
    ]
    process = subprocess.Popen(ffmpeg_cmd)
    """)

    # 帧统计
    frame_count = 0
    start_time = time.time()

    while True:
        success, frame = cap.read()
        if frame is None or frame.size == 0:
            break
        frame_count += 1

        # 模拟实时传输延迟
        time.sleep(0.1)

        if frame_count >= 30:  # 限制帧数以免运行太久
            print(f"\n演示完成 (读取了 {frame_count} 帧)")
            break

    cap.release()

    elapsed = time.time() - start_time
    print(f"\n流读取统计:")
    print(f"  读取帧数: {frame_count}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均帧率: {frame_count/elapsed:.2f} FPS")


# =============================================================================
# 主函数 - 运行所有练习
# =============================================================================

def main() -> None:
    """
    主函数 - 运行所有练习题

    每个练习都会打印清晰的标题和输出,便于验证结果。
    按练习编号顺序执行,每个练习完成后等待用户确认。

    练习覆盖范围:
        - 入门级 (1-3): 基础读写操作
        - 中级 (4-6): 播放控制和格式转换
        - 高级 (7-9): 多摄像头、处理流水线、水印
        - 挑战题 (10-11): 编码器和流媒体
    """
    print("\n" + "="*60)
    print("  OpenCV Videoio 模块 - 练习题集")
    print("="*60)
    print("\n作者: OpenCV Learning Team")
    print("版本: 1.0")
    print("\n练习列表:")
    print("  入门级:")
    print("    1. 读取视频文件并逐帧显示")
    print("    2. 打开摄像头并实时显示")
    print("    3. 从摄像头捕获视频并保存到文件")
    print("  中级:")
    print("    4. 实现视频播放器的暂停/继续功能")
    print("    5. 实现视频的时间跳转功能")
    print("    6. 实现批量视频格式转换")
    print("  高级:")
    print("    7. 实现多摄像头同步捕获")
    print("    8. 实现视频处理流水线")
    print("    9. 实现视频水印添加功能")
    print("  挑战题:")
    print("    10. 实现自定义视频编码器后端")
    print("    11. 实现 RTSP 流媒体服务器推送")

    # 练习选择
    print("\n" + "="*60)
    print("请选择要运行的练习:")
    print("  输入数字 1-11: 运行指定练习")
    print("  输入 'all': 运行所有练习")
    print("  输入 'q': 退出程序")
    print("="*60)

    while True:
        choice = input("\n> ").strip().lower()

        if choice == 'q':
            print("\n感谢使用! 再见.")
            break

        if choice == 'all':
            # 运行所有练习
            exercises = [
                (exercise_1_read_video_file, "练习 1: 读取视频文件并逐帧显示"),
                (exercise_2_camera_realtime, "练习 2: 打开摄像头并实时显示"),
                (exercise_3_camera_capture_and_save, "练习 3: 从摄像头捕获视频并保存到文件"),
                (exercise_4_pause_resume, "练习 4: 实现视频播放器的暂停/继续功能"),
                (exercise_5_seek_video, "练习 5: 实现视频的时间跳转功能"),
                (exercise_6_batch_convert, "练习 6: 实现批量视频格式转换"),
                (exercise_7_multi_camera_sync, "练习 7: 实现多摄像头同步捕获"),
                (exercise_8_processing_pipeline, "练习 8: 实现视频处理流水线"),
                (exercise_9_video_watermark, "练习 9: 实现视频水印添加功能"),
                (exercise_10_custom_encoder, "练习 10: 实现自定义视频编码器后端"),
                (exercise_11_rtsp_push, "练习 11: 实现 RTSP 流媒体服务器推送"),
            ]

            for exercise_func, exercise_name in exercises:
                print(f"\n\n{'#'*60}")
                print(f"## {exercise_name}")
                print(f"{'#'*60}")

                try:
                    # 某些练习需要用户交互或摄像头,捕获异常
                    if exercise_func in [exercise_2_camera_realtime,
                                         exercise_3_camera_capture_and_save,
                                         exercise_7_multi_camera_sync]:
                        # 这些练习需要摄像头,如果摄像头不可用则跳过
                        print("(注意: 此练习需要摄像头,可能需要手动确认)")
                        exercise_func()
                    elif exercise_func == exercise_5_seek_video:
                        # 这个练习需要交互式输入
                        print("(注意: 此练习需要交互式输入)")
                        exercise_func()
                    else:
                        exercise_func()
                except Exception as e:
                    print(f"练习执行出错: {e}")
                    import traceback
                    traceback.print_exc()

                # 暂停,等待用户确认继续
                input("\n按 Enter 键继续...")

            print("\n所有练习执行完毕!")
            break

        else:
            # 运行指定练习
            try:
                exercise_num = int(choice)
                if exercise_num < 1 or exercise_num > 11:
                    print("请输入 1-11 之间的数字,或 'all' 或 'q'")
                    continue

                exercise_map = {
                    1: (exercise_1_read_video_file, "练习 1"),
                    2: (exercise_2_camera_realtime, "练习 2"),
                    3: (exercise_3_camera_capture_and_save, "练习 3"),
                    4: (exercise_4_pause_resume, "练习 4"),
                    5: (exercise_5_seek_video, "练习 5"),
                    6: (exercise_6_batch_convert, "练习 6"),
                    7: (exercise_7_multi_camera_sync, "练习 7"),
                    8: (exercise_8_processing_pipeline, "练习 8"),
                    9: (exercise_9_video_watermark, "练习 9"),
                    10: (exercise_10_custom_encoder, "练习 10"),
                    11: (exercise_11_rtsp_push, "练习 11"),
                }

                exercise_func, exercise_name = exercise_map[exercise_num]
                print(f"\n运行: {exercise_name}")

                try:
                    exercise_func()
                except Exception as e:
                    print(f"练习执行出错: {e}")
                    import traceback
                    traceback.print_exc()

            except ValueError:
                print("无效输入,请输入数字 1-11, 'all' 或 'q'")


if __name__ == "__main__":
    main()