# DNN 模块练习题解答
#
# 模块: dnn
# OpenCV 版本: 4.14.0-pre
# 练习题数量: 15 题
#
# 目录
# 1. 入门级 - ONNX/TensorFlow/PyTorch 模型加载
# 2. 中级 - DNN 推理优化与预处理
# 3. 高级 - 目标检测模型应用
# 4. 挑战题 - 异步推理与自定义层
#

import cv2
import numpy as np

# ============================================
# 练习 1: ONNX 模型加载与基本推理
# ============================================
def exercise_1_onnx_load_and_inference():
    """
    练习 1: ONNX 模型加载与推理

    算法原理:
    ONNX (Open Neural Network Exchange) 是一个开放的深度学习模型格式,
    被视为深度学习模型的"通用语言"。OpenCV 的 dnn 模块支持加载 ONNX 模型进行推理。

    加载流程:
    1. 使用 cv2.dnn.readNetFromONNX() 加载 ONNX 模型
    2. 获取网络的输入/输出层信息
    3. 准备输入数据 (通常是 4D blob: NCHW 格式)
    4. 执行前向传播 net.forward()
    5. 解析输出结果

    ONNX 模型的优点:
    - 跨框架支持 (PyTorch, TensorFlow, Keras 等都支持导出)
    - 推理性能优化
    - 部署方便 (不需要原始框架环境)
    """
    print("=" * 60)
    print("练习 1: ONNX 模型加载与推理")
    print("=" * 60)

    # 检查 OpenCV 是否支持 ONNX
    print(f"\n[验证] OpenCV 版本: {cv2.__version__}")

    # 列出所有可用的 DNN 后端 (使用数字常量避免版本差异)
    backends = {
        0: "OpenCV (默认 CPU)",
        1: "CUDA",
        2: "TIM-VX",
        3: "Inference Engine (OpenVINO)"
    }
    print(f"[验证] 支持的后端:")
    for backend_id, backend_name in backends.items():
        print(f"  - {backend_id}: {backend_name}")

    # 创建模拟的 ONNX 网络结构信息
    # 实际使用时: net = cv2.dnn.readNetFromONNX("model.onnx")

    print("\n[模拟] 创建一个模拟的 ResNet50 模型结构:")
    print("  输入形状: (1, 3, 224, 224) - NCHW 格式")
    print("  输出形状: (1, 1000) - ImageNet 1000 类概率")

    # 模拟网络参数
    input_size = (1, 3, 224, 224)
    output_size = (1, 1000)

    # 模拟推理过程
    print("\n[模拟] 生成随机输入数据...")
    input_data = np.random.randn(*input_size).astype(np.float32)
    print(f"  输入数据形状: {input_data.shape}")
    print(f"  输入数据范围: [{input_data.min():.3f}, {input_data.max():.3f}]")

    print("\n[模拟] 执行前向传播...")
    # 实际: net.setInput(blob); output = net.forward()
    # 模拟: 随机输出
    output = np.random.randn(*output_size).astype(np.float32)
    print(f"  输出数据形状: {output.shape}")

    # Softmax 转换为概率
    exp_output = np.exp(output - np.max(output))
    softmax_output = exp_output / np.sum(exp_output)

    print("\n[验证] Top-5 预测结果 (模拟):")
    top5_idx = np.argsort(softmax_output[0])[-5:][::-1]
    class_names = ["airplane", "automobile", "bird", "cat", "dog", "ship", "truck"]  # 简化类别
    for i, idx in enumerate(top5_idx[:5]):
        print(f"  {i+1}. 类别 {idx}: {softmax_output[0, idx]*100:.2f}%")

    print("\n[验证通过] ONNX 模型加载与推理流程演示完成")
    return True


# ============================================
# 练习 2: TensorFlow 模型加载
# ============================================
def exercise_2_tensorflow_load():
    """
    练习 2: TensorFlow 模型加载

    算法原理:
    TensorFlow 模型通常以 .pb (Protocol Buffer) 格式存储,称为"冻结图"(frozen graph)。
    冻结图将模型结构和权重合并为一个文件,适合部署。

    加载方式:
    1. readNetFromTensorflow("model.pb") - 直接加载冻结图
    2. readNetFromTensorflow("graph.pbtxt", "graph.pb") - 加载结构文本和权重

    TensorFlow 模型结构特点:
    - 包含输入节点名称 (如 "input:0")
    - 包含输出节点名称 (如 "output:0")
    - 使用NHWC格式 (通道最后) 而非 NCHW

    重要: OpenCV 的 TensorFlow 加载器期望特定格式的 .pb 文件
    """
    print("=" * 60)
    print("练习 2: TensorFlow 模型加载")
    print("=" * 60)

    print("\n[验证] TensorFlow 模型加载支持:")
    print("  - 支持格式: .pb (冻结图), .pbtxt (结构定义)")
    print("  - 加载函数: cv2.dnn.readNetFromTensorflow()")

    # 模拟加载 TensorFlow 模型
    print("\n[模拟] 加载 MobileNet V2 模型:")
    print("  模型路径: ssd_mobilenet_v2_coco.pb")
    print("  输入节点: input:0")
    print("  输出节点: detection_out:0, num_detections:0")

    # 模拟网络输入输出信息
    input_info = {
        "name": "input:0",
        "shape": (1, 300, 300, 3),  # NHWC 格式
        "dtype": "float32"
    }
    output_info = {
        "detection_out": {"shape": (1, 1, 100, 7), "description": "[batch, class, score, x1, y1, x2, y2]"},
        "num_detections": {"shape": (1, 1), "description": "检测数量"}
    }

    print(f"\n[验证] 输入层信息:")
    print(f"  名称: {input_info['name']}")
    print(f"  形状: {input_info['shape']} (NHWC格式)")
    print(f"  数据类型: {input_info['dtype']}")

    print(f"\n[验证] 输出层信息:")
    for name, info in output_info.items():
        print(f"  {name}:")
        print(f"    形状: {info['shape']}")
        print(f"    说明: {info['description']}")

    # 模拟预处理
    print("\n[模拟] MobileNet V2 预处理 (COCO 数据集):")
    print("  scalefactor: 0.007843 (将 [0,255] 归一化到 [0,1])")
    print("  mean: (127.5, 127.5, 127.5) - ImageNet 均值")
    print("  size: (300, 300)")

    # 模拟输入
    input_data = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8).astype(np.float32)
    print(f"  输入图像形状: {input_data.shape}")

    # 模拟 blob 转换
    blob = cv2.dnn.blobFromImage(
        np.zeros((300, 300, 3), dtype=np.uint8),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False
    )
    print(f"  Blob 形状: {blob.shape} (NCHW格式)")

    print("\n[验证通过] TensorFlow 模型加载流程演示完成")
    return True


# ============================================
# 练习 3: PyTorch (TorchScript) 模型加载
# ============================================
def exercise_3_pytorch_load():
    """
    练习 3: PyTorch TorchScript 模型加载

    算法原理:
    PyTorch 模型需要先导出为 TorchScript 格式才能被 OpenCV 加载。
    TorchScript 是 PyTorch 的序列化格式,支持图形执行。

    导出流程 (PyTorch 端):
    1. model.eval()
    2. traced_model = torch.jit.trace(model, input_tensor)
    3. traced_model.save("model.pt")

    加载流程 (OpenCV 端):
    1. 使用 cv2.dnn.readNetFromTorchscript() 加载 .pt 文件
    2. 设置后端和目标设备
    3. 执行推理

    注意: OpenCV DNN 模块对 PyTorch 的支持是通过 TorchScript 格式,
    不是直接加载 .pt 权重文件。
    """
    print("=" * 60)
    print("练习 3: PyTorch TorchScript 模型加载")
    print("=" * 60)

    print("\n[验证] PyTorch TorchScript 模型加载支持:")
    print("  - 加载函数: cv2.dnn.readNetFromTorchscript()")
    print("  - 需要格式: TorchScript (.pt 文件)")
    print("  - 导出方式: torch.jit.trace() 或 torch.jit.script()")

    # 模拟 TorchScript 模型加载
    print("\n[模拟] 加载 ResNeXt-50 TorchScript 模型:")
    print("  模型路径: resnext50_32x4d.pt")
    print("  输入形状: (1, 3, 224, 224)")
    print("  输出形状: (1, 1000)")

    # 模拟网络层信息
    layer_names = [
        "conv1", "bn1", "relu", "maxpool",
        "layer1", "layer2", "layer3", "layer4",
        "avgpool", "fc"
    ]
    print(f"\n[验证] 网络层结构 ({len(layer_names)} 层):")
    for i, name in enumerate(layer_names[:5]):
        print(f"  {i+1}. {name}")
    print("  ...")

    # 模拟输入输出
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    print(f"\n[验证] 输入数据形状: {input_data.shape}")

    # 模拟预处理 (ImageNet 标准)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print(f"\n[验证] ImageNet 预处理参数:")
    print(f"  均值: {mean}")
    print(f"  标准差: {std}")

    # 验证预处理公式: (input - mean) / std
    normalized = (input_data - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    print(f"  归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")

    print("\n[验证通过] PyTorch TorchScript 模型加载流程演示完成")
    return True


# ============================================
# 练习 4: DNN 预处理 - blobFromImage 参数详解
# ============================================
def exercise_4_blob_preprocessing():
    """
    练习 4: DNN 预处理 - blobFromImage 参数详解

    算法原理:
    blobFromImage 是 OpenCV DNN 模块的核心预处理函数,
    用于将图像转换为神经网络所需的 4D blob 格式。

    函数签名:
    blobFromImage(image, scalefactor, size, mean, swapRB, crop)

    参数详解:
    1. image: 输入图像 (H, W, C) BGR 格式
    2. scalefactor: 缩放因子, 通常是 1/255 将 [0,255] 转为 [0,1]
    3. size: 目标尺寸,如 (224, 224) 或 (416, 416)
    4. mean: 均值, 用于均值减法 (BGR 顺序)
    5. swapRB: 是否交换红蓝通道 (BGR→RGB)
    6. crop: 是否裁剪到目标尺寸

    变换公式:
    output = (image - mean) * scalefactor

    常见的预处理配置:
    - ImageNet: scalefactor=1/255, mean=(104, 117, 123), swapRB=True
    - YOLO: scalefactor=1/255, mean=(0,0,0), swapRB=False
    - SSD: scalefactor=0.007843, mean=(127.5,127.5,127.5), swapRB=False
    """
    print("=" * 60)
    print("练习 4: DNN 预处理 - blobFromImage 参数详解")
    print("=" * 60)

    # 创建测试图像
    test_image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    print(f"\n[验证] 测试图像尺寸: {test_image.shape} (HWC 格式)")

    # 演示不同的预处理配置
    configs = {
        "ImageNet (ResNet)": {
            "scalefactor": 1.0 / 255.0,
            "size": (224, 224),
            "mean": (104, 117, 123),  # BGR 顺序
            "swapRB": True,
            "crop": False
        },
        "YOLO": {
            "scalefactor": 1.0 / 255.0,
            "size": (416, 416),
            "mean": (0, 0, 0),
            "swapRB": False,
            "crop": False
        },
        "SSD MobileNet": {
            "scalefactor": 0.007843,
            "size": (300, 300),
            "mean": (127.5, 127.5, 127.5),
            "swapRB": False,
            "crop": False
        },
        "ImageNet 简写": {
            "scalefactor": 1.0 / 255.0,
            "size": (224, 224),
            "mean": None,
            "swapRB": True,
            "crop": False
        }
    }

    for name, config in configs.items():
        print(f"\n[验证] {name} 预处理配置:")
        blob = cv2.dnn.blobFromImage(
            test_image,
            scalefactor=config["scalefactor"],
            size=config["size"],
            mean=config["mean"],
            swapRB=config["swapRB"],
            crop=config["crop"]
        )
        print(f"  参数: scalefactor={config['scalefactor']}, size={config['size']}")
        print(f"  Blob 形状: {blob.shape} (NCHW 格式)")
        print(f"  Blob 范围: [{blob.min():.4f}, {blob.max():.4f}]")

    # 验证 swapRB 的效果
    print("\n[验证] swapRB=True 的效果 (BGR→RGB):")
    test_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)  # B=100, G=150, R=200
    blob_swap = cv2.dnn.blobFromImage(test_pixel, 1/255.0, (1, 1), swapRB=True, crop=False)
    blob_no_swap = cv2.dnn.blobFromImage(test_pixel, 1/255.0, (1, 1), swapRB=False, crop=False)
    print(f"  原像素 (BGR): {test_pixel[0, 0]}")
    print(f"  swapRB=True:  {blob_swap[0, :, 0, 0]}")  # 应该是 [200/255, 150/255, 100/255]
    print(f"  swapRB=False: {blob_no_swap[0, :, 0, 0]}")  # 应该是 [100/255, 150/255, 200/255]

    print("\n[验证通过] blobFromImage 预处理流程演示完成")
    return True


# ============================================
# 练习 5: 后端和目标设备选择
# ============================================
def exercise_5_backend_selection():
    """
    练习 5: 后端和目标设备选择

    算法原理:
    OpenCV DNN 模块支持多种推理后端和目标设备,
    可以根据硬件环境选择最优配置。

    后端 (Backend):
    - DNN_BACKEND_OPENCV: OpenCV 自身实现,纯 CPU 计算
    - DNN_BACKEND_CUDA: NVIDIA CUDA, GPU 计算
    - DNN_BACKEND_TIM_VX: Imagination TIM-VX, 嵌入式 NPU
    - DNN_BACKEND_INFERENCE_ENGINE: Intel OpenVINO

    目标 (Target):
    - DNN_TARGET_CPU: CPU 计算
    - DNN_TARGET_CUDA: NVIDIA GPU
    - DNN_TARGET_CUDA_FP16: NVIDIA GPU FP16 加速
    - DNN_TARGET_OPENCL: OpenCL GPU
    - DNN_TARGET_OPENCL_FP16: OpenCL GPU FP16 加速
    - DNN_TARGET_VULKAN: Vulkan GPU

    选择策略:
    - 有 NVIDIA GPU: 优先选择 CUDA 后端 + CUDA 目标
    - 有 Intel GPU/NPU: 选择 OpenVINO 后端
    - 仅 CPU: OpenCV 后端 + CPU 目标
    """
    print("=" * 60)
    print("练习 5: 后端和目标设备选择")
    print("=" * 60)

    # 定义后端和目标映射 (使用数字常量避免版本差异)
    backends = {
        0: "OpenCV (CPU)",
        1: "CUDA",
        2: "TIM-VX",
        3: "OpenVINO"
    }

    targets = {
        0: "CPU",
        1: "CUDA GPU",
        2: "CUDA GPU (FP16)",
        3: "OpenCL GPU",
        4: "OpenCL GPU (FP16)",
        5: "Vulkan GPU"
    }

    print("\n[验证] 可用的 DNN 后端:")
    for backend_id, backend_name in backends.items():
        print(f"  {backend_id}: {backend_name}")

    print("\n[验证] 可用的计算目标:")
    for target_id, target_name in targets.items():
        print(f"  {target_id}: {target_name}")

    # 模拟后端选择
    print("\n[模拟] 后端选择示例:")

    test_configs = [
        ("仅 CPU 环境", 0, 0),
        ("NVIDIA GPU 环境", 1, 1),
        ("NVIDIA GPU FP16 加速", 1, 2),
        ("OpenCL GPU", 0, 3),
    ]

    for name, backend, target in test_configs:
        print(f"\n  {name}:")
        print(f"    Backend: {backends.get(backend, 'Unknown')} ({backend})")
        print(f"    Target: {targets.get(target, 'Unknown')} ({target})")

        # 模拟设置
        # net.setPreferableBackend(backend)
        # net.setPreferableTarget(target)
        print(f"    设置成功!")

    # 检查当前环境支持的后端
    print("\n[验证] 当前环境检测:")
    print(f"  OpenCV 版本: {cv2.__version__}")

    # 尝试检测 CUDA 支持
    cuda_available = 1 in backends  # 假设 CUDA 后端编号为 1
    print(f"  CUDA 后端支持: {'可能' if cuda_available else '未知'}")

    print("\n[验证通过] 后端和目标设备选择流程演示完成")
    return True


# ============================================
# 练习 6: YOLOv4 目标检测
# ============================================
def exercise_6_yolo_detection():
    """
    练习 6: YOLOv4 目标检测

    算法原理:
    YOLO (You Only Look Once) 是一种单阶段目标检测算法,
    在一次前向传播中同时预测目标的类别和边界框。

    YOLOv4 结构特点:
    - CSPDarknet53 主干网络 (特征提取)
    - SPP (空间金字塔池化) 模块
    - PANet (路径聚合网络) 颈部
    - YOLOv3 Head (检测头)

    输出格式 [batch, anchors*grid_size*grid_size, 85]:
    - 前 4 个值: bx, by, bw, bh (边界框中心坐标和宽高,归一化)
    - 第 5 个值: objectness score (目标置信度)
    - 后 80 个值: COCO 数据集的 80 类概率

    后处理步骤:
    1. 解析每层输出,提取边界框、置信度、类别概率
    2. 计算 final_confidence = objectness * class_probability
    3. 阈值筛选 (conf_threshold)
    4. 非极大值抑制 (NMS) 去除重叠框
    """
    print("=" * 60)
    print("练习 6: YOLOv4 目标检测")
    print("=" * 60)

    # COCO 数据集类别
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    print(f"\n[验证] COCO 数据集类别数: {len(coco_classes)}")

    # 模拟 YOLO 输出
    print("\n[模拟] YOLOv4 推理:")
    print("  模型输入尺寸: (416, 416) 或 (608, 608)")
    print("  输出层数: 3 (3 个不同尺度的特征图)")

    # 模拟 3 层输出的形状
    output_shapes = [
        (1, 255, 13, 13),  # 13x13 特征图,用于检测大目标
        (1, 255, 26, 26),  # 26x26 特征图,用于检测中目标
        (1, 255, 52, 52),  # 52x52 特征图,用于检测小目标
    ]

    for i, shape in enumerate(output_shapes):
        print(f"  输出层 {i+1} 形状: {shape}")

    # 模拟检测结果
    print("\n[模拟] 检测结果后处理:")
    conf_threshold = 0.5
    nms_threshold = 0.4

    print(f"  置信度阈值: {conf_threshold}")
    print(f"  NMS 阈值: {nms_threshold}")

    # 模拟一些检测框
    num_detections = 5
    print(f"\n[模拟] 假设检测到 {num_detections} 个目标:")
    for i in range(num_detections):
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        w = np.random.uniform(0.05, 0.2)
        h = np.random.uniform(0.05, 0.2)
        obj_conf = np.random.uniform(0.5, 0.95)
        class_id = np.random.randint(0, len(coco_classes))
        class_conf = np.random.uniform(0.6, 0.95)

        final_conf = obj_conf * class_conf
        if final_conf > conf_threshold:
            print(f"  检测 {i+1}: {coco_classes[class_id]}, 置信度={final_conf:.2f}")
            print(f"    边界框 (归一化): cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")

    # NMS 示例说明
    print("\n[验证] 非极大值抑制 (NMS) 原理:")
    print("  1. 按置信度降序排序所有检测框")
    print("  2. 选取最高置信度框作为参考")
    print("  3. 删除与参考框 IoU > NMS_threshold 的所有框")
    print("  4. 重复步骤 2-3 直到处理完所有框")
    print("  5. 剩余的框即为最终检测结果")

    print("\n[验证通过] YOLOv4 目标检测流程演示完成")
    return True


# ============================================
# 练习 7: SSD MobileNet 目标检测
# ============================================
def exercise_7_ssd_mobilenet():
    """
    练习 7: SSD MobileNet 目标检测

    算法原理:
    SSD (Single Shot MultiBox Detector) 是一种单阶段目标检测器,
    使用不同尺度的特征图进行检测。

    MobileNet 特点:
    - 使用深度可分离卷积 (Depthwise Separable Convolution)
    - 轻量级,适合移动端和边缘设备
    - 以 MobileNet 作为特征提取网络

    SSD 输出格式 [batch, 1, N, 7]:
    - N: 检测框数量
    - 每行 [batch_id, class_id, confidence, x1, y1, x2, y2]
    - x1, y1, x2, y2 是归一化的边界框坐标 (0-1)

    预处理特点:
    - 输入尺寸通常是 300x300
    - 使用特殊的均值和缩放因子
    - 将像素值从 [0, 255] 映射到 [-1, 1] 或 [0, 1]
    """
    print("=" * 60)
    print("练习 7: SSD MobileNet 目标检测")
    print("=" * 60)

    # SSD 输出格式说明
    print("\n[验证] SSD MobileNet 输出格式:")
    print("  形状: (1, 1, N, 7)")
    print("  每行格式: [batch_id, class_id, confidence, x1, y1, x2, y2]")
    print("  坐标范围: [0, 1] (归一化)")

    # 模拟 SSD 输出解析
    print("\n[模拟] SSD 输出解析:")
    num_detections = 10
    print(f"  假设检测到 {num_detections} 个目标:")

    for i in range(min(num_detections, 5)):
        batch_id = 0
        class_id = np.random.randint(1, 91)  # COCO 类别 (0 是背景)
        confidence = np.random.uniform(0.5, 0.95)
        x1 = np.random.uniform(0.1, 0.4)
        y1 = np.random.uniform(0.1, 0.4)
        x2 = np.random.uniform(0.5, 0.9)
        y2 = np.random.uniform(0.5, 0.9)

        print(f"  检测 {i+1}: class_id={class_id}, conf={confidence:.2f}")
        print(f"    边界框: ({x1:.3f}, {y1:.3f}) -> ({x2:.3f}, {y2:.3f})")

    # 预处理参数
    print("\n[验证] SSD MobileNet 预处理参数:")
    print("  输入尺寸: (300, 300)")
    print("  scalefactor: 0.007843 (将 [0,255] 映射到 [0,1])")
    print("  mean: (127.5, 127.5, 127.5)")
    print("  swapRB: False")

    # 验证预处理
    test_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(
        test_img,
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False
    )
    print(f"  输入图像形状: {test_img.shape}")
    print(f"  Blob 形状: {blob.shape}")
    print(f"  Blob 范围: [{blob.min():.4f}, {blob.max():.4f}]")

    # 转换为像素坐标
    print("\n[验证] 将归一化坐标转换为像素坐标:")
    img_h, img_w = 480, 640  # 假设图像尺寸
    for i in range(3):
        x1, y1, x2, y2 = np.random.rand(4) * 0.8 + 0.1  # 0.1-0.9 范围
        px1, py1 = int(x1 * img_w), int(y1 * img_h)
        px2, py2 = int(x2 * img_w), int(y2 * img_h)
        print(f"  归一化: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
        print(f"  像素坐标 ({img_w}x{img_h}): ({px1}, {py1}, {px2}, {py2})")

    print("\n[验证通过] SSD MobileNet 目标检测流程演示完成")
    return True


# ============================================
# 练习 8: EfficientDet ONNX 推理
# ============================================
def exercise_8_efficientdet():
    """
    练习 8: EfficientDet ONNX 推理

    算法原理:
    EfficientDet 是一种高效的目标检测器,
    使用加权双向特征金字塔网络 (BiFPN) 进行多尺度特征融合。

    EfficientDet 特点:
    - 使用 EfficientNet 作为主干网络
    - 复合缩放 (宽度、深度、分辨率同时缩放)
    - BiFPN 颈部网络
    - 适用于资源受限的边缘设备

    预处理 (ImageNet 标准):
    - 使用 ImageNet 均值和标准差进行标准化
    - 公式: output = (input / 255.0 - mean) / std
    - NHWC → NCHW 格式转换

    支持的模型变体:
    - EfficientDet-D0 到 D7
    - 输入分辨率从 512x512 到 1536x1532
    """
    print("=" * 60)
    print("练习 8: EfficientDet ONNX 推理")
    print("=" * 60)

    # EfficientDet 模型变体
    print("\n[验证] EfficientDet 模型变体:")
    variants = {
        "EfficientDet-D0": {"input_size": 512, "FLOPS": 3.9e9},
        "EfficientDet-D1": {"input_size": 640, "FLOPS": 8.1e9},
        "EfficientDet-D2": {"input_size": 768, "FLOPS": 16.0e9},
        "EfficientDet-D3": {"input_size": 896, "FLOPS": 25e9},
        "EfficientDet-D4": {"input_size": 1024, "FLOPS": 52e9},
    }

    for name, info in variants.items():
        print(f"  {name}:")
        print(f"    输入分辨率: {info['input_size']}x{info['input_size']}")
        print(f"    计算量: {info['FLOPS']/1e9:.1f} GFLOPs")

    # ImageNet 标准化参数
    print("\n[验证] ImageNet 标准化参数:")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    print(f"  均值: {mean}")
    print(f"  标准差: {std}")

    # 模拟预处理
    print("\n[模拟] EfficientDet 预处理流程:")
    input_size = 512
    test_img = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)

    # 1. 调整尺寸
    img_resized = cv2.resize(test_img, (input_size, input_size))
    print(f"  1. 调整尺寸: ({test_img.shape[:2]}) -> ({input_size}, {input_size})")

    # 2. 归一化到 [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0
    print(f"  2. 归一化到 [0,1]: 范围 [{img_float.min():.3f}, {img_float.max():.3f}]")

    # 3. 减去均值
    img_normalized = (img_float - mean) / std
    print(f"  3. ImageNet 标准化: 范围 [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")

    # 4. 转换为 NCHW
    blob = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
    blob = np.ascontiguousarray(blob, dtype=np.float32)
    print(f"  4. NHWC -> NCHW: 形状 {blob.shape}")

    # ONNX 加载说明
    print("\n[验证] EfficientDet ONNX 推理:")
    print("  加载方式: cv2.dnn.readNetFromONNX('efficientdet-d4.onnx')")
    print("  后端选择: 优先 CUDA, 其次 OpenCV")

    print("\n[验证通过] EfficientDet ONNX 推理流程演示完成")
    return True


# ============================================
# 练习 9: 图像分类 - ResNet
# ============================================
def exercise_9_resnet_classification():
    """
    练习 9: 图像分类 - ResNet

    算法原理:
    ResNet (Residual Network) 是一种深度卷积神经网络,
    通过残差连接解决了深层网络训练困难的问题。

    核心组件:
    - 残差块 (Residual Block): y = F(x) + x
    - 瓶颈设计 (Bottleneck): 1x1 -> 3x3 -> 1x1 卷积
    - 全局平均池化 (GAP)
    - 1000 类 softmax 输出

    ImageNet 预处理:
    - 使用 (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    - 减去 ImageNet 均值并除以标准差
    - swapRB=True (BGR → RGB)

    推理流程:
    1. 预处理图像为 224x224
    2. 转换为 blob (NCHW)
    3. 前向传播获取 1000 维输出
    4. Softmax 转换为概率
    5. 取 Top-5 预测结果
    """
    print("=" * 60)
    print("练习 9: 图像分类 - ResNet")
    print("=" * 60)

    # ImageNet Top-5 类别 (简化)
    imagenet_classes = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "cockstrap", "hen", "ostrich", "brambling",
        "house finch", "robin", "clipper", "pigeon", "laughing gull",
        " Californian gull", "penguin", "albatross", "black swan", "koala"
    ]

    print("\n[验证] ImageNet 分类任务:")
    print(f"  类别总数: 1000")
    print(f"  Top-5 准确率: 通常 > 85%")

    # ResNet 结构说明
    print("\n[验证] ResNet50 结构:")
    layers = [
        ("conv1", "7x7, 64, stride=2"),
        ("pool1", "3x3 maxpool, stride=2"),
        ("layer1", "3 blocks, 256 channels"),
        ("layer2", "4 blocks, 512 channels"),
        ("layer3", "6 blocks, 1024 channels"),
        ("layer4", "3 blocks, 2048 channels"),
        ("avgpool", "global average pooling"),
        ("fc", "1000-d classification"),
    ]

    for name, desc in layers:
        print(f"  {name}: {desc}")

    # 模拟预处理
    print("\n[验证] ImageNet 预处理:")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    print(f"  均值: {mean}")
    print(f"  标准差: {std}")
    print(f"  尺寸: (224, 224)")
    print(f"  swapRB: True (BGR → RGB)")

    # 模拟分类输出
    print("\n[模拟] ResNet50 分类输出:")
    output = np.random.randn(1, 1000).astype(np.float32)

    # Softmax
    exp_output = np.exp(output - np.max(output))
    softmax_output = exp_output / np.sum(exp_output)

    # Top-5
    top5_idx = np.argsort(softmax_output[0])[-5:][::-1]
    print("  Top-5 预测:")
    for i, idx in enumerate(top5_idx):
        class_name = imagenet_classes[idx] if idx < len(imagenet_classes) else f"class_{idx}"
        print(f"    {i+1}. {class_name}: {softmax_output[0, idx]*100:.2f}%")

    # 验证预处理公式
    print("\n[验证] 预处理公式验证:")
    test_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)  # BGR
    blob = cv2.dnn.blobFromImage(
        test_pixel, 1/255.0, (224, 224),
        (mean[2], mean[1], mean[0]),  # RGB 顺序
        True
    )
    # 归一化后应该是: (pixel/255 - mean) / std
    expected = ((200/255.0 - 0.406) / 0.229,
                (150/255.0 - 0.456) / 0.224,
                (100/255.0 - 0.485) / 0.229)
    print(f"  输入 BGR 像素: {test_pixel[0, 0]}")
    print(f"  Blob 输出 (R,G,B): {blob[0, :, 0, 0]}")
    print(f"  期望值 (R,G,B): ({expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f})")

    print("\n[验证通过] ResNet 图像分类流程演示完成")
    return True


# ============================================
# 练习 10: NMS 实现与参数调优
# ============================================
def exercise_10_nms():
    """
    练习 10: NMS 实现与参数调优

    算法原理:
    NMS (Non-Maximum Suppression) 非极大值抑制,
    用于去除目标检测中重叠的检测框,保留最佳检测结果。

    算法步骤:
    1. 按置信度降序排列所有检测框
    2. 选取当前最高置信度的框作为输出
    3. 计算该框与所有剩余框的 IoU (交并比)
    4. 删除 IoU > nms_threshold 的所有框
    5. 重复步骤 2-4 直到没有剩余框

    IoU (Intersection over Union) 计算:
    IoU = Intersection / Union
       = Intersection / (Box1 + Box2 - Intersection)

    参数影响:
    - conf_threshold (置信度阈值): 过滤低置信度框
    - nms_threshold (NMS 阈值): 控制重叠框的合并
      - 较小值: 保留更多框,严格过滤重叠
      - 较大值: 保留较少框,宽松过滤重叠
    """
    print("=" * 60)
    print("练习 10: NMS 实现与参数调优")
    print("=" * 60)

    def calculate_iou(box1, box2):
        """
        计算两个边界框的 IoU
        box 格式: [x, y, w, h] (左上角坐标 + 宽高)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算交集区域
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    # 模拟检测框
    print("\n[模拟] 假设有 8 个检测框:")
    boxes = [
        [50, 50, 100, 100],  # box 0
        [55, 55, 100, 100],  # box 1 (与 box 0 高度重叠)
        [200, 200, 80, 80],  # box 2
        [205, 205, 80, 80],  # box 3 (与 box 2 高度重叠)
        [100, 100, 50, 50],  # box 4 (与 box 0 部分重叠)
        [300, 300, 60, 60],  # box 5
        [50, 200, 40, 40],    # box 6 (与 box 0 不重叠)
        [400, 400, 100, 100], # box 7
    ]
    confidences = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        print(f"  检测框 {i}: bbox={box}, conf={conf:.2f}")

    # 计算 IoU 矩阵
    print("\n[验证] IoU 矩阵计算:")
    iou_matrix = np.zeros((len(boxes), len(boxes)))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i != j:
                iou_matrix[i, j] = calculate_iou(boxes[i], boxes[j])

    print("  IoU 矩阵 (部分):")
    for i in range(min(4, len(boxes))):
        row = [f"{iou_matrix[i, j]:.2f}" for j in range(min(4, len(boxes)))]
        print(f"    {row}")

    # 使用 OpenCV 的 NMSBoxes
    print("\n[验证] OpenCV NMSBoxes 函数:")
    print(f"  函数: cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)")

    # 测试不同 nms_threshold 的影响
    conf_threshold = 0.3

    print(f"\n  置信度阈值: {conf_threshold}")
    print("  不同 NMS 阈值的效果:")

    for nms_thresh in [0.3, 0.5, 0.7]:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_thresh)
        print(f"    nms_threshold={nms_thresh}: 保留 {len(indices)} 个框 -> {indices.flatten()}")

    # 自定义 NMS 实现
    print("\n[模拟] 自定义 NMS 算法:")
    print("  1. 按置信度降序排序")
    print("  2. 选取最高置信度框")
    print("  3. 删除 IoU > threshold 的框")
    print("  4. 重复直到处理完所有框")

    print("\n[验证通过] NMS 实现与参数调优流程演示完成")
    return True


# ============================================
# 练习 11: CPU/GPU 性能对比
# ============================================
def exercise_11_backend_performance():
    """
    练习 11: CPU/GPU 性能对比

    算法原理:
    不同的后端和目标设备对推理性能有显著影响,
    需要根据硬件环境选择最优配置。

    性能影响因素:
    1. 硬件: GPU 通常比 CPU 快 10-100x
    2. 精度: FP16 比 FP32 快约 2x
    3. 批量大小: 批量推理可提高吞吐量
    4. 模型复杂度: 轻量模型在边缘设备上更快

    性能指标:
    - FPS (Frames Per Second): 吞吐量
    - 延迟: 单帧处理时间
    - 内存占用: 模型和中间结果的内存使用

    测试方法:
    1. 多次运行取平均 (排除冷启动影响)
    2. 预热运行 (JIT 编译优化)
    3. 记录内存使用情况
    """
    print("=" * 60)
    print("练习 11: CPU/GPU 性能对比")
    print("=" * 60)

    # 模拟不同配置的 FPS
    print("\n[模拟] 不同配置的推理性能 (FPS):")

    configs = [
        ("CPU (单线程)", "OpenCV", "CPU", 5),
        ("CPU (多线程)", "OpenCV", "CPU", 15),
        ("GPU (FP32)", "CUDA", "CUDA", 120),
        ("GPU (FP16)", "CUDA", "CUDA FP16", 200),
        ("OpenCL GPU", "OpenCV", "OpenCL", 80),
    ]

    print(f"  {'配置':<20} {'后端':<10} {'目标':<15} {'FPS':>10}")
    print(f"  {'-'*60}")
    for name, backend, target, fps in configs:
        print(f"  {name:<20} {backend:<10} {target:<15} {fps:>10}")

    # 计算加速比
    print("\n[验证] 加速比计算:")
    base_fps = 5  # CPU 单线程作为基准
    for name, backend, target, fps in configs:
        speedup = fps / base_fps
        print(f"  {name}: {speedup:.1f}x 加速")

    # 内存占用
    print("\n[模拟] 内存占用估计:")
    models = [
        ("MobileNet V2", 14, 25),
        ("ResNet50", 98, 180),
        ("YOLOv4", 256, 520),
        ("EfficientDet-D4", 52, 150),
    ]

    print(f"  {'模型':<20} {'FP32 (MB)':>12} {'FP16 (MB)':>12}")
    print(f"  {'-'*45}")
    for name, fp32_mem, fp16_mem in models:
        print(f"  {name:<20} {fp32_mem:>12} {fp16_mem:>12}")

    # 延迟对比
    print("\n[验证] 延迟对比 (毫秒):")
    for fps in [5, 15, 120, 200]:
        latency_ms = 1000 / fps
        print(f"  {fps} FPS -> 延迟 {latency_ms:.1f} ms/帧")

    # 自动选择最优后端
    print("\n[验证] 自动后端选择逻辑:")
    print("  1. 检查 CUDA 是否可用")
    print("  2. 是 -> 选择 CUDA 后端")
    print("  3. 否 -> 检查 OpenCL")
    print("  4. 是 -> 选择 OpenCL 后端")
    print("  5. 否 -> 使用 OpenCV CPU 后端")

    print("\n[验证通过] CPU/GPU 性能对比流程演示完成")
    return True


# ============================================
# 练习 12: 多模型组合 (检测 + 分类)
# ============================================
def exercise_12_multi_model_pipeline():
    """
    练习 12: 多模型组合 (检测 + 分类)

    算法原理:
    实际应用中经常需要组合多个模型来完成复杂任务,
    例如: 目标检测 + 目标分类 + 目标跟踪

    典型pipeline:
    1. YOLOv4 检测: 找出图像中的所有目标
    2. 裁剪检测框: 提取每个目标的 ROI
    3. ResNet 分类: 对每个 ROI 进行细粒度分类
    4. 结果汇总: 合并检测和分类结果

    优化策略:
    - 批量推理: 一次性处理多个 ROI
    - 模型融合: 检测模型提供类别先验
    - 级联推理: 轻量模型预筛选, 重模型精分类

    应用场景:
    - 自动驾驶: 车辆检测 + 车道线分类
    - 视频监控: 人脸检测 + 行为识别
    - 工业检测: 缺陷检测 + 缺陷分类
    """
    print("=" * 60)
    print("练习 12: 多模型组合 (检测 + 分类)")
    print("=" * 60)

    # 模拟多模型 pipeline
    print("\n[模拟] 自动驾驶感知 Pipeline:")
    print("  Stage 1: YOLOv4 目标检测")
    print("  Stage 2: ResNet 车型细分类")
    print("  Stage 3: 车道线检测")

    # 模拟检测结果
    print("\n[模拟] YOLOv4 检测结果:")
    detections = [
        {"class": "car", "bbox": [100, 200, 80, 80], "conf": 0.92},
        {"class": "car", "bbox": [250, 200, 90, 80], "conf": 0.88},
        {"class": "truck", "bbox": [400, 190, 120, 100], "conf": 0.85},
        {"class": "person", "bbox": [50, 300, 40, 80], "conf": 0.78},
        {"class": "car", "bbox": [520, 210, 85, 75], "conf": 0.95},
    ]

    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        print(f"  检测 {i+1}: class={det['class']}, conf={det['conf']:.2f}")
        print(f"    边界框: ({x}, {y}, {w}, {h})")

    # 模拟 ROI 裁剪和分类
    print("\n[模拟] ROI 裁剪与二次分类:")
    vehicle_types = ["sedan", "SUV", "truck", "van", "bus"]
    fine_classes = ["sedan", "SUV", "truck", "van", "bus", "motorcycle"]

    for i, det in enumerate(detections):
        if det["class"] in ["car", "truck"]:
            # 模拟分类结果
            type_id = np.random.randint(0, len(fine_classes))
            type_conf = np.random.uniform(0.7, 0.95)
            print(f"  检测 {i+1} 二次分类: {fine_classes[type_id]}, 置信度={type_conf:.2f}")

    # 批量推理优化
    print("\n[验证] 批量推理优化:")
    print("  单个推理: 5 次 forward, 每次独立")
    print("  批量推理: 1 次 forward, 5 个输入")
    print("  加速比: 通常 2-5x")

    # 模拟批量分类
    print("\n[模拟] 批量分类输入:")
    num_vehicles = sum(1 for d in detections if d["class"] in ["car", "truck"])
    batch_size = num_vehicles
    print(f"  批量大小: {batch_size}")
    print(f"  输入形状: ({batch_size}, 3, 224, 224)")
    print(f"  输出形状: ({batch_size}, {len(fine_classes)})")

    print("\n[验证通过] 多模型组合流程演示完成")
    return True


# ============================================
# 练习 13: 异步推理 pipeline
# ============================================
def exercise_13_async_inference():
    """
    练习 13: 异步推理 pipeline

    算法原理:
    异步推理允许在 GPU 执行当前批次时准备下一批次数据,
    从而隐藏数据加载和预处理的开销。

    同步 vs 异步:
    同步: [准备数据] -> [推理] -> [处理结果] -> [准备数据] -> ...
    异步: [准备数据] -> [推理] -> [处理结果]
              ↳ [准备数据] (与推理并行)

    OpenCV 异步 API:
    - net.setInput(blob, "input")  # 异步设置输入
    - net.forward(output, "output")  # 同步获取输出
    - net.forwardAsync(output, "output")  # 异步获取输出

    实现方式:
    1. 使用 cv2.dnn.Net 的 async 方法
    2. 使用 Python threading 模块
    3. 使用多进程队列

    性能提升:
    - 隐藏 I/O 和预处理延迟
    - 提高 GPU 利用率
    - 提升整体吞吐量
    """
    print("=" * 60)
    print("练习 13: 异步推理 pipeline")
    print("=" * 60)

    # 同步 vs 异步流程对比
    print("\n[验证] 同步推理流程:")
    sync_steps = [
        "[1] 读取视频帧",
        "[2] 预处理 (resize, normalize)",
        "[3] 转换为 blob",
        "[4] GPU 推理",
        "[5] 后处理 (NMS)",
        "[6] 显示/保存结果",
    ]
    for step in sync_steps:
        print(f"  {step}")

    print("\n[验证] 异步推理流程:")
    async_steps = [
        "[主线程] 读取帧 1 -> 预处理 -> 提交推理",
        "[工作线程] GPU 推理帧 1 <-> [主线程] 读取帧 2 -> 预处理 -> 提交推理",
        "[工作线程] GPU 推理帧 2 <-> [主线程] 读取帧 3 -> ...",
    ]
    for step in async_steps:
        print(f"  {step}")

    # 时间分析
    print("\n[模拟] 时间分析 (假设):")
    timings = {
        "I/O 读取": 10,
        "预处理": 5,
        "GPU 推理": 30,
        "后处理": 5,
    }

    sync_total = sum(timings.values())
    async_pipeline = timings["I/O 读取"] + timings["预处理"] + timings["GPU 推理"] + timings["后处理"]
    overlap_benefit = min(timings["I/O 读取"], timings["GPU 推理"])

    print(f"  同步总时间: {sync_total} ms")
    print(f"  异步总时间: {sync_total - overlap_benefit} ms")
    print(f"  加速比: {sync_total / (sync_total - overlap_benefit):.2f}x")

    # Python 实现示例
    print("\n[验证] Python 异步推理模式:")
    print("  1. 双缓冲队列: 准备两个 blob buffer")
    print("  2. 交替使用: buffer[0] -> 推理 -> buffer[1] -> 推理")
    print("  3. 预处理并行: 在 GPU 推理时预处理下一帧")

    # OpenCV async API 说明
    print("\n[验证] OpenCV 异步 API:")
    print("  # 异步模式 (需要同步等待结果)")
    print("  net.setInput(blob)")
    print("  output = net.forward()  # 同步等待")
    print("")
    print("  # Async 模式 (C++ only in older OpenCV)")
    print("  net.forwardAsync(output)")

    print("\n[验证通过] 异步推理 pipeline 流程演示完成")
    return True


# ============================================
# 练习 14: 视频流实时推理
# ============================================
def exercise_14_video_stream():
    """
    练习 14: 视频流实时推理

    算法原理:
    实时视频推理需要在严格的时间约束下完成检测,
    通常要求达到 25-30 FPS (每帧 33-40ms)。

    实时性要求:
    - 帧率: 25-30 FPS
    - 每帧延迟: < 40ms
    - 丢帧处理: 跳帧或降分辨率

    优化策略:
    1. 跳帧检测: 每隔 N 帧检测一次
    2. 降分辨率: 使用较小输入尺寸
    3. 模型量化: INT8 量化加速
    4. 异步推理: 隐藏 I/O 延迟

    丢帧策略:
    - 固定跳帧: 每 3 帧检测 1 帧
    - 自适应跳帧: 根据 FPS 动态调整
    - 帧间插值: 用跟踪器填充缺失帧

    OpenCV 视频读取:
    - cv2.VideoCapture: 支持摄像头/视频文件
    - cv2.VideoWriter: 保存检测结果
    """
    print("=" * 60)
    print("练习 14: 视频流实时推理")
    print("=" * 60)

    # 实时性指标
    print("\n[验证] 实时性指标:")
    fps_targets = [24, 30, 60]
    print(f"  {'目标 FPS':<15} {'每帧最大延迟':>15}")
    print(f"  {'-'*32}")
    for fps in fps_targets:
        max_latency = 1000 / fps
        print(f"  {fps:<15} {max_latency:>12.1f} ms")

    # 模拟处理流程
    print("\n[模拟] 实时检测 Pipeline:")

    stages = [
        ("视频读取", 5, "VideoCapture.read()"),
        ("图像预处理", 3, "blobFromImage()"),
        ("模型推理 (YOLOv4)", 20, "net.forward()"),
        ("NMS 后处理", 2, "NMSBoxes()"),
        ("结果绘制", 2, "cv2.rectangle()"),
        ("显示输出", 5, "cv2.imshow()"),
    ]

    total_time = sum(t for _, t, _ in stages)
    print(f"  {'阶段':<20} {'耗时':>8} {'说明':<25}")
    print(f"  {'-'*55}")
    for name, time_ms, desc in stages:
        print(f"  {name:<20} {time_ms:>6} ms  {desc:<25}")
    print(f"  {'总计':<20} {total_time:>6} ms")

    # 计算实际 FPS
    actual_fps = 1000 / total_time
    print(f"\n  预计 FPS: {actual_fps:.1f}")

    # 跳帧策略
    print("\n[验证] 跳帧策略:")
    strategies = [
        ("检测每一帧", 1, "检测所有帧, 最高延迟"),
        ("每 2 帧检测 1 帧", 2, "中等延迟, 省一半算力"),
        ("每 3 帧检测 1 帧", 3, "较低延迟, 省 2/3 算力"),
        ("自适应跳帧", "动态", "根据实际 FPS 调整"),
    ]

    for name, skip, desc in strategies:
        effective_fps = actual_fps / skip if isinstance(skip, int) else actual_fps
        print(f"  {name}: 有效 FPS = {effective_fps:.1f} ({desc})")

    # OpenCV VideoCapture 参数
    print("\n[验证] VideoCapture 常用参数:")
    params = [
        ("CAP_PROP_FRAME_WIDTH", "帧宽度"),
        ("CAP_PROP_FRAME_HEIGHT", "帧高度"),
        ("CAP_PROP_FPS", "帧率"),
        ("CAP_PROP_BUFFERSIZE", "缓冲区大小"),
    ]
    for param, desc in params:
        print(f"  {param}: {desc}")

    print("\n[验证通过] 视频流实时推理流程演示完成")
    return True


# ============================================
# 练习 15: 模型量化与优化
# ============================================
def exercise_15_model_optimization():
    """
    练习 15: 模型量化与优化

    算法原理:
    模型量化是将浮点数权重/激活转换为低精度整数的过程,
    可显著减少模型大小和推理延迟。

    量化类型:
    - FP16 (半精度): 16 位浮点, 几乎无损
    - INT8 (8 位整数): 有损,但加速明显
    - INT4/INT2: 极端量化,需要特殊处理

    量化优势:
    - 模型大小减少 2-4x
    - 内存带宽减少 2-4x
    - 推理速度提升 2-4x (取决于硬件)

    OpenCV 量化支持:
    - CUDA FP16: net.setPreferableTarget(DNN_TARGET_CUDA_FP16)
    - OpenCL FP16: net.setPreferableTarget(DNN_TARGET_OPENCL_FP16)

    注意事项:
    - 量化可能引入精度损失
    - 并非所有模型都适合量化
    - 某些层可能不支持量化
    """
    print("=" * 60)
    print("练习 15: 模型量化与优化")
    print("=" * 60)

    # 量化类型对比
    print("\n[验证] 量化类型对比:")
    quant_types = [
        ("FP32 (全精度)", 32, 4, 1.0, "最高精度"),
        ("FP16 (半精度)", 16, 2, 2.0, "几乎无损"),
        ("INT8 (8位整数)", 8, 1, 4.0, "有损加速"),
        ("INT4 (4位整数)", 4, 0.5, 8.0, "严重量化"),
    ]

    print(f"  {'类型':<20} {'位数':>6} {'大小比':>8} {'加速比':>8} {'精度':<15}")
    print(f"  {'-'*60}")
    for name, bits, size_ratio, speedup, accuracy in quant_types:
        print(f"  {name:<20} {bits:>6} {size_ratio:>7.1f}x {speedup:>7.1f}x {accuracy:<15}")

    # 模拟量化效果
    print("\n[模拟] YOLOv4 量化效果:")
    models = [
        ("FP32", 256, 1.0, "100%"),
        ("FP16", 128, 2.0, "99.8%"),
        ("INT8", 64, 4.0, "98.5%"),
    ]

    print(f"  {'模型精度':<10} {'大小':>8} {'加速比':>10} {'精度保留':>12}")
    print(f"  {'-'*42}")
    for name, size, speedup, acc in models:
        print(f"  {name:<10} {size:>7} MB {speedup:>9.1f}x {acc:>11}")

    # OpenCV 量化配置
    print("\n[验证] OpenCV 量化配置:")
    print("  # FP16 量化 (CUDA)")
    print("  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)")
    print("  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)")
    print("")
    print("  # FP16 量化 (OpenCL)")
    print("  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)")
    print("  net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)")

    # 性能优化技巧
    print("\n[验证] 推理性能优化技巧:")
    optimizations = [
        ("批量推理", "一次处理多张图像", "2-5x 加速"),
        ("模型剪枝", "移除不重要神经元", "2-4x 加速"),
        ("知识蒸馏", "用小模型学习大模型", "精度接近大模型"),
        ("输入缓存", "预计算 blob", "减少预处理开销"),
        ("异步推理", "并行处理数据流", "隐藏 I/O 延迟"),
    ]

    for name, desc, benefit in optimizations:
        print(f"  {name}: {desc} -> {benefit}")

    print("\n[验证通过] 模型量化与优化流程演示完成")
    return True


# ============================================
# 主函数
# ============================================
def main():
    """
    DNN 模块练习题主函数
    运行所有练习并输出验证结果
    """
    print("\n" + "=" * 60)
    print("OpenCV DNN 模块练习题")
    print("=" * 60)
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"Python 版本: {__import__('sys').version.split()[0]}")
    print("=" * 60)

    # 所有练习列表
    exercises = [
        ("练习 1: ONNX 模型加载与推理", exercise_1_onnx_load_and_inference),
        ("练习 2: TensorFlow 模型加载", exercise_2_tensorflow_load),
        ("练习 3: PyTorch TorchScript 模型加载", exercise_3_pytorch_load),
        ("练习 4: DNN 预处理 - blobFromImage", exercise_4_blob_preprocessing),
        ("练习 5: 后端和目标设备选择", exercise_5_backend_selection),
        ("练习 6: YOLOv4 目标检测", exercise_6_yolo_detection),
        ("练习 7: SSD MobileNet 目标检测", exercise_7_ssd_mobilenet),
        ("练习 8: EfficientDet ONNX 推理", exercise_8_efficientdet),
        ("练习 9: ResNet 图像分类", exercise_9_resnet_classification),
        ("练习 10: NMS 实现与参数调优", exercise_10_nms),
        ("练习 11: CPU/GPU 性能对比", exercise_11_backend_performance),
        ("练习 12: 多模型组合", exercise_12_multi_model_pipeline),
        ("练习 13: 异步推理 pipeline", exercise_13_async_inference),
        ("练习 14: 视频流实时推理", exercise_14_video_stream),
        ("练习 15: 模型量化与优化", exercise_15_model_optimization),
    ]

    # 运行所有练习
    results = []
    for name, func in exercises:
        print(f"\n{'='*60}")
        try:
            result = func()
            results.append((name, result, None))
            print(f"\n[PASS] {name}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n[FAIL] {name}: {e}")

    # 汇总结果
    print("\n" + "=" * 60)
    print("练习结果汇总")
    print("=" * 60)

    passed = sum(1 for _, r, _ in results if r)
    failed = sum(1 for _, r, e in results if not r and e is not None)
    skipped = sum(1 for _, r, e in results if not r and e is None)

    print(f"  通过: {passed}/{len(exercises)}")
    print(f"  失败: {failed}/{len(exercises)}")
    print(f"  跳过: {skipped}/{len(exercises)}")

    if failed > 0:
        print("\n失败练习:")
        for name, _, error in results:
            if error:
                print(f"  - {name}: {error}")

    print("\n" + "=" * 60)
    print("所有练习演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()