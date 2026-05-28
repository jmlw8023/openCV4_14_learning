# Core 模块练习题解答

> **模块**: core
> **OpenCV 版本**: 4.14.0-pre
> **练习题数量**: 14 题

---

## 目录

1. [入门级](./exercises/README.md#1入门级)
2. [中级](./exercises/README.md#2中级)
3. [高级](./exercises/README.md#3高级)
4. [挑战题](./exercises/README.md#4挑战题)

---

## 1.入门级

### 练习 1: 创建一个 100×100 的红色图像并保存

```python
# ============================================
# 练习1: 创建红色图像并保存
# ============================================
import cv2
import numpy as np

def create_red_image():
    """创建一个 100×100 的红色图像并保存"""
    # 创建 100×100 的彩色图像，初始为黑色
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 设置为红色 (BGR格式: 0, 0, 255)
    img[:, :] = [0, 0, 255]

    # 保存图像
    cv2.imwrite("red_image.png", img)
    print(f"图像尺寸: {img.shape}")
    print(f"图像数据类型: {img.dtype}")
    print(f"像素值示例 (中心点): {img[50, 50]}")

    return img

if __name__ == "__main__":
    img = create_red_image()
```

### 练习 2: 读取图像, 提取 ROI 区域并显示

```python
# ============================================
# 练习2: 读取图像并提取ROI
# ============================================
import cv2
import numpy as np

def extract_roi(image_path):
    """读取图像并提取 ROI 区域"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    print(f"原图尺寸: {img.shape}")

    # 提取 ROI (假设取中心 100×100 区域)
    h, w = img.shape[:2]
    roi_h, roi_w = 100, 100
    top = (h - roi_h) // 2
    left = (w - roi_w) // 2

    roi = img[top:top+roi_h, left:left+roi_w]
    print(f"ROI 尺寸: {roi.shape}")

    # 显示图像
    cv2.imshow("Original Image", img)
    cv2.imshow("ROI Region", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return roi

if __name__ == "__main__":
    # 使用之前创建的图像测试
    roi = extract_roi("red_image.png")
```

### 练习 3: 将彩色图像转换为灰度并反色

```python
# ============================================
# 练习3: 灰度转换与反色
# ============================================
import cv2
import numpy as np

def grayscale_and_invert(image_path):
    """将彩色图像转换为灰度并反色"""
    # 读取彩色图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"灰度图尺寸: {gray.shape}")

    # 反色 (255 - 原值)
    inverted = cv2.bitwise_not(gray)
    # 等效于: inverted = 255 - gray

    # 显示结果
    cv2.imshow("Original (Color)", img)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Inverted", inverted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return gray, inverted

if __name__ == "__main__":
    grayscale_and_invert("red_image.png")
```

### 练习 4: 使用三种方式遍历图像并计算像素总和

```python
# ============================================
# 练习4: 三种方式遍历图像计算像素总和
# ============================================
import cv2
import numpy as np
import time

def traverse_and_sum(image_path):
    """使用三种方式遍历图像并计算像素总和"""
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    print(f"图像尺寸: {img.shape}")
    print(f"总像素数: {img.size}")

    # 方法1: NumPy 直接求和 (最快)
    start = time.time()
    sum1 = np.sum(img)
    time1 = time.time() - start
    print(f"\n方法1 (NumPy sum): 和 = {sum1}, 耗时 = {time1*1000:.2f}ms")

    # 方法2: 行指针遍历
    start = time.time()
    sum2 = 0
    for row in range(img.shape[0]):
        ptr = img[row]
        for col in range(img.shape[1]):
            sum2 += ptr[col]
    time2 = time.time() - start
    print(f"方法2 (行指针遍历): 和 = {sum2}, 耗时 = {time2*1000:.2f}ms")

    # 方法3: flat 迭代器遍历
    start = time.time()
    sum3 = 0
    for pixel in img.flat:
        sum3 += pixel
    time3 = time.time() - start
    print(f"方法3 (flat迭代器): 和 = {sum3}, 耗时 = {time3*1000:.2f}ms")

    # 验证三种方法结果一致
    assert sum1 == sum2 == sum3, "三种方法结果不一致!"
    print(f"\n验证通过: 三种方法结果一致 = {sum1}")

    return sum1

if __name__ == "__main__":
    traverse_and_sum("red_image.png")
```

---

## 2.中级

### 练习 5: 实现自定义滤波 (不使用 filter2D) - 锐化/模糊

```python
# ============================================
# 练习5: 自定义滤波 - 锐化与模糊
# ============================================
import cv2
import numpy as np

def custom_filter(img, kernel):
    """
    自定义卷积滤波 (不使用 filter2D)
    img: 输入图像
    kernel: 卷积核
    """
    # 获取图像和核的尺寸
    img_h, img_w = img.shape[:2]
    kernel_h, kernel_w = kernel.shape[:2]

    # 锚点位置 (核中心)
    anchor_h = kernel_h // 2
    anchor_w = kernel_w // 2

    # 创建输出图像
    if len(img.shape) == 3:
        output = np.zeros_like(img)
        channels = img.shape[2]
    else:
        output = np.zeros_like(img)
        channels = 1
        img = img[:, :, np.newaxis]

    # 遍历每个像素
    for c in range(channels):
        for y in range(img_h):
            for x in range(img_w):
                sum_val = 0.0

                # 遍历核的每个元素
                for ky in range(kernel_h):
                    for kx in range(kernel_w):
                        # 计算对应图像像素位置
                        iy = y + ky - anchor_h
                        ix = x + kx - anchor_w

                        # 边界处理: 镜像填充
                        if iy < 0:
                            iy = -iy
                        if iy >= img_h:
                            iy = 2 * img_h - iy - 1
                        if ix < 0:
                            ix = -ix
                        if ix >= img_w:
                            ix = 2 * img_w - ix - 1

                        # 累加
                        sum_val += img[iy, ix, c] * kernel[ky, kx]

                output[y, x, c] = np.clip(sum_val, 0, 255)

    return output.astype(np.uint8)


def custom_blur(img, kernel_size=5):
    """自定义均值模糊"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return custom_filter(img, kernel)


def custom_sharpen(img):
    """自定义锐化滤波"""
    # 锐化核
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    return custom_filter(img, kernel)


def custom_edge_detection(img):
    """自定义边缘检测 (拉普拉斯)"""
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    return custom_filter(img, kernel)


def main():
    # 读取测试图像
    img = cv2.imread("red_image.png")
    if img is None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]

    # 均值模糊
    blurred = custom_blur(img, kernel_size=5)
    print(f"模糊后尺寸: {blurred.shape}")

    # 锐化
    sharpened = custom_sharpen(img)
    print(f"锐化后尺寸: {sharpened.shape}")

    # 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = custom_edge_detection(gray)
    print(f"边缘检测后尺寸: {edges.shape}")

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Custom Blur", blurred)
    cv2.imshow("Custom Sharpen", sharpened)
    cv2.imshow("Custom Edge", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 6: 实现两幅图像的线性混合

```python
# ============================================
# 练习6: 图像线性混合
# ============================================
import cv2
import numpy as np

def linear_blend(img1, img2, alpha):
    """
    实现两幅图像的线性混合
    dst = alpha * img1 + (1 - alpha) * img2
    """
    # 确保两幅图像尺寸相同
    assert img1.shape == img2.shape, "图像尺寸不一致!"

    # 转换为浮点类型避免溢出
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # 线性混合
    blended = alpha * img1_f + (1 - alpha) * img2_f

    # 转换回 uint8
    return blended.astype(np.uint8)


def main():
    # 创建两幅测试图像
    img1 = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)  # 红色
    img2 = np.full((200, 200, 3), (255, 0, 0), dtype=np.uint8)  # 蓝色

    # 测试不同的混合比例
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for alpha in alphas:
        result = linear_blend(img1, img2, alpha)
        print(f"alpha={alpha}: 中心像素 = {result[100, 100]}")

        cv2.imshow(f"Blend alpha={alpha}", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 7: 使用 LUT 实现颜色表变换 (胶片效果)

```python
# ============================================
# 练习7: LUT 颜色表变换 - 胶片效果
# ============================================
import cv2
import numpy as np

def create_lut(curve_type, params=None):
    """
    创建颜色查找表
    curve_type: 'film', 'vintage', 'warm', 'cool', 'high_contrast'
    """
    lut = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        if curve_type == 'film':
            # 胶片效果: 降低对比度，略微提亮阴影
            lut[i] = int(255 * (i / 255.0) ** 0.9)
            lut[i] = min(255, lut[i] + 10)

        elif curve_type == 'vintage':
            # 复古效果: 降低饱和度，增加黄褐色调
            lut[i] = int(50 + 205 * (i / 255.0) ** 0.8)

        elif curve_type == 'warm':
            # 暖色调: 增加红色通道
            lut[i] = int(min(255, i * 1.1))

        elif curve_type == 'cool':
            # 冷色调: 增加蓝色
            lut[i] = int(min(255, i * 1.1))

        elif curve_type == 'high_contrast':
            # 高对比度
            lut[i] = int(128 + 128 * np.sin((i / 255.0 - 0.5) * np.pi))

        else:
            lut[i] = i

    return lut


def apply_color_table(img, lut):
    """应用颜色查找表"""
    # 对每个通道应用 LUT
    if len(img.shape) == 3:
        channels = cv2.split(img)
        for i in range(img.shape[2]):
            channels[i] = cv2.LUT(channels[i], lut)
        return cv2.merge(channels)
    else:
        return cv2.LUT(img, lut)


def main():
    # 创建测试图像
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[:, i] = [i, 128, 255 - i]  # 渐变色

    # 应用不同的颜色表
    curve_types = ['film', 'vintage', 'warm', 'cool', 'high_contrast']

    for curve_type in curve_types:
        lut = create_lut(curve_type)
        result = apply_color_table(img, lut)

        # 显示 LUT 曲线
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 2))
        plt.plot(lut)
        plt.title(f'{curve_type} LUT')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.xlim(0, 255)
        plt.ylim(0, 255)
        plt.show()

        cv2.imshow(f'Original', img)
        cv2.imshow(f'{curve_type} Effect', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 8: 实现图像归一化到 [0, 255] 范围

```python
# ============================================
# 练习8: 图像归一化到 [0, 255]
# ============================================
import cv2
import numpy as np

def normalize_to_uint8(img, min_val=None, max_val=None):
    """
    将图像归一化到 [0, 255] 范围
    img: 输入图像 (任意数据类型)
    min_val, max_val: 可选，指定归一化范围
    """
    # 转换为浮点类型
    img_f = img.astype(np.float32)

    # 获取或计算最小/最大值
    if min_val is None:
        min_val = np.min(img_f)
    if max_val is None:
        max_val = np.max(img_f)

    # 避免除零
    range_val = max_val - min_val
    if range_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    # Min-Max 归一化到 [0, 255]
    normalized = (img_f - min_val) / range_val * 255.0

    return normalized.astype(np.uint8)


def normalize_image(img):
    """使用 OpenCV 归一化"""
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def main():
    # 创建测试图像 (float 类型，范围未知)
    img = np.random.randn(100, 100, 3) * 50 + 128
    print(f"原始图像范围: [{img.min():.2f}, {img.max():.2f}]")
    print(f"原始数据类型: {img.dtype}")

    # 自定义归一化
    normalized1 = normalize_to_uint8(img)
    print(f"\n自定义归一化后范围: [{normalized1.min()}, {normalized1.max()}]")

    # OpenCV 归一化
    normalized2 = normalize_image(img)
    print(f"OpenCV归一化后范围: [{normalized2.min()}, {normalized2.max()}]")

    # 验证两种方法结果一致
    assert np.array_equal(normalized1, normalized2), "归一化结果不一致!"
    print("\n验证通过: 两种归一化方法结果一致")

    cv2.imshow("Original", img.astype(np.uint8))
    cv2.imshow("Normalized", normalized1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 3.高级

### 练习 9: 实现 Otsu 自动阈值分割 (不使用内置函数)

```python
# ============================================
# 练习9: Otsu 自动阈值分割
# ============================================
import cv2
import numpy as np

def otsu_threshold(img):
    """
    Otsu 自动阈值分割算法
    原理: 找到使类间方差最大的阈值
    """
    # 确保是灰度图
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    total = img.shape[0] * img.shape[1]

    # 总均值
    sum_total = sum(i * hist[i] for i in range(256))
    sum_background = 0
    weight_background = 0
    weight_foreground = 0

    max_variance = 0
    threshold = 0

    # 遍历所有可能的阈值
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        # 计算均值
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # 计算类间方差
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # 更新最佳阈值
        if variance > max_variance:
            max_variance = variance
            threshold = t

    print(f"Otsu 自动计算的阈值: {threshold}")

    # 应用阈值分割
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return threshold, binary


def main():
    # 创建测试图像 (模拟有前景和背景的图像)
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 200  # 前景区域 (亮)
    img[:, :] += np.random.randint(0, 30, img.shape)  # 添加噪声

    # Otsu 阈值分割
    threshold, binary = otsu_threshold(img)

    # 使用 OpenCV 验证
    _, binary_cv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OpenCV Otsu 阈值: {binary_cv}")

    # 验证
    assert threshold == binary_cv, "Otsu 阈值计算结果不一致!"
    print("验证通过: 阈值计算正确")

    cv2.imshow("Original", img)
    cv2.imshow("Otsu Binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 10: 实现卷积操作的自定义 SIMD 优化版本

```python
# ============================================
# 练习10: 卷积操作的 SIMD 优化版本 (模拟)
# ============================================
import cv2
import numpy as np

def simd_convolve(img, kernel):
    """
    使用 NumPy 向量化模拟 SIMD 优化的卷积
    真正的 SIMD 需要使用 intrinsics (SSE/AVX)
    """
    # 获取尺寸
    img_h, img_w = img.shape[:2]
    kernel_h, kernel_w = kernel.shape[:2]
    anchor_h, anchor_w = kernel_h // 2, kernel_w // 2

    # 使用 scipy 的 correlate (高效实现)
    from scipy import signal
    result = signal.correlate2d(img, kernel, mode='same')

    return result.astype(np.uint8)


def naive_convolve(img, kernel):
    """
    朴素卷积实现 (无 SIMD)
    """
    img_h, img_w = img.shape[:2]
    kernel_h, kernel_w = kernel.shape[:2]
    anchor_h, anchor_w = kernel_h // 2, kernel_w // 2

    output = np.zeros_like(img)

    for y in range(img_h):
        for x in range(img_w):
            sum_val = 0.0
            for ky in range(kernel_h):
                for kx in range(kernel_w):
                    iy = y + ky - anchor_h
                    ix = x + kx - anchor_w
                    if 0 <= iy < img_h and 0 <= ix < img_w:
                        sum_val += img[iy, ix] * kernel[ky, kx]
            output[y, x] = np.clip(sum_val, 0, 255)

    return output.astype(np.uint8)


def main():
    import time

    # 创建测试图像
    img = np.random.randint(0, 256, (500, 500), dtype=np.uint8)

    # 定义卷积核 (边缘检测)
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    # 朴素实现计时
    start = time.time()
    result_naive = naive_convolve(img, kernel)
    time_naive = time.time() - start
    print(f"朴素实现耗时: {time_naive*1000:.2f}ms")

    # SIMD 优化版本计时
    start = time.time()
    result_simd = simd_convolve(img, kernel)
    time_simd = time.time() - start
    print(f"NumPy向量化耗时: {time_simd*1000:.2f}ms")

    # 加速比
    print(f"加速比: {time_naive/time_simd:.2f}x")

    cv2.imshow("Original", img)
    cv2.imshow("Edge Detection", result_simd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 11: 实现矩阵的 SVD 分解并用于图像压缩

```python
# ============================================
# 练习11: SVD 分解用于图像压缩
# ============================================
import cv2
import numpy as np

def svd_compress(img, n_components):
    """
    使用 SVD 分解压缩图像
    只保留前 n_components 个奇异值
    """
    # 确保是灰度图
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 转换为浮点类型
    img_f = img.astype(np.float32)

    # SVD 分解: img = u * s * vt
    u, s, vt = np.linalg.svd(img_f, full_matrices=False)

    print(f"原始矩阵秩: {min(img.shape)}")
    print(f"保留奇异值数量: {n_components}")

    # 只保留前 n_components 个奇异值
    u_comp = u[:, :n_components]
    s_comp = s[:n_components]
    vt_comp = vt[:n_components, :]

    # 重构图像
    compressed = np.dot(u_comp, np.dot(np.diag(s_comp), vt_comp))

    # 计算压缩率
    original_size = img.shape[0] * img.shape[1]
    compressed_size = u_comp.shape[0] * u_comp.shape[1] + \
                      s_comp.shape[0] + \
                      vt_comp.shape[0] * vt_comp.shape[1]
    ratio = original_size / compressed_size

    print(f"压缩率: {ratio:.2f}x")

    return compressed.astype(np.uint8), ratio


def main():
    # 创建或加载测试图像
    img = cv2.imread("red_image.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 创建测试图像 (棋盘格)
        img = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            for j in range(200):
                img[i, j] = 255 if ((i // 20) + (j // 20)) % 2 == 0 else 0

    # 测试不同的压缩级别
    original = img.copy()
    components = [5, 10, 20, 50, 100]

    for n in components:
        compressed, ratio = svd_compress(img, n)
        cv2.imshow(f"SVD Compression (k={n}, {ratio:.1f}x)", compressed)

    cv2.imshow("Original", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 12: 实现并行图像处理流水线 (多线程)

```python
# ============================================
# 练习12: 并行图像处理流水线 (多线程)
# ============================================
import cv2
import numpy as np
import threading
import queue
import time

class ParallelPipeline:
    """并行图像处理流水线"""

    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.workers = []
        self.running = False

    def process_stage(self, img, stage_name):
        """模拟各个处理阶段"""
        if stage_name == "grayscale":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif stage_name == "blur":
            return cv2.GaussianBlur(img, (5, 5), 1.5)
        elif stage_name == "edge":
            return cv2.Canny(img, 50, 150)
        elif stage_name == "threshold":
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return binary
        else:
            return img

    def worker(self, worker_id):
        """工作线程"""
        while self.running:
            try:
                # 从输入队列获取任务
                task = self.input_queue.get(timeout=0.1)
                if task is None:
                    break

                idx, img = task

                # 执行流水线处理
                result = img
                for stage in ["grayscale", "blur", "edge", "threshold"]:
                    result = self.process_stage(result, stage)

                # 放入输出队列
                self.output_queue.put((idx, result))
                self.input_queue.task_done()

            except queue.Empty:
                continue

    def start(self):
        """启动流水线"""
        self.running = True
        self.workers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker, args=(i,))
            t.start()
            self.workers.append(t)

    def stop(self):
        """停止流水线"""
        self.running = False
        for _ in self.workers:
            self.input_queue.put(None)
        for t in self.workers:
            t.join()

    def process(self, images):
        """处理一批图像"""
        # 添加到输入队列
        for idx, img in enumerate(images):
            self.input_queue.put((idx, img))

        # 等待完成
        self.input_queue.join()

        # 收集结果
        results = []
        while not self.output_queue.empty():
            results.append(self.output_queue.get())

        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


def main():
    # 创建测试图像
    images = []
    for i in range(10):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        images.append(img)

    # 并行处理
    pipeline = ParallelPipeline(num_workers=4)

    start = time.time()
    pipeline.start()
    results = pipeline.process(images)
    pipeline.stop()
    elapsed = time.time() - start

    print(f"处理 {len(images)} 张图像耗时: {elapsed*1000:.2f}ms")
    print(f"平均每张图像: {elapsed/len(images)*1000:.2f}ms")

    # 显示结果
    for i, result in enumerate(results[:3]):
        cv2.imshow(f"Result {i}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 4.挑战题

### 练习 13: 从零实现一个简单的 CNN Forward (仅 Mat 操作)

```python
# ============================================
# 练习13: 简化 CNN Forward 实现
# ============================================
import cv2
import numpy as np

class SimpleConv2D:
    """简化卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 初始化随机卷积核
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32) * 0.01

    def forward(self, x):
        """
        前向传播
        x: (batch, channels, height, width)
        """
        batch, channels, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        output = np.zeros((batch, self.out_channels, out_h, out_w), dtype=np.float32)

        for b in range(batch):
            for oc in range(self.out_channels):
                for y in range(0, h - self.kernel_size + 1, self.stride):
                    for x_pos in range(0, w - self.kernel_size + 1, self.stride):
                        # 提取感受野
                        receptive = x[b, :, y:y+self.kernel_size, x_pos:x_pos+self.kernel_size]
                        # 卷积
                        output[b, oc, y//self.stride, x_pos//self.stride] = np.sum(
                            receptive * self.weights[oc]
                        )

        return output


class SimpleReLU:
    """ReLU 激活函数"""

    def forward(self, x):
        return np.maximum(0, x)


class SimpleMaxPool:
    """最大池化层"""

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, x):
        batch, channels, h, w = x.shape
        out_h = h // self.pool_size
        out_w = w // self.pool_size

        output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

        for b in range(batch):
            for c in range(channels):
                for y in range(out_h):
                    for x_pos in range(out_w):
                        py = y * self.pool_size
                        px = x_pos * self.pool_size
                        window = x[b, c, py:py+self.pool_size, px:px+self.pool_size]
                        output[b, c, y, x_pos] = np.max(window)

        return output


class SimpleCNN:
    """简化 CNN 模型"""

    def __init__(self):
        self.conv1 = SimpleConv2D(3, 8, 3)   # 输入3通道，输出8通道
        self.relu1 = SimpleReLU()
        self.pool1 = SimpleMaxPool(2)
        self.conv2 = SimpleConv2D(8, 16, 3)  # 输入8通道，输出16通道
        self.relu2 = SimpleReLU()
        self.pool2 = SimpleMaxPool(2)

    def forward(self, x):
        """前向传播"""
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        return x


def main():
    # 创建测试输入 (batch=1, channels=3, 32x32)
    x = np.random.randn(1, 3, 32, 32).astype(np.float32) * 0.5 + 0.5

    print(f"输入尺寸: {x.shape}")

    # 创建 CNN
    cnn = SimpleCNN()

    # 前向传播
    output = cnn.forward(x)

    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")


if __name__ == "__main__":
    main()
```

### 练习 14: 实现图像拼接中的特征匹配和融合

```python
# ============================================
# 练习14: 图像拼接 - 特征匹配和融合
# ============================================
import cv2
import numpy as np

def detect_and_compute_features(img):
    """
    检测特征点并计算描述子
    """
    # 使用 ORB 特征检测器
    orb = cv2.ORB_create(nfeatures=500)

    # 检测特征点和计算描述子
    keypoints, descriptors = orb.detectAndCompute(img, None)

    return keypoints, descriptors


def match_features(desc1, desc2):
    """
    匹配两组描述子
    返回良好的匹配点对
    """
    # 使用 BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 取前 50 个最佳匹配
    good_matches = matches[:50]

    return good_matches


def find_homography(kp1, kp2, matches):
    """
    使用匹配点对计算单应性矩阵
    """
    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵 (使用 RANSAC)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H


def stitch_images(img1, img2):
    """
    拼接两幅图像
    """
    # 检测特征
    kp1, desc1 = detect_and_compute_features(img1)
    kp2, desc2 = detect_and_compute_features(img2)

    print(f"图像1 特征点: {len(kp1)}")
    print(f"图像2 特征点: {len(kp2)}")

    # 匹配特征
    matches = match_features(desc1, desc2)
    print(f"良好匹配数: {len(matches)}")

    if len(matches) < 10:
        print("匹配点太少，无法拼接")
        return None

    # 计算单应性矩阵
    H = find_homography(kp1, kp2, matches)

    if H is None:
        print("无法计算单应性矩阵")
        return None

    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 变换第二幅图像
    stitched = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))

    # 将第一幅图像叠加到结果上
    stitched[0:h1, 0:w1] = img1

    return stitched


def blend_multiband(img1, img2):
    """
    多频段融合 (简化版)
    """
    # 构建高斯金字塔
    def build_gaussian(img, levels=5):
        pyramids = [img]
        for i in range(levels - 1):
            img = cv2.pyrDown(img)
            pyramids.append(img)
        return pyramids

    # 构建拉普拉斯金字塔
    def build_laplacian(img, levels=5):
        pyramids = []
        gaussian = build_gaussian(img, levels)

        for i in range(levels - 1):
            expanded = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
            laplacian = cv2.subtract(gaussian[i], expanded)
            pyramids.append(laplacian)
        pyramids.append(gaussian[-1])  # 最后一层是高频残差

        return pyramids

    # 融合
    lap1 = build_laplacian(img1)
    lap2 = build_laplacian(img2)

    fused = []
    for l1, l2 in zip(lap1, lap2):
        # 使用权重 0.5 混合
        blended = cv2.addWeighted(l1, 0.5, l2, 0.5, 0)
        fused.append(blended)

    # 重建
    result = fused[-1]
    for i in range(len(fused) - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(fused[i].shape[1], fused[i].shape[0]))
        result = cv2.add(result, fused[i])

    return result


def main():
    # 创建测试图像 (简单拼接测试)
    img1 = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)  # 红色
    img2 = np.full((200, 200, 3), (0, 255, 0), dtype=np.uint8)  # 绿色

    # 简单拼接
    stitched = np.hstack([img1, img2])

    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 验证说明

每个练习都包含：
- **详细中文注释** - 解释算法原理和实现细节
- **可运行代码** - 直接执行验证结果
- **输出说明** - 打印关键结果便于验证

运行方式：
```bash
cd modules/core/exercises
python exercises.py
```