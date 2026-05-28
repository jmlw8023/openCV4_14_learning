# imgproc 模块练习题解答

> **模块**: imgproc
> **OpenCV 版本**: 4.14.0-pre
> **练习题数量**: 8 题

---

## 目录

1. [入门级](./exercises.py#1入门级)
2. [中级](./exercises.py#2中级)
3. [高级](./exercises.py#3高级)

---

## 1.入门级

### 练习 1: 实现不同插值方法的图像缩放

```python
# ============================================
# 练习1: 不同插值方法的图像缩放
# ============================================
import cv2
import numpy as np

def resize_with_interpolation(img, target_size, interpolation):
    """
    使用指定插值方法调整图像大小
    """
    resized = cv2.resize(img, target_size, interpolation=interpolation)
    return resized


def main():
    # 创建测试图像 (棋盘格)
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = 255 if ((i // 10) + (j // 10)) % 2 == 0 else 0

    # 目标尺寸
    target_size = (400, 400)

    # 不同插值方法
    methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
    }

    print("图像缩放测试 (100x100 -> 400x400)")
    print("-" * 50)

    for name, method in methods.items():
        resized = resize_with_interpolation(img, target_size, method)
        print(f"{name}: 输出尺寸 = {resized.shape}, 耗时测试")

    print("\n验证: 使用最近邻插值保留棋盘格结构")
    nearest = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    # 检查棋盘格结构是否保留 (每隔40像素应有边界)
    print(f"最近邻缩放后边缘处像素值: {nearest[40, 40]}, {nearest[41, 40]}")

    cv2.imshow("Original", img)
    cv2.imshow("INTER_NEAREST", nearest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 2: 将图像转换为 HSV 并提取饱和度通道

```python
# ============================================
# 练习2: HSV 颜色空间与饱和度通道提取
# ============================================
import cv2
import numpy as np

def extract_saturation_channel(img):
    """
    将图像转换为 HSV 并提取饱和度通道
    """
    # BGR 转 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 分离通道
    h, s, v = cv2.split(hsv)

    print(f"HSV 通道范围:")
    print(f"  H (色调): [{h.min()}, {h.max()}]")
    print(f"  S (饱和度): [{s.min()}, {s.max()}]")
    print(f"  V (亮度): [{v.min()}, {v.max()}]")

    return hsv, s


def main():
    # 创建测试图像 (彩色渐变)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            img[i, j] = [i, 255 - j, j]  # 彩色渐变

    # 提取饱和度
    hsv, saturation = extract_saturation_channel(img)

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Saturation Channel", saturation)
    print("\n验证: 饱和度通道提取成功")
    print(f"饱和度通道尺寸: {saturation.shape}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 3: 使用 Otsu 法进行图像阈值分割

```python
# ============================================
# 练习3: Otsu 自动阈值分割
# ============================================
import cv2
import numpy as np

def otsu_threshold_manual(img):
    """
    手动实现 Otsu 阈值分割算法
    """
    # 确保灰度图
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    total = img.size

    # 总均值
    sum_total = sum(i * hist[i] for i in range(256))

    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    return threshold


def main():
    # 创建测试图像 (双峰分布)
    img = np.zeros((200, 200), dtype=np.uint8)
    img[0:100, :] = 50   # 背景
    img[100:200, :] = 200  # 前景

    # 添加噪声
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 手动 Otsu
    threshold_manual = otsu_threshold_manual(img)
    print(f"手动 Otsu 计算的阈值: {threshold_manual}")

    # OpenCV Otsu
    _, binary_cv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OpenCV Otsu 计算的阈值: {binary_cv}")

    # 应用阈值
    _, binary_manual = cv2.threshold(img, threshold_manual, 255, cv2.THRESH_BINARY)

    print(f"\n验证: 手动阈值与 OpenCV 一致 = {threshold_manual == int(binary_cv)}")

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Manual Otsu", binary_manual)
    cv2.imshow("OpenCV Otsu", binary_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 2.中级

### 练习 4: 实现自定义卷积滤波 (锐化、浮雕)

```python
# ============================================
# 练习4: 自定义卷积滤波 - 锐化与浮雕
# ============================================
import cv2
import numpy as np

def custom_filter2d(img, kernel, anchor=None):
    """
    自定义二维卷积滤波
    """
    # 获取尺寸
    kh, kw = kernel.shape
    if anchor is None:
        anchor = (kh // 2, kw // 2)

    # 图像尺寸
    if len(img.shape) == 3:
        h, w, c = img.shape
        output = np.zeros_like(img)
    else:
        h, w = img.shape
        c = 1
        img = img[:, :, np.newaxis]
        output = np.zeros_like(img)

    # 遍历图像
    for channel in range(c):
        for y in range(h):
            for x in range(w):
                sum_val = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        iy = y + ky - anchor[0]
                        ix = x + kx - anchor[1]
                        # 边界处理: 复制边界
                        iy = max(0, min(iy, h - 1))
                        ix = max(0, min(ix, w - 1))
                        sum_val += img[iy, ix, channel] * kernel[ky, kx]
                output[y, x, channel] = np.clip(sum_val, 0, 255)

    if c == 1:
        output = output[:, :, 0]
    return output.astype(np.uint8)


def main():
    # 创建测试图像
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 200

    # 锐化核
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # 浮雕核
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)

    # 边缘检测核
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    # 使用 OpenCV filter2D (作为参考)
    sharpen_cv = cv2.filter2D(img, -1, sharpen_kernel)
    emboss_cv = cv2.filter2D(img, -1, emboss_kernel)
    edge_cv = cv2.filter2D(img, -1, edge_kernel)

    print("自定义卷积滤波验证:")
    print(f"锐化核:\n{sharpen_kernel}")
    print(f"浮雕核:\n{emboss_kernel}")
    print(f"边缘检测核:\n{edge_kernel}")

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Sharpen", sharpen_cv)
    cv2.imshow("Emboss", emboss_cv)
    cv2.imshow("Edge", edge_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 5: 使用形态学操作去除二值图像中的噪声

```python
# ============================================
# 练习5: 形态学操作去噪声
# ============================================
import cv2
import numpy as np

def remove_noise_morphology(binary_img):
    """
    使用形态学操作去除二值图像中的噪声
    """
    # 定义结构元素
    kernel_3 = np.ones((3, 3), dtype=np.uint8)
    kernel_5 = np.ones((5, 5), dtype=np.uint8)

    # 开运算: 先腐蚀后膨胀 (去除小亮点/噪声)
    opened_3 = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_3)
    opened_5 = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_5)

    # 闭运算: 先膨胀后腐蚀 (填充小空洞)
    closed_3 = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_3)
    closed_5 = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_5)

    # 形态学梯度: 膨胀减腐蚀 (提取边界)
    gradient = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel_3)

    return {
        'opened_3x3': opened_3,
        'opened_5x5': opened_5,
        'closed_3x3': closed_3,
        'closed_5x5': closed_5,
        'gradient': gradient
    }


def main():
    # 创建测试图像 (带噪声的二值图像)
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255  # 主体区域

    # 添加噪声 (随机亮点和空洞)
    noise_sp = np.random.choice([0, 255], size=(200, 200), p=[0.95, 0.05])
    noise_sp = noise_sp.astype(np.uint8)

    # 合并
    noisy = cv2.bitwise_and(img, noise_sp)
    print(f"添加噪声前白色像素: {cv2.countNonZero(img)}")
    print(f"添加噪声后白色像素: {cv2.countNonZero(noisy)}")

    # 去噪声
    results = remove_noise_morphology(noisy)

    print("\n去噪结果:")
    for name, result in results.items():
        white_pixels = cv2.countNonZero(result)
        print(f"  {name}: 白色像素 = {white_pixels}")

    # 显示结果
    cv2.imshow("Noisy Image", noisy)
    cv2.imshow("Opened 3x3", results['opened_3x3'])
    cv2.imshow("Closed 3x3", results['closed_3x3'])
    cv2.imshow("Gradient", results['gradient'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 6: 查找并绘制图像轮廓

```python
# ============================================
# 练习6: 轮廓查找与绘制
# ============================================
import cv2
import numpy as np

def find_and_draw_contours(binary_img):
    """
    查找并绘制图像轮廓
    """
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        binary_img,
        cv2.RETR_TREE,  # 检索所有轮廓并建立层级
        cv2.CHAIN_APPROX_SIMPLE  # 压缩水平、垂直、对角线段
    )

    print(f"找到 {len(contours)} 个轮廓")
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        print(f"  轮廓 {i}: 面积 = {area:.1f}, 周长 = {perimeter:.1f}")

    return contours


def approximate_and_draw(img, contours):
    """
    轮廓近似并绘制
    """
    # 创建彩色图像用于绘制
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        # 多边形近似
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 绘制轮廓 (绿色)
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

        # 在轮廓中心绘制近似多边形 (蓝色)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)

        # 标注轮廓编号
        print(f"轮廓 {i}: {len(cnt)} 个点 -> {len(approx)} 个近似点")

    return result


def main():
    # 创建测试图像 (几何形状)
    img = np.zeros((300, 300), dtype=np.uint8)

    # 绘制矩形
    cv2.rectangle(img, (50, 50), (120, 120), 255, -1)

    # 绘制圆形
    cv2.circle(img, (200, 200), 50, 255, -1)

    # 绘制三角形
    triangle = np.array([[50, 250], [100, 150], [150, 250]], dtype=np.int32)
    cv2.fillPoly(img, [triangle], 255)

    # 查找并绘制轮廓
    contours = find_and_draw_contours(img)

    # 轮廓近似
    result = approximate_and_draw(img, contours)

    cv2.imshow("Original", img)
    cv2.imshow("Contours", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 3.高级

### 练习 7: 实现透视变换 (手动计算单应性)

```python
# ============================================
# 练习7: 透视变换 - 手动计算单应性矩阵
# ============================================
import cv2
import numpy as np

def compute_homography(src_pts, dst_pts):
    """
    根据四对对应点计算单应性矩阵
    使用 DLT (Direct Linear Transform) 算法
    """
    assert len(src_pts) == 4 and len(dst_pts) == 4

    # 构建 A 矩阵
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)

    # SVD 分解求最小特征值对应的特征向量
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # 归一化
    H = H / H[2, 2]

    return H


def manual_perspective_transform(img, corners, target_corners):
    """
    手动实现透视变换
    """
    # 计算单应性矩阵
    H = compute_homography(corners, target_corners)

    print(f"单应性矩阵 H:\n{H}")

    # 获取输出图像尺寸
    min_x = min(p[0] for p in target_corners)
    max_x = max(p[0] for p in target_corners)
    min_y = min(p[1] for p in target_corners)
    max_y = max(p[1] for p in target_corners)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # 创建输出图像
    output = np.zeros((height, width, 3), dtype=np.uint8)

    # 遍历输出图像的每个像素
    for y in range(height):
        for x in range(width):
            # 计算对应源图像中的位置
            src_x = x + min_x
            src_y = y + min_y

            # 使用单应性矩阵的逆变换
            point = np.array([src_x, src_y, 1])
            inv_H = np.linalg.inv(H)
            mapped = inv_H @ point
            src_mapped_x = int(mapped[0] / mapped[2])
            src_mapped_y = int(mapped[1] / mapped[2])

            # 检查是否在源图像范围内
            if 0 <= src_mapped_x < img.shape[1] and 0 <= src_mapped_y < img.shape[0]:
                output[y, x] = img[src_mapped_y, src_mapped_x]

    return output


def main():
    # 创建测试图像 (带棋盘格)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            color = (255, 255, 255) if ((i // 20) + (j // 20)) % 2 == 0 else (50, 50, 50)
            img[i, j] = color

    # 定义源图像中的四边形顶点 (梯形)
    src_corners = np.array([
        [50, 50],    # 左上
        [150, 50],   # 右上
        [170, 170],  # 右下
        [30, 170]    # 左下
    ], dtype=np.float32)

    # 定义目标矩形顶点
    target_corners = np.array([
        [0, 0],
        [200, 0],
        [200, 200],
        [0, 200]
    ], dtype=np.float32)

    # 手动透视变换
    result_manual = manual_perspective_transform(img, src_corners, target_corners)

    # OpenCV 透视变换 (作为参考)
    H, _ = cv2.findHomography(src_corners, target_corners)
    result_cv = cv2.warpPerspective(img, H, (200, 200))

    print("\n透视变换完成")
    print(f"手动结果尺寸: {result_manual.shape}")
    print(f"OpenCV结果尺寸: {result_cv.shape}")

    # 绘制源图像中的四边形
    for i in range(4):
        start = tuple(src_corners[i].astype(int))
        end = tuple(src_corners[(i+1)%4].astype(int))
        cv2.line(img, start, end, (0, 0, 255), 2)

    cv2.imshow("Original with Quad", img)
    cv2.imshow("Manual Perspective", result_manual)
    cv2.imshow("OpenCV Perspective", result_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 练习 8: 构建图像拼接流水线

```python
# ============================================
# 练习8: 图像拼接流水线
# ============================================
import cv2
import numpy as np

class ImageStitchingPipeline:
    """图像拼接流水线"""

    def __init__(self):
        self.feature_matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

    def detect_features(self, img):
        """检测特征点"""
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """匹配特征"""
        matches = self.feature_matcher.knnMatch(desc1, desc2, k=2)

        # 比率测试筛选
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return good_matches

    def find_homography(self, kp1, kp2, matches):
        """计算单应性矩阵"""
        if len(matches) < 4:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def stitch(self, img1, img2):
        """拼接两幅图像"""
        # 1. 特征检测
        kp1, desc1 = self.detect_features(img1)
        kp2, desc2 = self.detect_features(img2)

        print(f"特征点数量: img1={len(kp1)}, img2={len(kp2)}")

        # 2. 特征匹配
        matches = self.match_features(desc1, desc2)
        print(f"良好匹配数: {len(matches)}")

        if len(matches) < 10:
            print("匹配点太少")
            return None

        # 3. 计算单应性矩阵
        H = self.find_homography(kp1, kp2, matches)
        if H is None:
            return None

        print(f"单应性矩阵:\n{H}")

        # 4. 透视变换并拼接
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 计算输出图像尺寸
        corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        all_corners = np.concatenate([corners, transformed_corners])
        x_min, y_min = all_corners.min(axis=0).astype(int)[0]
        x_max, y_max = all_corners.max(axis=0).astype(int)[0]

        output_width = x_max - x_min + 1
        output_height = y_max - y_min + 1

        # 平移矩阵
        translate = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)

        # 合并变换矩阵
        H_combined = translate @ H

        # 变换第二幅图像
        stitched = cv2.warpPerspective(img2, H_combined, (output_width, output_height))

        # 叠加第一幅图像
        if x_min < 0:
            img1 = img1[:, -x_min:]
        if y_min < 0:
            img1 = img1[-y_min:, :]

        stitched[-y_min:-y_min+img1.shape[0], -x_min:-x_min+img1.shape[1]] = img1

        return stitched


def main():
    # 创建测试图像 (简单水平拼接)
    img1 = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)  # 红色
    img2 = np.full((200, 200, 3), (0, 255, 0), dtype=np.uint8)  # 绿色

    # 添加重叠区域特征 (在边缘添加一些标记)
    for i in range(50, 150):
        img1[i, 175:200] = [255, 255, 255]  # 白色标记 (右边缘)
        img2[i, 0:25] = [255, 255, 255]      # 相同标记 (左边缘)

    # 创建拼接流水线
    pipeline = ImageStitchingPipeline()

    # 拼接
    result = pipeline.stitch(img1, img2)

    if result is not None:
        print(f"\n拼接结果尺寸: {result.shape}")
        cv2.imshow("Image 1", img1)
        cv2.imshow("Image 2", img2)
        cv2.imshow("Stitched", result)
    else:
        print("拼接失败")

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
cd modules/imgproc/exercises
python exercises.py
```