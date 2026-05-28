# OpenCV ml 模块学习指南

> **模块**: ml
> **OpenCV 版本**: 4.14.0-pre
> **阶段**: Stage 5 - 专家

---

## 目录

1. [概述](./README.md#1概述)
2. [机器学习基础](./README.md#2机器学习基础)
3. [模型训练与预测](./README.md#3模型训练与预测)
4. [分类器](./README.md#4分类器)
5. [回归模型](./README.md#5回归模型)
6. [聚类](./README.md#6聚类)
7. [代码示例](./README.md#7代码示例)
8. [练习题](./README.md#8练习题)
9. [参考资料](./README.md#9参考资料)

---

## 1.概述

**ml** (Machine Learning) 模块提供传统机器学习算法:

| 功能 | 描述 |
|---------|-------------|
| **分类** | SVM, 决策树, 随机森林, AdaBoost |
| **回归** | 线性回归, 神经网络 |
| **聚类** | K-Means, EM |
| **神经网络** | ANN, DNN 简版 |

**头文件**: `opencv2/ml.hpp`

---

## 2.机器学习基础

### 2.1 数据格式

```cpp
// ============================================
// C++ 训练数据格式
// ============================================
// 训练数据: rows × cols (样本数 × 特征数)
// 标签数据: rows × 1 (每行一个标签)

// 准备训练数据 (4个样本, 2个特征)
Mat trainData = (Mat_<float>(4, 2) <<
    1, 1,
    1, 2,
    2, 1,
    2, 2);

// 准备标签 (2分类)
Mat labels = (Mat_<int>(4, 1) <<
    0,  // 样本1标签
    0,  // 样本2标签
    1,  // 样本3标签
    1); // 样本4标签
```

```python
# ============================================
# Python 训练数据格式
# ============================================
import numpy as np

# 训练数据 (4个样本, 2个特征)
train_data = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2]
], dtype=np.float32)

# 标签 (2分类)
labels = np.array([0, 0, 1, 1], dtype=np.int32)
```

### 2.2 模型训练流程

```cpp
// ============================================
// C++ 训练流程
// ============================================
// 1. 创建模型
Ptr<SVM> model = SVM::create();

// 2. 设置参数
model->setType(SVM::C_SVC);           // C-SVC
model->setKernel(SVM::RBF);            // RBF 核
model->setC(1.0);                      // 正则化参数
model->setGamma(0.5);                  // RBF 参数

// 3. 训练
model->train(trainData, ROW_SAMPLE, labels);

// 4. 保存
model->save("svm_model.xml");

// 5. 预测
Mat testSample = (Mat_<float>(1, 2) << 1.5, 1.5);
float response = model->predict(testSample);
cout << "预测结果: " << response << endl;
```

```python
# ============================================
# Python 训练流程
# ============================================
import cv2
import numpy as np

# 1. 创建模型
model = cv2.ml.SVM_create()

# 2. 设置参数
model.setType(cv2.ml.SVM_C_SVC)
model.setKernel(cv2.ml.SVM_RBF)
model.setC(1.0)
model.setGamma(0.5)

# 3. 训练
model.train(train_data, cv2.ml.ROW_SAMPLE, labels)

# 4. 保存
model.save("svm_model.xml")

# 5. 预测
test_sample = np.array([[1.5, 1.5]], dtype=np.float32)
_, response = model.predict(test_sample)
print(f"预测结果: {response}")
```

---

## 3.模型训练与预测

### 3.1 SVM 支持向量机

```cpp
// ============================================
// C++ SVM
// ============================================
#include <opencv2/ml.hpp>

// 创建 SVM
Ptr<SVM> svm = SVM::create();

// 参数设置
svm->setType(SVM::C_SVC);              // C-SVC (分类)
svm->setType(SVM::ONE_CLASS);          // 一类 SVM (异常检测)
svm->setKernel(SVM::LINEAR);           // 线性核
svm->setKernel(SVM::RBF);              // RBF 核 (常用)
svm->setDegree(3.0);                   // 多项式核度数
svm->setC(1.0);                        // C 参数 (惩罚系数)
svm->setGamma(0.1);                    // RBF/多项式核参数
svm->setCoef0(0.0);                   // 多项式核系数
svm->setNu(0.5);                      // Nu 参数 (nu-SVM)
svm->setP(0.1);                        // EPS 回归精度

// 自动训练
svm->trainAuto(trainData, ROW_SAMPLE, labels,
               10,  // Fold 数 (交叉验证)
               SVM::getDefaultGrid(SVM::C),
               SVM::getDefaultGrid(SVM::GAMMA));

// 训练
svm->train(trainData, ROW_SAMPLE, labels);

// 预测
float result = svm->predict(testSample);

// 获取支持向量
Mat supportVectors = svm->getSupportVectors();
```

```python
# ============================================
# Python SVM
# ============================================
import cv2
import numpy as np

# 创建 SVM
svm = cv2.ml.SVM_create()

# 参数设置
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(1.0)
svm.setGamma(0.1)

# 自动训练 (参数网格搜索)
svm.trainAuto(train_data, cv2.ml.ROW_SAMPLE, labels, 10)

# 训练
svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)

# 预测
test_sample = np.array([[1.5, 1.5]], dtype=np.float32)
_, response = svm.predict(test_sample)
print(f"预测结果: {response}")
```

### 3.2 决策树

```cpp
// ============================================
// C++ 决策树
// ============================================
Ptr<DTrees> tree = DTrees::create();

// 参数
tree->setMaxDepth(10);                 // 最大深度
tree->setMinSampleCount(2);           // 叶节点最小样本数
tree->setCVFolds(1);                 // 交叉验证折数
tree->setUseSurrogates(false);       // 使用替代分割

// 训练和预测
tree->train(trainData, ROW_SAMPLE, labels);
float result = tree->predict(testSample);
```

```python
# ============================================
# Python 决策树
# ============================================
tree = cv2.ml.DTrees_create()

tree.setMaxDepth(10)
tree.setMinSampleCount(2)
tree.setCVFolds(1)
tree.setUseSurrogates(False)

tree.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, response = tree.predict(test_sample)
```

### 3.3 随机森林

```cpp
// ============================================
// C++ 随机森林 (RTrees)
// ============================================
Ptr<RTrees> forest = RTrees::create();

// 参数
forest->setMaxDepth(10);               // 树最大深度
forest->setMinSampleCount(2);         // 叶节点最小样本数
forest->setNestimators(100);         // 树的数量
forest->setMaxCategories(15);         // 最大类别数
forest->setVarImportanceType(GINI);  // 重要性类型

// 训练
forest->train(trainData, ROW_SAMPLE, labels);

// 预测 (返回所有树的平均结果)
float result = forest->predict(testSample);

// 获取特征重要性
Mat varImportance = forest->getVarImportance();
```

```python
# ============================================
# Python 随机森林
# ============================================
forest = cv2.ml.RTrees_create()

forest.setMaxDepth(10)
forest.setMinSampleCount(2)
forest.setNestimators(100)
forest.setMaxCategories(15)
forest.setVarImportanceType(cv2.ml.VAR_GINI)

forest.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, response = forest.predict(test_sample)
```

### 3.4 AdaBoost

```cpp
// ============================================
// C++ AdaBoost
// ============================================
Ptr<Boost> boost = Boost::create();

// 参数
boost->setBoostType(Boost::REAL);     // REAL AdaBoost
boost->setWeakCount(100);             // 弱分类器数量
boost->setWeightTrimRate(0.95);       // 权重裁剪率
boost->setMaxDepth(1);               // 弱分类器深度

// 训练和预测
boost->train(trainData, ROW_SAMPLE, labels);
float result = boost->predict(testSample);
```

```python
# ============================================
# Python AdaBoost
# ============================================
boost = cv2.ml.AdaBoost_create()

boost.setBoostType(cv2.ml.AdaBoost_REAL)
boost.setWeakCount(100)
boost.setWeightTrimRate(0.95)
boost.setMaxDepth(1)

boost.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, response = boost.predict(test_sample)
```

---

## 4.分类器对比

| 分类器 | 适用场景 | 优点 | 缺点 |
|--------|----------|------|------|
| SVM | 高维, 小样本 | 泛化能力强 | 训练慢 |
| 决策树 | 可解释性重要 | 直观, 快 | 容易过拟合 |
| 随机森林 | 通用 | 抗噪, 并行 | 占用内存大 |
| AdaBoost | 级联分类 | 精度高 | 对噪声敏感 |
| KNN | 简单任务 | 简单 | 预测慢 |

---

## 5.回归模型

### 5.1 线性回归

```cpp
// ============================================
// C++ 线性回归
// ============================================
Ptr<LinearRegression> reg = LinearRegression::create();

// 训练
reg->train(trainData, ROW_SAMPLE, labels);

// 预测
float prediction = reg->predict(testSample);

// 获取参数
Mat coeffs = reg->getLearnedCoefficients();
float intercept = reg->getIntercept();
```

```python
# ============================================
# Python 线性回归
# ============================================
reg = cv2.ml.LinearRegression_create()
reg.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, response = reg.predict(test_sample)
```

### 5.2 逻辑斯蒂回归

```cpp
// ============================================
// C++ 逻辑斯蒂回归
// ============================================
Ptr<LogisticRegression> lr = LogisticRegression::create();

// 参数
lr->setIterations(100);               // 迭代次数
lr->setLearningRate(0.1);            // 学习率
lr->setMethod(LogisticRegression::BATCH)); // BATCH 或 MINI_BATCH
lr->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));

// 训练和预测
lr->train(trainData, ROW_SAMPLE, labels);
float result = lr->predict(testSample);
```

```python
# ============================================
# Python 逻辑斯蒂回归
# ============================================
lr = cv2.ml.LogisticRegression_create()

lr.setIterations(100)
lr.setLearningRate(0.1)
lr.setMethod(cv2.ml.LogisticRegression_BATCH)
lr.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.01))

lr.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, response = lr.predict(test_sample)
```

---

## 6.聚类

### 6.1 K-Means

```cpp
// ============================================
// C++ K-Means
// ============================================
#include <opencv2/imgcodecs.hpp>

Mat data = imread("data.jpg", IMREAD_GRAYSCALE);
data = data.reshape(1, data.rows * data.cols);
data.convertTo(data, CV_32F);

// K-Means 参数
TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
int attempts = 3;
int flags = KMEANS_PP_CENTERS;  // 或 KMEANS_RANDOM_CENTERS

Mat bestLabels, centers;

// 执行 K-Means
kmeans(data, K, bestLabels, criteria, attempts, flags, centers);

// 重新整形标签到图像形状
Mat labelsImg = bestLabels.reshape(1, img.rows);
```

```python
# ============================================
# Python K-Means
# ============================================
import cv2
import numpy as np

# 准备数据
data = img.reshape(-1, img.shape[2] if len(img.shape) > 2 else 1).astype(np.float32)

# K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.1)
K = 5
ret, labels, centers = cv2.kmeans(data, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

# 重新整形
labels_img = labels.reshape(img.shape[:2])
```

### 6.2 EM (期望最大化)

```cpp
// ============================================
// C++ EM 聚类
// ============================================
Ptr<EM> em = EM::create();

// 参数
em->setClustersNumber(5);            // 高斯数量
em->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
em->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));

// 训练
em->train(trainData, noArray(), labels, means, covs, weights);

// 预测
Vec2d result = em->predict(testSample);
double probs = result[0];  // 概率
int cluster = result[1];    // 类别
```

```python
# ============================================
# Python EM 聚类
# ============================================
em = cv2.ml.EM_create()

em.setClustersNumber(5)
em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
em.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.01))

# 注意: EM 在 Python 中需要手动实现部分功能
```

---

## 7.代码示例

### 7.1 SVM 手写数字识别

```cpp
// ============================================
// C++ SVM 手写数字识别
// ============================================
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    // 加载训练数据 (MNIST 格式)
    Mat trainData, labels;
    // ... 加载数据 ...

    // 创建并训练 SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setC(10.0);
    svm->setGamma(0.0001);
    svm->train(trainData, ROW_SAMPLE, labels);

    // 测试
    Mat testImg = imread("digit.png", IMREAD_GRAYSCALE);
    resize(testImg, testImg, Size(20, 20));
    testImg = testImg.reshape(1, 1);
    testImg.convertTo(testImg, CV_32F);

    float result = svm->predict(testImg);
    cout << "识别结果: " << result << endl;

    return 0;
}
```

```python
# ============================================
# Python SVM 手写数字识别
# ============================================
import cv2
import numpy as np

def main():
    # 加载训练数据
    train_data, labels = load_mnist_data()

    # 创建并训练 SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(10.0)
    svm.setGamma(0.0001)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)

    # 测试
    test_img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img, (20, 20))
    test_img = test_img.reshape(1, -1).astype(np.float32)

    _, result = svm.predict(test_img)
    print(f"识别结果: {result}")

if __name__ == "__main__":
    main()
```

### 7.2 随机森林分类

```cpp
// ============================================
// C++ 随机森林分类
// ============================================
void randomForestClassification(const Mat& trainData, const Mat& labels) {
    // 创建随机森林
    Ptr<RTrees> forest = RTrees::create();

    // 设置参数
    forest->setMaxDepth(10);
    forest->setMinSampleCount(2);
    forest->setNestimators(100);
    forest->setVarImportanceType(RAW_VAR);

    // 训练
    forest->train(trainData, ROW_SAMPLE, labels);

    // 测试
    Mat testSamples;
    forest->predict(testSamples, noArray());

    // 保存
    forest->save("forest_model.xml");
}
```

```python
# ============================================
# Python 随机森林分类
# ============================================
def random_forest_classification(train_data, labels):
    # 创建随机森林
    forest = cv2.ml.RTrees_create()

    # 设置参数
    forest.setMaxDepth(10)
    forest.setMinSampleCount(2)
    forest.setNestimators(100)
    forest.setVarImportanceType(cv2.ml.VAR_IMPORTANCE_RAW)

    # 训练
    forest.train(train_data, cv2.ml.ROW_SAMPLE, labels)

    # 测试
    _, result = forest.predict(test_samples)

    # 保存
    forest.save("forest_model.xml")

    return result
```

### 7.3 K-Means 颜色量化

```cpp
// ============================================
// C++ K-Means 颜色量化
// ============================================
void colorQuantization(const Mat& img, int K = 16) {
    // 转换到浮点并reshape
    Mat data;
    img.convertTo(data, CV_32F);
    data = data.reshape(1, img.rows * img.cols);

    // K-Means
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    Mat labels, centers;
    kmeans(data, K, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

    // 重建图像
    centers = centers.reshape(K, img.channels());
    Mat reconstructed = centers.at(labels).reshape(img.channels(), img.rows);
    reconstructed.convertTo(reconstructed, CV_8U);

    imshow("原图", img);
    imshow("量化结果", reconstructed);
    waitKey(0);
}
```

```python
# ============================================
# Python K-Means 颜色量化
# ============================================
def color_quantization(img, K=16):
    # 转换到浮点并reshape
    data = img.reshape(-1, img.shape[2]).astype(np.float32)

    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.1)
    ret, labels, centers = cv2.kmeans(data, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # 重建图像
    centers = centers.astype(np.uint8)
    result = centers[labels.flatten()].reshape(img.shape)

    cv2.imshow("原图", img)
    cv2.imshow("量化结果", result)
    cv2.waitKey(0)

    return result
```

---

## 8.练习题

### 入门级
1. 使用 SVM 进行二分类 (C++ / Python)
2. 使用 K-Means 进行颜色量化
3. 使用随机森林进行多分类

### 中级
4. 实现交叉验证选择最佳参数
5. 实现模型持久化 (保存/加载)
6. 比较不同分类器的性能

### 高级
7. 实现手写数字识别系统
8. 实现目标检测特征分类器
9. 实现图像分割 (超像素 + K-Means)

### 挑战题
10. 实现集成学习系统 (多模型融合)
11. 实现深度学习与传统ML混合系统

---

## 9.参考资料

| 资源 | 链接 |
|----------|------|
| 官方文档 | [ml 模块](https://docs.opencv.org/4.14.0/d8/d23/group__ml.html) |
| SVM 参考 | [SVM](https://docs.opencv.org/4.14.0/d1/d2d/group__ml.html#ga821f5c70b8e16ef14fd8327ba0b620f2) |
| K-Means | [kmeans](https://docs.opencv.org/4.14.0/d5/d07/group__ml.html#ga4a2a285b6f1e0e37b25c8d4b8e3db01e) |
| OpenCV ML 教程 | [ML Tutorial](https://docs.opencv.org/4.14.0/dc/dd6/tutorial_py_ml.html) |

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | 初始 ml 模块文档 (C++/Python 双语) |