# OpenCV Core Module Guide

> **Module**: core
> **OpenCV Version**: 4.14.0-pre
> **Stage**: 1 (Getting Started)

---

## Table of Contents

1. [Overview](./README.md#1-overview)
2. [Key Data Structures](./README.md#2-key-data-structures)
3. [Core APIs](./README.md#3-core-apis)
4. [Implementation Analysis](./README.md#4-implementation-analysis)
5. [Code Examples](./README.md#5-code-examples)
6. [Exercises](./README.md#6-exercises)
7. [References](./README.md#7-references)

---

## 1. Overview

**core** is the foundational module that all other OpenCV modules depend on. It provides:

| Feature | Description |
|---------|-------------|
| **Mat Data Structure** | Multi-dimensional dense array for images and matrices |
| **Basic Array Operations** | Arithmetic, logical, and statistical operations |
| **XML/YAML Persistence** | Configuration storage and retrieval |
| **Parallel Computing** | Multi-threading support |
| **Hardware Acceleration Layer** | SIMD, OpenCL, CUDA support |

**Header**: `opencv2/core.hpp`

---

## 2. Key Data Structures

### 2.1 Mat - Multi-dimensional Dense Array

```cpp
#include <opencv2/core.hpp>

// Create image
Mat img(480, 640, CV_8UC3);              // 480 rows x 640 cols, 3 channels uint8_t
Mat img2(Size(640, 480), CV_8UC3);       // Same effect using Size

// Load from file
Mat img = imread("photo.jpg", IMREAD_COLOR);

// Basic properties
img.rows;           // Number of rows (height)
img.cols;           // Number of columns (width)
img.channels();      // Number of channels (3 for BGR)
img.depth();         // Depth identifier (CV_8U = 0)
img.type();          // Type identifier (CV_8UC3 = 16)
img.step;            // Bytes per row
img.data;           // Raw data pointer (uchar*)
img.isContinuous(); // Whether memory is contiguous
img.total();        // Total number of elements
img.elemSize();     // Bytes per element
```

### 2.2 Mat Memory Model

```
Memory Layout (3x4 BGR image):

Row-major contiguous storage:
data[0]  data[1]  data[2]  data[3]  ...  B G R B G R B G R ...
            ↓
Step = cols × channels × elemSize() = 640 × 3 × 1 = 1920 bytes

ROI Extraction (no data copy, shared memory):
Mat roi = img(Rect(100, 100, 200, 200));  // Shares data pointer
```

### 2.3 Mat Creation Methods Comparison

```cpp
// Method 1: Constructor with initialization
Mat img1(480, 640, CV_8UC3, Scalar(0, 0, 255));  // Red background

// Method 2: create (lazy allocation)
Mat img2;
img2.create(480, 640, CV_8UC3);

// Method 3: Factory methods
Mat img3 = Mat::zeros(480, 640, CV_8UC3);
Mat img4 = Mat::ones(Size(640, 480), CV_32FC1);
Mat img5 = Mat::eye(3, 3, CV_32FC1);

// Method 4: Comma initializer
Mat img6 = (Mat_<float>(2, 2) << 1, 2, 3, 4);

// Method 5: Template class Mat_
Mat_<uchar> img7(480, 640);
img7(100, 100) = 255;  // Uses () instead of at<>
```

### 2.4 Input/Output Array Types

OpenCV uses `InputArray`/`OutputArray` for flexible function parameters:

```cpp
// Supported types: Mat, Matx, vector, UMat, etc.
void process(InputArray src, OutputArray dst);

// Usage
Mat m;
vector<Point2f> vec;
Matx33f matx;
UMat um;

process(m, result);    // Mat
process(vec, result); // std::vector
process(matx, result); // Matx
process(um, result);  // UMat (GPU)
```

### 2.5 Data Type Depth

| Depth ID | Type | Range |
|----------|------|-------|
| CV_8U | uint8_t | 0-255 |
| CV_8S | int8_t | -128 to 127 |
| CV_16U | uint16_t | 0-65535 |
| CV_16S | int16_t | -32768 to 32767 |
| CV_32S | int32_t | -2147483648 to 2147483647 |
| CV_32F | float | -1.0 to 1.0 (normalized) |
| CV_64F | double | Full range |
| CV_16F | half float | FP16 |

**Type Formula**: `CV_<depth>C<channels>`

```cpp
CV_8UC3  = 0 + 8×3  = 16   // 3-channel uint8_t
CV_32FC1 = 5 + 8×1  = 13   // 1-channel float
```

---

## 3. Core APIs

### 3.1 Arithmetic Operations

```cpp
// Addition: dst = src1 + src2 (saturate)
Mat dst = src1 + src2;
add(src1, src2, dst);
add(src1, src2, dst, mask);  // With mask

// Weighted addition (blend): dst = α·src1 + β·src2 + γ
addWeighted(src1, 0.5, src2, 0.5, 0, dst);

// Subtraction
subtract(src1, src2, dst);

// Element-wise multiplication
multiply(src1, src2, dst, scale=1);

// Matrix dot product
Mat dotResult = src1.t() * src2;

// Comparison
compare(src1, src2, dst, CMP_EQ);  // dst = (src1 == src2)
```

### 3.2 Logical Operations

```cpp
bitwise_and(src1, src2, dst);    // Bitwise AND
bitwise_or(src1, src2, dst);     // Bitwise OR
bitwise_xor(src1, src2, dst);    // Bitwise XOR
bitwise_not(src, dst);           // Bitwise NOT
inRange(src, lower, upper, dst); // src ∈ [lower, upper]
```

### 3.3 Statistical Operations

```cpp
Scalar sum = sum(src);              // Sum of all elements
Scalar mean = mean(src, mask);     // Mean value
meanStdDev(src, mean, stddev, mask); // Mean and standard deviation

double minVal, maxVal;
Point minLoc, maxLoc;
minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask);

int nz = countNonZero(src);        // Count non-zero elements
double normL2 = norm(src, NORM_L2); // L2 norm
```

### 3.4 Array Transformations

```cpp
// Split/Merge channels
vector<Mat> channels;
split(src, channels);   // Split into single channels
merge(channels, dst);   // Merge

// Channel mixing
Mat bgr, alpha;
int fromTo[] = {0,2, 1,1, 2,0, 3,3};  // BGR → GRB + alpha
mixChannels(&src, 1, &bgr, 1, fromTo, 4);

// Flip
flip(src, dst, 0);   // 0: vertical, >0: horizontal, <0: both

// Transpose
transpose(src, dst);

// Rotate 90°
rotate(src, dst, ROTATE_90_CLOCKWISE);

// Concatenate
hconcat(src1, src2, dst);  // Horizontal
vconcat(src1, src2, dst);  // Vertical

// Repeat/Replication
repeat(src, 2, 3, dst);  // Replicate 2×3 times
```

### 3.5 Type Conversion

```cpp
// Convert data type
src.convertTo(dst, CV_32FC1, alpha=1, beta=0);

// Saturated uchar conversion
convertScaleAbs(src, dst);

// Normalize
normalize(src, dst, 1.0, 0, NORM_L2);  // L2 normalization
```

### 3.6 Look-Up Table (LUT)

```cpp
// Create LUT
Mat lut(256, 1, CV_8U);
for (int i = 0; i < 256; i++)
    lut.at<uchar>(i) = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255);

// Apply LUT
LUT(src, lut, dst);
```

---

## 4. Implementation Analysis

### 4.1 Row Continuity

```cpp
// Check: isContinuous()
Mat img(480, 640, CV_8UC3);

// Memory layout:
// Contiguous: data has img.rows × img.step bytes
// Non-contiguous: extra row padding exists

// Fast traversal (when row-continuous):
uchar* p = img.data;
for (size_t i = 0; i < img.total() * img.channels(); i++) {
    p[i] = processing(p[i]);
}
```

### 4.2 ROI (Region of Interest)

```cpp
Mat img = imread("photo.jpg");

// Extract ROI (no data copy, shared memory)
Rect roi(100, 100, 200, 200);
Mat region = img(roi);

// Modifying region affects img
region = Scalar(0, 0, 0);  // Top-left corner becomes black

// Copy ROI when needed
Mat regionCopy = img(roi).clone();
```

### 4.3 Scalar Type

```cpp
// 4-element vector for multi-channel pixel values
Scalar s(255);           // Single channel: (255)
Scalar s(255, 128, 64);  // 3 channels: B=255, G=128, R=64

// Usage in BGR image
img = Scalar(0, 0, 255);  // Red
```

### 4.4 saturate_cast - Saturating Cast

```cpp
// Prevents overflow
uchar val = saturate_cast<uchar>(300.0);  // Returns 255
uchar val = saturate_cast<uchar>(-10.0);   // Returns 0

// All arithmetic operations use saturation
Mat src1(1, 1, CV_8U, Scalar(250));
Mat src2(1, 1, CV_8U, Scalar(10));
Mat dst;
add(src1, src2, dst);  // dst = 255 (not 260)
```

### 4.5 Source Code Reference

| File | Description |
|------|-------------|
| `modules/core/src/mat.cpp` | Mat implementation |
| `modules/core/src/arithm.cpp` | Arithmetic operations |
| `modules/core/src/convert.cpp` | Type conversion |
| `modules/core/src/copy.cpp` | Copy and ROI operations |

---

## 5. Code Examples

### 5.1 Basic Image Operations

```cpp
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // Read image
    Mat img = imread("photo.jpg", IMREAD_COLOR);

    // Convert to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Create inverted image
    Mat inverted;
    bitwise_not(img, inverted);

    // Adjust brightness and contrast
    Mat adjusted;
    img.convertTo(adjusted, -1, 1.5, 50);  // alpha=1.5, beta=50

    // Display
    imshow("Original", img);
    imshow("Gray", gray);
    imshow("Inverted", inverted);
    waitKey(0);

    return 0;
}
```

### 5.2 Pixel Access Methods

```cpp
Mat img(480, 640, CV_8UC3);

// Method 1: at<> (type-safe, compile-time checked)
img.at<Vec3b>(y, x)[0] = 255;  // B channel

// Method 2: ptr<> (pointer access, row start)
uchar* row = img.ptr<uchar>(y);
row[x * 3] = 255;              // B
row[x * 3 + 1] = 0;           // G

// Method 3: data + offset (fastest)
uchar* p = img.data + y * img.step + x * img.elemSize();
p[0] = 255;                   // B

// Method 4: MatIterator_
for (auto it = img.begin<Vec3b>(); it != img.end<Vec3b>(); ++it) {
    (*it)[0] = 255;
}
```

### 5.3 Batch Image Processing

```cpp
void batchProcess(const vector<string>& filenames) {
    vector<Mat> images;
    for (const auto& filename : filenames) {
        Mat img = imread(filename);
        if (img.empty()) continue;

        Mat processed;
        cvtColor(img, processed, COLOR_BGR2GRAY);
        normalize(processed, processed, 0, 255, NORM_MINMAX);
        images.push_back(processed);
    }

    // Calculate average image
    Mat avgImg = Mat::zeros(images[0].size(), CV_32FC1);
    for (auto& img : images) {
        Mat floatImg;
        img.convertTo(floatImg, CV_32FC1);
        avgImg += floatImg;
    }
    avgImg /= images.size();
}
```

### 5.4 Gamma Correction using LUT

```cpp
Mat gammaCorrect(const Mat& src, double gamma) {
    Mat lut(256, 1, CV_8U);
    for (int i = 0; i < 256; i++)
        lut.at<uchar>(i) = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255);

    Mat dst;
    LUT(src, lut, dst);
    return dst;
}
```

---

## 6. Exercises

### Basic Level
1. Create a 100×100 red image and save it
2. Read an image, extract ROI, and display it
3. Convert a color image to grayscale and invert it

### Intermediate Level
4. Implement a custom filter (without filter2D)
5. Implement image blending (linear blend two images)
6. Implement gamma correction using LUT

### Advanced Level
7. Implement Otsu's automatic thresholding (without built-in)
8. Optimize convolution with SIMD intrinsics

---

## 7. References

| Resource | Link |
|----------|------|
| Official Documentation | [docs.opencv.org/4.14.0/](https://docs.opencv.org/4.14.0/) |
| Mat Class Reference | [Class Mat](https://docs.opencv.org/4.14.0/d3/d63/classcv_1_1Mat.html) |
| Core Module Tutorial | [Core Operations](https://docs.opencv.org/4.14.0/d7/d16/tutorial_py_core_ops.html) |
| Source Code | [github.com/opencv/opencv](https://github.com/opencv/opencv) |

---

## Update History

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | Initial core module documentation |