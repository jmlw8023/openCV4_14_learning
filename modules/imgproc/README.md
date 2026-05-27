# OpenCV imgproc Module Guide

> **Module**: imgproc
> **OpenCV Version**: 4.14.0-pre
> **Stage**: 2 (Fundamentals)

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

**imgproc** (Image Processing) provides comprehensive image transformation functions:

| Feature | Description |
|---------|-------------|
| **Geometric Transformations** | Resize, rotate, warp, perspective |
| **Filtering** | blur, Gaussian, median, bilateral |
| **Morphology** | erode, dilate, open, close, gradient |
| **Color Spaces** | BGR, HSV, Lab, YCrCb conversion |
| **Histogram** | Equalization, backprojection |
| **Contours** | Finding, approximation, moments |
| **Image Gradients** | Sobel, Laplacian, Canny |
| **Thresholding** | Binary, Otsu, adaptive |

**Header**: `opencv2/imgproc.hpp`

---

## 2. Key Data Structures

### 2.1 Point, Size, Rect

```cpp
#include <opencv2/imgproc.hpp>

// Point
Point2f pt(100.f, 200.f);
Point pt2(100, 200);

// Size
Size sz(640, 480);
Size2f szf(640.5f, 480.5f);

// Rect (x, y, width, height)
Rect roi(100, 100, 200, 150);
roi.x; roi.y; roi.width; roi.height;
roi.area();        // width * height
roi.contains(pt); // Check if point inside
```

### 2.2 RotatedRect

```cpp
RotatedRect rr(Point2f(center.x, center.y), Size2f(w, h), angle);

// Access members
rr.center;    // Center point
rr.size;      // Width and height
rr.angle;     // Rotation angle in degrees

// Bounding box
Rect2f bbox = rr.boundingRect();
```

---

## 3. Core APIs

### 3.1 Resize and Geometric Transformations

```cpp
// Resize
Mat dst;
resize(src, dst, Size(w, h));                    // Explicit size
resize(src, dst, Size(), 0.5, 0.5, INTER_LINEAR); // Scale

// Rotate 90°
rotate(src, dst, ROTATE_90_CLOCKWISE);

// Warp affine
Mat M = getRotationMatrix2D(center, angle, scale);
warpAffine(src, dst, M, Size(w, h));

// Perspective transform
Mat H = findHomography(srcPoints, dstPoints);
warpPerspective(src, dst, H, Size(w, h));
```

### 3.2 Filtering

```cpp
// Blur (average)
blur(src, dst, Size(5, 5));

// Gaussian blur
GaussianBlur(src, dst, Size(5, 5), sigmaX);

// Median blur
medianBlur(src, dst, 5);

// Bilateral filter (edge-preserving)
bilateralFilter(src, dst, 9, 75, 75);

// Custom filter
Mat kernel = (Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
filter2D(src, dst, -1, kernel);
```

### 3.3 Morphology

```cpp
Mat dst;

// Erode (shrink)
erode(src, dst, getStructuringElement(MORPH_RECT, Size(3,3)));

// Dilate (expand)
dilate(src, dst, getStructuringElement(MORPH_RECT, Size(3,3)));

// Opening (erode then dilate)
morphologyEx(src, dst, MORPH_OPEN, kernel);

// Closing (dilate then erode)
morphologyEx(src, dst, MORPH_CLOSE, kernel);

// Gradient (outer - inner)
morphologyEx(src, dst, MORPH_GRADIENT, kernel);

// Black hat and top hat
morphologyEx(src, dst, MORPH_BLACKHAT, kernel);
morphologyEx(src, dst, MORPH_TOPHAT, kernel);
```

### 3.4 Color Space Conversion

```cpp
// BGR to Grayscale
cvtColor(src, dst, COLOR_BGR2GRAY);

// BGR to HSV
cvtColor(src, dst, COLOR_BGR2HSV);

// BGR to Lab
cvtColor(src, dst, COLOR_BGR2Lab);

// Split channels
vector<Mat> bgr;
split(src, bgr);

// Merge channels
merge(bgr, dst);
```

### 3.5 Thresholding

```cpp
double thresh = 128;
double maxVal = 255;

// Simple threshold
threshold(src, dst, thresh, maxVal, THRESH_BINARY);

// Otsu's method (auto)
threshold(src, dst, 0, maxVal, THRESH_BINARY | THRESH_OTSU);

// Adaptive threshold
adaptiveThreshold(src, dst, maxVal, ADAPTIVE_THRESH_GAUSSIAN_C,
                   THRESH_BINARY, 11, 2);
```

### 3.6 Edge Detection

```cpp
// Sobel
Sobel(src, dst, CV_8U, 1, 0, 3);  // dx=1, dy=0, ksize=3

// Laplacian (2nd derivative)
Laplacian(src, dst, CV_8U, 3);

// Canny edge detector
Canny(src, dst, 50, 150);  // low, high thresholds
```

---

## 4. Implementation Analysis

### 4.1 Gaussian Blur Algorithm

```
1. Create 1D Gaussian kernel
2. Separable convolution:
   - Convolve rows with 1D kernel
   - Convolve result with transposed kernel
3. Complexity: O(w*h*ksize) vs O(w*h*ksize²) for 2D
```

### 4.2 Canny Edge Detection Steps

```
1. Gaussian smoothing
2. Gradient calculation (Sobel)
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis
```

---

## 5. Code Examples

### 5.1 Basic Image Processing Pipeline

```cpp
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    Mat img = imread("photo.jpg");
    Mat gray, blurred, edges;

    // Convert to grayscale
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Apply Gaussian blur
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);

    // Detect edges
    Canny(blurred, edges, 50, 150);

    imshow("Original", img);
    imshow("Edges", edges);
    waitKey(0);

    return 0;
}
```

### 5.2 Histogram Equalization

```cpp
void equalizeImage(const Mat& src) {
    Mat gray, eq;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, eq);

    imshow("Original", gray);
    imshow("Equalized", eq);
    waitKey(0);
}
```

### 5.3 Morphological Operations

```cpp
void processWithMorphology(const Mat& src) {
    Mat binary, opened, closed;

    // Threshold to binary
    threshold(src, binary, 127, 255, THRESH_BINARY);

    // Create structuring element
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    // Opening (removes noise)
    morphologyEx(binary, opened, MORPH_OPEN, kernel);

    // Closing (fills holes)
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);

    imshow("Original", src);
    imshow("Opened", opened);
    imshow("Closed", closed);
    waitKey(0);
}
```

---

## 6. Exercises

### Basic Level
1. Implement image resize with different interpolation methods
2. Convert image to HSV and extract the Saturation channel
3. Apply Otsu's thresholding to segment an image

### Intermediate Level
4. Implement a custom convolution filter (sharpen, emboss)
5. Use morphological operations to remove noise from a binary image
6. Find and draw contours on an image

### Advanced Level
7. Implement a perspective transformation with manual homography
8. Build an image stitching pipeline using imgproc functions

---

## 7. References

| Resource | Link |
|----------|------|
| Official Documentation | [imgproc module](https://docs.opencv.org/4.14.0/d7/da4/group__imgproc.html) |
| resize Reference | [resize](https://docs.opencv.org/4.14.0/d4/d86/group__imgproc__filter.html#ga4d0a3e5d0ae5e42fb6d18de5b70f6f02) |
| filter2D Reference | [filter2D](https://docs.opencv.org/4.14.0/d4/d86/group__imgproc__filter.html#gae67d2b4c1ae2ac9d3fef2c90029f3a62) |
| Canny Reference | [Canny](https://docs.opencv.org/4.14.0/dd/d1a/group__imgproc__feature.html#ga04723e100754a3e6f8858d1ac53d0db5) |

---

## Update History

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | Initial imgproc module documentation |