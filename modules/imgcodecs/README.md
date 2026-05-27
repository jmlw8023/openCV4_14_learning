# OpenCV imgcodecs Module Guide

> **Module**: imgcodecs
> **OpenCV Version**: 4.14.0-pre
> **Stage**: 1 (Getting Started)

---

## Table of Contents

1. [Overview](./README.md#1-overview)
2. [Supported Formats](./README.md#2-supported-formats)
3. [Core APIs](./README.md#3-core-apis)
4. [Implementation Analysis](./README.md#4-implementation-analysis)
5. [Code Examples](./README.md#5-code-examples)
6. [Exercises](./README.md#6-exercises)
7. [References](./README.md#7-references)

---

## 1. Overview

**imgcodecs** provides image decoding and encoding capabilities:

| Feature | Description |
|---------|-------------|
| **Image Loading** | Read images from files in various formats |
| **Image Saving** | Write images to files in various formats |
| **Format Detection** | Automatic format detection based on file content |
| **Parameter Control** | Fine-tune loading/saving parameters |

**Header**: `opencv2/imgcodecs.hpp`

---

## 2. Supported Formats

### Built-in Image Formats

| Format | Extension | Codec | Support |
|--------|-----------|-------|---------|
| JPEG | .jpg, .jpeg | libjpeg | ✅ |
| PNG | .png | libpng | ✅ |
| TIFF | .tif, .tiff | libtiff | ✅ |
| BMP | .bmp | Built-in | ✅ |
| WebP | .webp | libwebp | ✅ |
| OpenEXR | .exr | OpenEXR | ✅ |
| HDR | .hdr | OpenEXR | ✅ |
| PPM/PGM | .ppm, .pgm | Built-in | ✅ |
| SunRaster | .sr, .ras | Built-in | ✅ |

### Load Flags (imread)

| Flag | Description |
|------|-------------|
| `IMREAD_UNCHANGED` | Load as-is (with alpha channel) |
| `IMREAD_GRAYSCALE` | Convert to grayscale |
| `IMREAD_COLOR` | Convert to 3-channel BGR |
| `IMREAD_ANYDEPTH` | Preserve bit depth |
| `IMREAD_ANYCOLOR` | Load as any color format |
| `IMREAD_LOAD_GDAL` | Use GDAL driver |

---

## 3. Core APIs

### 3.1 Image Loading

```cpp
#include <opencv2/imgcodecs.hpp>

// Basic loading
Mat img = imread("photo.jpg");  // Uses IMREAD_COLOR default

// With explicit flag
Mat img = imread("photo.png", IMREAD_GRAYSCALE);

// Check if loaded
if (img.empty()) {
    // Handle error
}

// Check image properties
img.rows;        // Height
img.cols;        // Width
img.channels();  // Channels (1 for grayscale, 3 for color)
img.depth();     // Bit depth
```

### 3.2 Image Saving

```cpp
#include <opencv2/imgcodecs.hpp>

// Basic saving (format determined by extension)
imwrite("output.jpg", img);
imwrite("output.png", img);

// With parameters
vector<int> params = {
    IMWRITE_JPEG_QUALITY, 95,      // JPEG quality 0-100
    IMWRITE_PNG_COMPRESSION, 9    // PNG compression 0-9
};
imwrite("output.jpg", img, params);

// PNG with alpha
imwrite("output.png", img, {IMWRITE_PNG_COMPRESSION, 9});
```

### 3.3 Image Parameter Constants

```cpp
// JPEG parameters
IMWRITE_JPEG_QUALITY       // 1-100 (default 95)
IMWRITE_JPEG_PROGRESSIVE   // Enable progressive JPEG (0/1)
IMWRITE_JPEG_OPTIMIZE      // Enable optimization (0/1)
IMWRITE_JPEG_RST_INTERVAL  // Restart marker interval
IMWRITE_JPEG_LUMA_QUALITY  // Separate luma quality (0-100)

// PNG parameters
IMWRITE_PNG_COMPRESSION    // 0-9 (default 3)
IMWRITE_PNG_STRATEGY        // Strategy: DEFAULT, FILTERED, Huffman, RLE
IMWRITE_PNG_BILEVEL         // Bilevel output (0/1)

// TIFF parameters
IMWRITE_TIFF_COMPRESSION   // Compression scheme
IMWRITE_TIFF_PREDICTOR      // Predictor (0 for none, 1 for horizontal)
```

---

## 4. Implementation Analysis

### 4.1 Image Loading Flow

```cpp
// modules/imgcodecs/src/loadsave.cpp
Mat imread(const String& filename, int flags) {
    // 1. Create image decoder based on file extension
    ImageDecoder decoder = findDecoder(filename);

    // 2. Open file and read header
    decoder->load(filename);

    // 3. Set reading parameters from flags
    decoder->setScale(1);  // Or from flags

    // 4. Read image data
    Mat result;
    decoder->read(result);

    // 5. Apply color conversion if needed
    if (flags == IMREAD_GRAYSCALE)
        cvtColor(result, result, COLOR_BGR2GRAY);
    else if (flags == IMREAD_UNCHANGED)
        ;  // Keep as-is

    return result;
}
```

### 4.2 Format Detection Priority

```
1. Check file extension first (fast path)
2. If GDAL enabled, try GDAL
3. Check magic bytes (JPEG: FF D8, PNG: 89 50 4E 47...)
4. Try each codec in sequence
```

---

## 5. Code Examples

### 5.1 Basic Load and Save

```cpp
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // Load image
    Mat img = imread("photo.jpg", IMREAD_COLOR);
    if (img.empty()) {
        printf("Error: Cannot load image\n");
        return -1;
    }

    // Save in different formats
    imwrite("output.png", img);
    imwrite("output.jpg", img, {IMWRITE_JPEG_QUALITY, 90});
    imwrite("output_gray.png", img, {IMWRITE_PNG_COMPRESSION, 6});

    // Display
    imshow("Image", img);
    waitKey(0);

    return 0;
}
```

### 5.2 Batch Image Processing

```cpp
void processImages(const vector<string>& inputFiles, const string& outputDir) {
    for (const auto& inputFile : inputFiles) {
        Mat img = imread(inputFile, IMREAD_COLOR);
        if (img.empty()) continue;

        // Process
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // Save with quality setting
        string outputFile = outputDir + "/" + getFilename(inputFile);
        imwrite(outputFile, gray, {IMWRITE_JPEG_QUALITY, 85});
    }
}
```

### 5.3 Load with Specific Bit Depth

```cpp
// Load 16-bit grayscale PNG
Mat img16 = imread("16bit.png", IMREAD_ANYDEPTH | IMREAD_GRAYSCALE);
if (img16.depth() == CV_16U) {
    // Normalize to 8-bit for display
    Mat normalized;
    img16.convertTo(normalized, CV_8U, 255.0/65535.0);
    imshow("Normalized", normalized);
}

// Load 16-bit color PNG
Mat img16Color = imread("16bit_color.png", IMREAD_ANYDEPTH | IMREAD_COLOR);
```

---

## 6. Exercises

### Basic Level
1. Load an image, convert to grayscale, and save
2. Load a PNG with transparency and preserve alpha channel
3. Save an image with different JPEG quality levels (50, 75, 95)

### Intermediate Level
4. Implement a batch converter for multiple images
5. Load a 16-bit image, normalize, and display
6. Detect image format from magic bytes without extension

### Advanced Level
7. Implement WebP loading with specific quality settings
8. Create a custom image loader plugin

---

## 7. References

| Resource | Link |
|----------|------|
| Official Documentation | [imgcodecs module](https://docs.opencv.org/4.14.0/d4/da4/group__imgcodecs.html) |
| imread Reference | [imread](https://docs.opencv.org/4.14.0/d4/da4/group__imgcodecs.html#ga288baf1028cd81d463b89f1b1919efba) |
| imwrite Reference | [imwrite](https://docs.opencv.org/4.14.0/d4/da4/group__imgcodecs.html#ga49961cbfd440e1657c5fc4bd3f172550) |
| Image File Codecs | [OpenCV imgcodecs](https://docs.opencv.org/4.14.0/d2/de6/group__imgcodecs.html) |

---

## Update History

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | Initial imgcodecs module documentation |