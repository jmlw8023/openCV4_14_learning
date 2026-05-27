# OpenCV highgui Module Guide

> **Module**: highgui
> **OpenCV Version**: 4.14.0-pre
> **Stage**: 1 (Getting Started)

---

## Table of Contents

1. [Overview](./README.md#1-overview)
2. [Window Management](./README.md#2-window-management)
3. [Mouse and Trackbar Events](./README.md#3-mouse-and-trackbar-events)
4. [Core APIs](./README.md#4-core-apis)
5. [Implementation Analysis](./README.md#5-implementation-analysis)
6. [Code Examples](./README.md#6-code-examples)
7. [Exercises](./README.md#7-exercises)
8. [References](./README.md#8-references)

---

## 1. Overview

**highgui** provides high-level GUI functionality:

| Feature | Description |
|---------|-------------|
| **Window Management** | Create, display, and destroy windows |
| **Image Display** | Show images in windows with auto-scaling |
| **Mouse Events** | Interactive mouse callback handling |
| **Trackbar Control** | Create sliders and progress bars |
| **Keyboard Input** | Wait for and detect key presses |

**Headers**:
- `opencv2/highgui.hpp` (C API)
- `opencv2/highgui.hpp` (WASM/JS compatible)

---

## 2. Window Management

### Window Flags

| Flag | Description |
|------|-------------|
| `WINDOW_NORMAL` | Resizable window |
| `WINDOW_FULLSCREEN` | Fullscreen window |
| `WINDOW_FREERATIO` | Auto-fit image without extending window |
| `WINDOW_KEEPRATIO` | Maintain image aspect ratio |
| `WINDOW_AUTOSIZE` | Fixed size matching image |

### Window Actions

```cpp
#include <opencv2/highgui.hpp>

// Create window (auto WINDOW_AUTOSIZE by default)
namedWindow("Display", WINDOW_NORMAL);
namedWindow("Display", WINDOW_KEEPRATIO | WINDOW_FREERATIO);

// Move and resize window
moveWindow("Display", x, y);      // Position
resizeWindow("Display", w, h);   // Size

// Destroy window
destroyWindow("Display");

// Destroy all windows
destroyAllWindows();
```

---

## 3. Mouse and Trackbar Events

### 3.1 Mouse Callback

```cpp
#include <opencv2/highgui.hpp>

// Mouse event callback
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    switch(event) {
        case EVENT_MOUSEMOVE:     // Mouse moved
            break;
        case EVENT_LBUTTONDOWN:   // Left button pressed
            break;
        case EVENT_RBUTTONDOWN:   // Right button pressed
            break;
        case EVENT_MBUTTONDOWN:   // Middle button pressed
            break;
        case EVENT_LBUTTONUP:     // Left button released
            break;
        case EVENT_LBUTTONDBLCLK:// Double click
            break;
    }
}

// Register callback
setMouseCallback("Window", mouseCallback, nullptr);
```

### 3.2 Trackbar (Slider)

```cpp
#include <opencv2/highgui.hpp>

int trackbarValue = 50;

// Create trackbar
createTrackbar("Brightness", "Window", &trackbarValue, 100,
    [](int pos, void* userdata) {
        // Callback when trackbar changes
        printf("Value: %d\n", pos);
    }, nullptr);

// Get trackbar value
int currentValue = getTrackbarPos("Brightness", "Window");

// Set trackbar value programmatically
setTrackbarPos("Brightness", "Window", 75);
```

---

## 4. Core APIs

### 4.1 Image Display

```cpp
#include <opencv2/highgui.hpp>

// Display image in window
imshow("Window", img);

// Update window (for after waitKey or when using WINDOW_FREERATIO)
waitKey(1);

// Combined with waitKey (recommended)
int key = waitKey(0);  // Wait indefinitely for keypress
int key = waitKey(30); // Wait 30ms (video frames)

// Key detection
if (key == 'q' || key == 'Q')
    break;
if (key == 27)  // ESC key
    break;
if (key == 's' || key == 'S')
    saveImage();
```

### 4.2 Display Modes

```cpp
// WINDOW_NORMAL - Allows manual resize
namedWindow("Viewer", WINDOW_NORMAL);
resizeWindow("Viewer", 800, 600);
imshow("Viewer", img);
waitKey(0);

// WINDOW_AUTOSIZE - Fixed to image size (default)
namedWindow("Viewer", WINDOW_AUTOSIZE);
imshow("Viewer", img);
waitKey(0);

// WINDOW_FREERATIO + WINDOW_KEEPRATIO - Smart resizing
namedWindow("Viewer", WINDOW_FREERATIO | WINDOW_KEEPRATIO);
```

---

## 5. Implementation Analysis

### 5.1 Event Loop Architecture

```
┌─────────────────────────────────────────┐
│           Event Loop (waitKey)          │
├─────────────────────────────────────────┤
│ 1. Poll system events (mouse, key)     │
│ 2. Update trackbar positions            │
│ 3. Trigger callbacks                   │
│ 4. Call OpenGL/glfw render if enabled   │
│ 5. Return key code or timeout          │
└─────────────────────────────────────────┘
```

### 5.2 Platform Backend

| Platform | Backend |
|---------|---------|
| Windows | Win32 API |
| Linux | GTK+ 3.0 or Qt |
| macOS | Cocoa |
| Web/WASM | JavaScript Canvas |
| iOS | UIKit |
| Android | Java UI |

---

## 6. Code Examples

### 6.1 Basic Image Viewer

```cpp
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    Mat img = imread(argv[1]);
    if (img.empty()) {
        printf("Error: Cannot load image\n");
        return -1;
    }

    namedWindow("Image Viewer", WINDOW_KEEPRATIO | WINDOW_FREERATIO);
    imshow("Image Viewer", img);

    printf("Press 'q' or ESC to quit\n");
    int key;
    while ((key = waitKey(0)) != 'q' && key != 27) {
        // Wait for quit
    }

    destroyAllWindows();
    return 0;
}
```

### 6.2 Interactive Brightness Control

```cpp
#include <opencv2/highgui.hpp>

Mat originalImg;
int brightness = 50;

void updateBrightness(int, void*) {
    Mat adjusted;
    originalImg.convertTo(adjusted, -1, 1.0, brightness - 50);
    imshow("Brightness Control", adjusted);
}

int main() {
    originalImg = imread("photo.jpg");
    if (originalImg.empty()) return -1;

    namedWindow("Brightness Control", WINDOW_AUTOSIZE);

    createTrackbar("Brightness", "Brightness Control", &brightness, 100, updateBrightness);

    updateBrightness(0, nullptr);
    waitKey(0);

    return 0;
}
```

### 6.3 Mouse Event Handler

```cpp
#include <opencv2/highgui.hpp>

Mat img;
vector<Point> clickPoints;

void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        circle(img, Point(x, y), 5, Scalar(0, 0, 255), -1);
        clickPoints.push_back(Point(x, y));
        imshow("Click Demo", img);
        printf("Clicked at (%d, %d)\n", x, y);
    }
}

int main() {
    img = imread("photo.jpg");
    namedWindow("Click Demo");
    setMouseCallback("Click Demo", onMouse);

    imshow("Click Demo", img);
    waitKey(0);

    printf("Total clicks: %zu\n", clickPoints.size());
    return 0;
}
```

### 6.4 Trackbar + Mouse Combined

```cpp
#include <opencv2/highgui.hpp>

Mat img;
int radius = 10;
int colorIdx = 0;

Scalar getColor(int idx) {
    Scalar colors[] = {Scalar(0,0,255), Scalar(0,255,0), Scalar(255,0,0)};
    return colors[idx % 3];
}

void draw(int, void*) {
    Mat display = img.clone();
    for (const auto& pt : clickPoints) {
        circle(display, pt, radius, getColor(colorIdx), -1);
    }
    imshow("Interactive", display);
}

vector<Point> clickPoints;

void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        clickPoints.push_back(Point(x, y));
        draw(0, nullptr);
    }
}

int main() {
    img = imread("photo.jpg");
    namedWindow("Interactive", WINDOW_AUTOSIZE);

    createTrackbar("Radius", "Interactive", &radius, 50, draw);
    createTrackbar("Color", "Interactive", &colorIdx, 2, draw);

    setMouseCallback("Interactive", onMouse);
    draw(0, nullptr);

    waitKey(0);
    return 0;
}
```

---

## 7. Exercises

### Basic Level
1. Create a simple image viewer with keyboard navigation (arrow keys)
2. Add a trackbar to control image brightness
3. Display mouse coordinates when clicking on an image

### Intermediate Level
4. Create a zoom tool with mouse wheel support
5. Implement a color picker using trackbars for B, G, R channels
6. Build an image annotation tool with different shapes

### Advanced Level
7. Implement a custom window with Qt-like behavior using mouse callbacks
8. Create a real-time filter viewer with multiple trackbars

---

## 8. References

| Resource | Link |
|----------|------|
| Official Documentation | [highgui module](https://docs.opencv.org/4.14.0/d4/da4/group__highgui.html) |
| namedWindow Reference | [namedWindow](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#ga9d805922020e34e47b805da6f6a44d57) |
| imshow Reference | [imshow](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#ga5813e1ef30c55bcbb9ed9b820e5ffe9f) |
| setMouseCallback Reference | [setMouseCallback](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#ga84e2f77f525b50d8c8c0e4a7b9f0eae8) |
| createTrackbar Reference | [createTrackbar](https://docs.opencv.org/4.14.0/d7/dfc/group__highgui.html#gac5a3d2c47c26c1e340c8e8af3dd637a3) |

---

## Update History

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | Initial highgui module documentation |