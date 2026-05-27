# OpenCV Learning Guide

> **Version**: OpenCV 4.14.0-pre
> **Repository**: https://github.com/jmlw8023/opencv-4.14.0-learning
> **License**: Apache 2.0

---

## Version Information

| Item | Value |
|------|-------|
| OpenCV Version | 4.14.0-pre |
| Major | 4 |
| Minor | 14 |
| Patch | 0 |
| Status | pre |
| Source | [opencv/opencv](https://github.com/opencv/opencv) |

---

## Table of Contents

- [Learning Roadmap](./README.md#learning-roadmap)
- [Module Guides](./modules/)
- [How to Contribute](./CONTRIBUTING.md)

---

## Learning Roadmap

### Stage 1: Getting Started

| Module | Description | Status |
|--------|-------------|--------|
| [core](./modules/core/README.md) | Core functionality: Mat, arrays, XML/YAML, parallel processing | 🔄 In Progress |
| [imgcodecs](./modules/imgcodecs/README.md) | Image codec: imread/imwrite, image format decoding | 📋 Planned |
| [highgui](./modules/highgui/README.md) | High-level GUI: namedWindow, imshow, Trackbar | 📋 Planned |

### Stage 2: Fundamentals

| Module | Description | Status |
|--------|-------------|--------|
| [imgproc](./modules/imgproc/README.md) | Image processing: filtering, morphology, geometry, color spaces | 📋 Planned |
| [videoio](./modules/videoio/README.md) | Video I/O: VideoCapture, VideoWriter | 📋 Planned |
| [video](./modules/video/README.md) | Video analysis: KCF, MOSSE tracking, OpticalFlow | 📋 Planned |

### Stage 3: Intermediate

| Module | Description | Status |
|--------|-------------|--------|
| [features2d](./modules/features2d/README.md) | Feature detection: SIFT/SURF/ORB, BFMatcher/FLANN | 📋 Planned |
| [calib3d](./modules/calib3d/README.md) | Camera calibration: intrinsics/extrinsics, stereo matching | 📋 Planned |
| [objdetect](./modules/objdetect/README.md) | Object detection: Haar, HOG, SSD/YOLO | 📋 Planned |

### Stage 4: Advanced

| Module | Description | Status |
|--------|-------------|--------|
| [dnn](./modules/dnn/README.md) | Deep Neural Network: ONNX, TensorFlow, PyTorch model loading | 📋 Planned |
| [photo](./modules/photo/README.md) | Computational photography: HDR, denoising, inpainting | 📋 Planned |
| [stitching](./modules/stitching/README.md) | Image stitching: panorama, multi-band blending | 📋 Planned |

### Stage 5: Expert

| Module | Description | Status |
|--------|-------------|--------|
| [gapi](./modules/gapi/README.md) | Graph API: kernel development, async execution | 📋 Planned |
| [ml](./modules/ml/README.md) | Machine Learning: SVM, Decision Trees, Neural Networks | 📋 Planned |
| [flann](./modules/flann/README.md) | FLANN: KD-Tree, LSH, hierarchical clustering | 📋 Planned |
| [world](./modules/world/README.md) | Unified入口 module | 📋 Planned |

---

## Module Documentation Structure

Each module follows this structure:

```markdown
# Module Name

## Overview
## Key Data Structures
## Core APIs
## Implementation Analysis
## Code Examples
## Exercises
## References
```

---

## Development Guidelines

### Git Workflow

```bash
# 1. Create a new branch for each module
git checkout -b module/core

# 2. Make changes and commit
git add .
git commit -m "docs(core): add detailed analysis of Mat structure"

# 3. Push to remote
git push -u origin module/core
```

### Commit Message Convention

Format: `<type>(<module>): <description>`

Types:
- `docs` - Documentation only
- `fix` - Bug fix
- `feat` - New feature
- `refactor` - Code refactoring
- `test` - Test updates

Examples:
```bash
git commit -m "docs(core): add Mat memory model analysis"
git commit -m "docs(imgproc): add filtering algorithms guide"
git commit -m "feat(core): add practice exercises"
```

---

## References

- [Official Documentation](https://docs.opencv.org/4.14.0/)
- [OpenCV Source](https://github.com/opencv/opencv)
- [OpenCV Tutorials](https://docs.opencv.org/4.14.0/tutorials/tutorials.html)
- [Learning OpenCV 4](https://www.oreilly.com/library/view/learning-opencv-4/)

---

## Update History

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-27 | 4.14.0-pre | Initial commit with learning roadmap and core module guide |