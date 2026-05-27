# Contributing to OpenCV Learning

## How to Contribute

### Module Documentation

Each module documentation should follow the structure:

```markdown
# Module Name

## 1. Overview
Brief description of the module's purpose and capabilities.

## 2. Key Data Structures
Core data structures with detailed explanations.

## 3. Core APIs
Main functions with usage examples.

## 4. Implementation Analysis
Source code analysis and key implementation details.

## 5. Code Examples
Practical examples demonstrating module usage.

## 6. Exercises
Practice problems from basic to advanced.

## 7. References
External resources and official documentation links.
```

### Module Status Legend

| Symbol | Meaning |
|--------|---------|
| 📋 Planned | Not yet started |
| 🔄 In Progress | Currently studying |
| ✅ Completed | Documentation complete |

### Quality Standards

1. **Code Examples**: All code must be compilable and tested
2. **Explanations**: Include both Chinese and English for key terms
3. **References**: Link to official OpenCV documentation
4. **Updates**: Log all changes with date and version

---

## Module Study Order

1. Start with [core](./modules/core/README.md) (Stage 1)
2. Proceed to imgcodecs and highgui (Stage 1)
3. Continue with imgproc → videoio → video (Stage 2)
4. Then features2d → calib3d → objdetect (Stage 3)
5. Follow with dnn → photo → stitching (Stage 4)
6. Finish with gapi → ml → flann (Stage 5)

---

## Update Log Format

```markdown
## Update History

| Date | Version | Changes |
|------|---------|---------|
| YYYY-MM-DD | x.x.x | Description |
```

---

## Git Commit Convention

```bash
# Format
git commit -m "<type>(<module>): <description>"

# Types
docs    - Documentation only
fix     - Bug fix
feat    - New feature
refactor- Code refactoring
test    - Test updates

# Examples
git commit -m "docs(core): add Mat memory model analysis"
git commit -m "docs(imgproc): add filtering algorithms guide"
git commit -m "feat(core): add practice exercises"
```