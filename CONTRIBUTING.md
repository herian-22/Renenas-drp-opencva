# Contributing to DRP-AI MOIL Fisheye + YOLOv8

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

1. **Cross-Compilation Toolchain**: Renesas Poky SDK 3.1.31 or later
2. **Development Board**: RZ/V2H or RZ/V2N with DRP-AI support
3. **Tools**: Git, CMake 3.10+, text editor/IDE

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/herian-22/Renesas-drp-opencva.git
cd Renesas-drp-opencva

# Source the SDK
source /opt/poky/3.1.31/environment-setup-aarch64-poky-linux

# Set TVM path
export TVM_HOME=/path/to/tvm

# Build
./build.sh
```

## Code Style Guidelines

### C++ Style

- **Indentation**: 4 spaces (no tabs)
- **Naming Conventions**:
  - Classes: `PascalCase` (e.g., `DrpAiYolo`)
  - Functions: `camelCase` (e.g., `runDetection`)
  - Variables: `snake_case` (e.g., `frame_count`)
  - Constants/Macros: `UPPER_CASE` (e.g., `MAX_DETECTIONS`)
- **Comments**: Use `//` for inline comments, `/* */` for block comments
- **Headers**: Always use include guards or `#pragma once`

### Example

```cpp
class ImageProcessor {
public:
    ImageProcessor();
    bool initializeBuffers(int width, int height);
    
private:
    int frame_count;
    static const int MAX_BUFFER_SIZE = 1024;
};
```

## Testing Changes

### Building

Always test your changes with a clean build:

```bash
rm -rf build
./build.sh
```

### Running on Target

Deploy to the development board and test:

```bash
# Copy executable to board
scp build/moildev_app+DrpAiYolo root@<board-ip>:/home/root/

# SSH to board
ssh root@<board-ip>

# Run application
cd /home/root
./moildev_app+DrpAiYolo 0
```

### Performance Testing

For performance-sensitive changes:

1. Test with different resolutions (use 'R' key)
2. Monitor FPS display
3. Check AI inference times in console output
4. Verify CPU usage: `top -p $(pidof moildev_app+DrpAiYolo)`

## Submitting Changes

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Build/tooling changes

Example:
```
feat(detection): add confidence threshold adjustment

Add runtime adjustment of YOLO detection confidence threshold
via keyboard shortcuts (T/Y keys).

Closes #42
```

### Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/my-feature`
3. **Make changes** following code style guidelines
4. **Test thoroughly** on actual hardware
5. **Commit** with descriptive messages
6. **Push** to your fork
7. **Open a Pull Request** with:
   - Clear description of changes
   - Why the change is needed
   - Any breaking changes
   - Test results/screenshots

### PR Checklist

- [ ] Code builds without warnings
- [ ] Tested on RZ/V2H or RZ/V2N board
- [ ] No performance regression
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

## Areas for Contribution

### High Priority

1. **Additional Camera Support**:
   - CSI camera interface support
   - Multiple concurrent cameras
   - Camera auto-detection

2. **Model Flexibility**:
   - Support for other YOLO versions (v5, v7, v9)
   - Custom object detection models
   - Runtime model switching

3. **Recording Features**:
   - Video recording of processed output
   - Snapshot capture
   - Detection event logging

### Medium Priority

1. **UI Enhancements**:
   - Configuration panel for runtime settings
   - Detection statistics overlay
   - Custom color schemes

2. **Performance Monitoring**:
   - Detailed performance profiling
   - Memory usage tracking
   - Temperature monitoring

3. **Configuration**:
   - JSON-based configuration
   - Profile management (save/load complete settings)
   - Command-line argument parsing

### Low Priority

1. **Testing**:
   - Unit tests for key components
   - Integration test suite
   - CI/CD pipeline

2. **Portability**:
   - Support for other Renesas platforms
   - Generic ARM64 fallback mode

## Debugging Tips

### Common Issues

**Segmentation Fault**:
```bash
# Run with gdb on target
gdb ./moildev_app+DrpAiYolo
(gdb) run 0
# After crash:
(gdb) backtrace
```

**DRP-AI Initialization Fails**:
- Check `/dev/drpai0` permissions: `ls -l /dev/drpai0`
- Verify TVM_HOME: `echo $TVM_HOME`
- Check model files exist: `ls -la unicornv8n/`

**Performance Issues**:
- Enable timing debug: Uncomment `#define DEBUG_TIME_FLG` in `define.h`
- Check DRP-OCA activation in console output
- Monitor with: `top`, `htop`, `perf`

### Logging

Add debug output:

```cpp
#ifdef DEBUG_VERBOSE
    std::cout << "[DEBUG] Frame processed: " << frame_id << std::endl;
#endif
```

## Hardware-Specific Considerations

### Memory Management

- Use `libmmngr` for DMA-capable buffers
- Avoid large stack allocations (use heap)
- Clean up resources in destructors

### Real-time Performance

- Minimize memory allocations in hot paths
- Use pre-allocated buffers
- Leverage NEON intrinsics for critical loops
- Profile with `perf record` / `perf report`

### DRP-AI Usage

- Inference buffer must be DMA-capable
- Respect alignment requirements (32-byte for DRP-AI)
- Check return codes from DRP-AI API calls

## Documentation

### Code Documentation

- Document all public APIs
- Explain non-obvious algorithms
- Add references for mathematical formulas

Example:
```cpp
/**
 * @brief Decode DFL (Distribution Focal Loss) prediction
 * 
 * Converts YOLOv8 DFL output to bounding box coordinates.
 * Reference: https://arxiv.org/abs/2006.04388
 * 
 * @param tensor Pointer to DFL tensor [64 elements]
 * @return Decoded coordinate value
 */
float dfl_decode(float* tensor);
```

### Documentation Updates

When adding features, update:
- `README.md` - User-facing features
- `CONTRIBUTING.md` - Development guidelines (this file)
- Code comments - Implementation details
- `include/define.h` - Configuration options

## Getting Help

- **Issues**: Open a GitHub issue for bugs/questions
- **Discussions**: Use GitHub Discussions for general topics
- **Renesas Support**: For DRP-AI hardware/SDK issues

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project.

Thank you for contributing!
