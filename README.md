# DRP-AI MOIL Fisheye + YOLOv8 Object Detection

A real-time fisheye camera dewarping and object detection application leveraging Renesas DRP-AI (Deep Learning Processing) hardware acceleration and MOIL (Multiscale Omnidirectional Image Layout) technology.

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in 15 minutes
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions
- **[Contributing Guide](CONTRIBUTING.md)** - Development guidelines
- **[This README](#)** - Comprehensive project documentation

## Features

### Core Capabilities
- **Real-time Fisheye Dewarping**: Process omnidirectional fisheye camera images using MOIL algorithm
- **Hardware-Accelerated Object Detection**: YOLOv8 inference using Renesas DRP-AI accelerator
- **Multiple View Modes**: 
  - Car View (multi-angle dashboard view)
  - Original (raw fisheye image)
  - Panorama (360° unwrapped view)
  - Triple View (3x480 split screen)
- **DRP-OCA Integration**: OpenCV operations accelerated via Renesas DRP-OpenCV Accelerator
- **GTK3 GUI**: Interactive user interface with real-time controls

### Performance Optimizations
- **Hardware Acceleration**: 
  - DRP-AI for neural network inference
  - DRP-OCA for image processing operations
  - NEON SIMD optimizations (Cortex-A55)
- **Multi-threading**: Separate threads for capture, processing, and display
- **LUT-based Dewarping**: Pre-computed lookup tables for fast fisheye correction

## Requirements

### Hardware
- Renesas RZ/V2H or RZ/V2N evaluation board
- Fisheye camera (USB or CSI interface)
- Display with 1280x720 minimum resolution

### Software Dependencies
- **Build Tools**: CMake 3.10+, GCC 8.3+ with ARM cross-compilation support
- **Libraries**:
  - OpenCV 4.x (with Renesas DRP-OCA support)
  - GTK+ 3.20+
  - TVM Runtime (Apache TVM for DRP-AI)
  - libmmngr (Renesas memory manager)
  - GLib/GResource
- **SDK**: Renesas Poky SDK 3.1.31 or compatible

## Installation

### 1. Setup Development Environment

```bash
# Source the Poky SDK environment
source /opt/poky/3.1.31/environment-setup-aarch64-poky-linux

# Set TVM_HOME for DRP-AI runtime
export TVM_HOME=/path/to/tvm
```

### 2. Prepare Model and Resources

```bash
# Place your YOLOv8 TVM model in the project directory
# Model directory should contain: deploy.so, deploy.json, deploy.params
mkdir -p unicornv8n

# Create label file listing class names (one per line)
cat > unicornv8m.txt << EOF
class1
class2
class3
class4
class5
EOF

# Place camera parameters in asset/parameter_camera/
# Format: PARA followed by 6 polynomial coefficients
```

### 3. Build

```bash
# Check dependencies (recommended)
./check_deps.sh

# Clean build
./build.sh

# Or manual build
mkdir -p build
cd build
cmake ..
make -j8
```

The executable `moildev_app+DrpAiYolo` will be created in the `build/` directory.

## Configuration

### Camera Parameters

Fisheye camera calibration parameters are loaded from `asset/parameter_camera/*.txt`:

```
PARA <p0> <p1> <p2> <p3> <p4> <p5>
ORG_W <image_width>
ORG_H <image_height>
ORG_ICX <center_x>
ORG_ICY <center_y>
```

Example files provided:
- `LRCP_IMX586_4.txt` - Sony IMX586 sensor
- `wxsj_7730_4.txt` - Custom fisheye calibration

### Model Configuration

Edit `include/define.h` to configure your YOLOv8 model:

```cpp
const std::string model_dir = "unicornv8n";      // Model directory
const std::string label_list = "unicornv8m.txt"; // Label file
#define NUM_CLASS (5)                             // Number of classes
#define TH_PROB (0.25f)                          // Confidence threshold
#define TH_NMS (0.45f)                           // NMS IoU threshold
```

## Usage

### Running the Application

```bash
# Basic usage (default camera /dev/video0)
./moildev_app+DrpAiYolo 0

# Specify camera index
./moildev_app+DrpAiYolo 1

# The camera parameter file can be selected at startup
```

### Keyboard Controls

| Key | Function |
|-----|----------|
| `1` | Car View mode (main multi-angle view) |
| `2` | Original mode (raw fisheye) |
| `3` | Panorama mode (360° unwrapped) |
| `4` | Triple View mode (3x480 split) |
| `↑/↓` | Adjust vertical angle (beta) |
| `←/→` | Adjust horizontal angle (alpha) |
| `+/-` | Adjust zoom level |
| `H` | Toggle horizontal flip |
| `V` | Toggle vertical flip |
| `S` | Save current view configuration |
| `R` | Cycle through resolution benchmarks |
| `Esc` | Exit application |

### Mouse Controls

- **Click & Drag**: Pan the view (in PTZ modes)
- **Scroll**: Zoom in/out

## Architecture

### Threading Model

```
Main Thread (GTK)
    ├─> Capture Thread: Camera frame grabbing
    ├─> Map Update Thread: Fisheye dewarping computation  
    └─> Pipeline Thread: DRP-AI inference + Display rendering
```

### Processing Pipeline

1. **Camera Capture**: Grab frames via OpenCV VideoCapture
2. **Fisheye Dewarping**: Apply MOIL algorithm using pre-computed maps
3. **Object Detection**: 
   - Resize to 640x640 (YOLOv8 input)
   - DRP-AI inference
   - Post-processing (NMS, bounding boxes)
4. **Rendering**: Overlay detections and UI elements
5. **Display**: Update GTK image widget

## Performance Tuning

### Compiler Optimizations

The project uses aggressive optimizations for embedded ARM platform:

```cmake
-O3                  # Maximum optimization
-mcpu=cortex-a55    # Target Cortex-A55 (enables NEON)
-ffast-math         # Fast floating point (critical for fisheye math)
-ftree-vectorize    # Auto-vectorization
-fopenmp            # OpenMP parallelization
```

### DRP-AI Configuration

Adjust inference frequency in `include/define.h`:

```cpp
#define DRPAI_FREQ (2)  // Run detection every N frames
```

### Resolution Settings

Balance between quality and performance:

```cpp
#define DRPAI_IN_WIDTH (640)   // Lower = faster, less detail
#define DRPAI_IN_HEIGHT (480)
```

## Troubleshooting

### DRP-OCA Acceleration Not Working

Check if the OpenCV library has DRP-OCA support:

```bash
strings /usr/lib64/libopencv_imgproc.so | grep OCA_Activate
```

If missing, the application will fall back to CPU (slower but functional).

### Model Loading Fails

Verify:
1. Model directory path is correct
2. TVM_HOME environment variable is set
3. Files present: `deploy.so`, `deploy.json`, `deploy.params`
4. Model was compiled for the correct DRP-AI architecture

### Camera Not Found

```bash
# List available cameras
ls -la /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices
```

## Development

### Adding New View Modes

1. Add enum to `ViewMode` in `main_drp_moildev.cpp`
2. Implement rendering logic in `pipelineThread()`
3. Add button handler in GUI callbacks
4. Update keyboard shortcut handling

### Custom MOIL Parameters

Calibrate your fisheye camera using the MOIL calibration tool, then:

1. Save parameters to `asset/parameter_camera/custom.txt`
2. Load at startup via `loadCameraParamsText("custom.txt")`

## License

This project integrates multiple components:
- MOIL dewarping library (proprietary/research)
- Renesas DRP-AI SDK (Renesas Electronics license)
- OpenCV (Apache 2.0)
- TVM (Apache 2.0)

Refer to individual component licenses for usage terms.

## References

- [Renesas RZ/V2H DRP-AI](https://www.renesas.com/products/microcontrollers-microprocessors/rz-arm-based-high-end-32-64-bit-mpus/rzv2h-ultra-high-speed-edge-ai-mpu)
- [Apache TVM](https://tvm.apache.org/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [MOIL Technology](https://moil.ncut.edu.tw/)

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style conventions
- Changes maintain real-time performance
- Cross-compilation compatibility with Poky SDK

## Support

For issues specific to:
- **DRP-AI/Hardware**: Contact Renesas support
- **MOIL Algorithm**: Refer to MOIL documentation
- **This Application**: Open a GitHub issue
