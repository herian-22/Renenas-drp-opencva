# Quick Start Guide

This guide will help you get the DRP-AI MOIL application running quickly.

## Prerequisites Checklist

- [ ] Renesas RZ/V2H or RZ/V2N board with display
- [ ] Fisheye USB camera
- [ ] Poky SDK installed at `/opt/poky/3.1.31` or similar
- [ ] TVM runtime for DRP-AI
- [ ] YOLOv8 model compiled for DRP-AI (TVM format)

## Step-by-Step Setup

### 1. Prepare Your Environment (5 minutes)

```bash
# Source the SDK
source /opt/poky/3.1.31/environment-setup-aarch64-poky-linux

# Set TVM path
export TVM_HOME=/path/to/tvm

# Verify environment
echo "SDK: $SDKTARGETSYSROOT"
echo "TVM: $TVM_HOME"
```

### 2. Prepare Model Files (2 minutes)

Create your model directory and label file:

```bash
# Create model directory (use your model name)
mkdir -p unicornv8n

# Copy TVM model files into the directory
cp /path/to/your/model/*.{so,json,params} unicornv8n/

# Verify files are present
ls unicornv8n/
# Expected: deploy.so, deploy.json, deploy.params

# Create label file
cat > unicornv8m.txt << 'EOF'
person
car
bicycle
motorcycle
truck
EOF
```

### 3. Configure Model Settings (2 minutes)

Edit `include/define.h`:

```cpp
// Line 7: Set your model directory name
const std::string model_dir = "unicornv8n";  // Your model folder

// Line 11: Set your label file name
const std::string label_list = "unicornv8m.txt";  // Your labels file

// Line 14: Set number of classes (must match your model)
#define NUM_CLASS (5)  // Change to your class count

// Line 22-23: Adjust detection thresholds if needed
#define TH_PROB (0.25f)  // Lower = more detections (more false positives)
#define TH_NMS (0.45f)   // Lower = fewer overlapping boxes
```

### 4. Build the Application (3 minutes)

```bash
# Run build script
./build.sh

# Build should complete with:
# "Build successful!"
# "Executable: build/moildev_app+DrpAiYolo"
```

If build fails, check:
- SDK environment is sourced: `echo $SDKTARGETSYSROOT`
- All dependencies installed: `pkg-config --list-all | grep opencv`

### 5. Prepare Camera Parameters (Optional, 2 minutes)

If using default parameters, skip this step. Otherwise:

```bash
# Copy example
cp camera_params.txt.example asset/parameter_camera/mycamera.txt

# Edit with your camera's calibration values
nano asset/parameter_camera/mycamera.txt
```

The application will prompt you to select camera parameters at startup.

### 6. Run the Application (1 minute)

```bash
cd build

# Run with camera 0 (default /dev/video0)
./moildev_app+DrpAiYolo 0

# Or camera 1
./moildev_app+DrpAiYolo 1
```

### 7. First Launch

When the application starts:

1. **Camera Selection Dialog** will appear
   - Select your camera from the dropdown
   - Click OK

2. **Parameter Selection Dialog** will appear
   - Select camera parameter file (e.g., `LRCP_IMX586_4.txt`)
   - Click OK

3. **Main Window** should open showing:
   - Camera feed with fisheye dewarping
   - Object detection bounding boxes
   - FPS counter

## Basic Usage

### View Modes

Click buttons at bottom of window:
- **MODE: CAR VIEW** - Multi-angle dashboard view (recommended for vehicles)
- **MODE: ORIGINAL** - Raw fisheye image
- **MODE: PANORAMA** - 360° unwrapped view
- **MODE: 3x480** - Three-way split view

### Essential Keyboard Shortcuts

- `1-4` - Switch view modes
- `↑↓←→` - Pan the view
- `+/-` - Zoom in/out
- `H` - Horizontal flip
- `V` - Vertical flip
- `S` - Save current settings
- `Esc` - Exit

### Mouse Control

- **Click and drag** - Pan view manually
- **Scroll wheel** - Zoom

## Troubleshooting

### "Failed to open /dev/drpai0"

**Solution**: Check device permissions
```bash
ls -l /dev/drpai0
sudo chmod 666 /dev/drpai0
```

### "Failed to init YOLO model"

**Checklist**:
- [ ] Model directory exists: `ls unicornv8n/`
- [ ] Files present: `ls unicornv8n/*.{so,json,params}`
- [ ] TVM_HOME set: `echo $TVM_HOME`
- [ ] Model compiled for correct DRP-AI version

### "OCA Library not found"

This is a **warning**, not an error. Application will run on CPU (slower).

**Optional fix**: Ensure OpenCV with DRP-OCA support is installed:
```bash
strings /usr/lib64/libopencv_imgproc.so | grep OCA_Activate
```

### No camera detected

**Check**:
```bash
# List cameras
ls -la /dev/video*

# Test camera
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all
```

### Low FPS / Slow performance

**Optimize**:
1. Lower input resolution in `include/define.h`:
   ```cpp
   #define DRPAI_IN_WIDTH (640)   // Try 320 for faster
   #define DRPAI_IN_HEIGHT (480)  // Try 240 for faster
   ```
2. Increase detection interval:
   ```cpp
   #define DRPAI_FREQ (2)  // Try 3 or 4 (detect every N frames)
   ```
3. Rebuild: `./build.sh`

## Next Steps

- **Fine-tune detection**: Adjust `TH_PROB` and `TH_NMS` in `define.h`
- **Customize views**: Modify PTZ angles in the code
- **Train custom model**: See YOLOv8 documentation, compile with TVM for DRP-AI
- **Read full docs**: See `README.md` and `CONTRIBUTING.md`

## Quick Reference

### File Locations

- Executable: `build/moildev_app+DrpAiYolo`
- Configuration: `include/define.h`
- Camera params: `asset/parameter_camera/`
- Model directory: `unicornv8n/` (your choice)
- Label file: `unicornv8m.txt` (your choice)
- Saved settings: `moildev_config.txt` (auto-generated)

### Important Constants

| Parameter | File | Default | Purpose |
|-----------|------|---------|---------|
| `model_dir` | define.h | "unicornv8n" | Model directory name |
| `NUM_CLASS` | define.h | 5 | Number of detection classes |
| `TH_PROB` | define.h | 0.25 | Detection confidence threshold |
| `TH_NMS` | define.h | 0.45 | NMS IoU threshold |
| `DRPAI_FREQ` | define.h | 2 | Detection every N frames |
| `DRPAI_IN_WIDTH` | define.h | 640 | AI input width |
| `DRPAI_IN_HEIGHT` | define.h | 480 | AI input height |

## Getting Help

- Check `README.md` for detailed documentation
- Check `CONTRIBUTING.md` for development guidelines
- Open an issue on GitHub for bugs or questions
- Contact Renesas support for hardware/SDK issues

---

**Total estimated setup time: 15-20 minutes**

Good luck! 🚀
