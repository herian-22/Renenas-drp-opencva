# Deployment Guide

This guide covers deploying the DRP-AI MOIL application to production Renesas boards.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Building for Production](#building-for-production)
3. [Packaging](#packaging)
4. [Deployment](#deployment)
5. [System Configuration](#system-configuration)
6. [Auto-Start Setup](#auto-start-setup)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

Before deploying to production:

- [ ] Application tested on development board
- [ ] Model achieves acceptable accuracy
- [ ] Performance meets requirements (FPS, latency)
- [ ] Camera parameters calibrated for production camera
- [ ] All dependencies identified and available
- [ ] Backup/recovery plan in place

## Building for Production

### 1. Optimize Build Configuration

Edit `CMakeLists.txt` for production:

```cmake
# Enable maximum optimization
set(CMAKE_BUILD_TYPE Release)

# Strip debug symbols for smaller binary
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
```

### 2. Build

```bash
# Clean build
rm -rf build

# Build with production flags
./build.sh

# Verify binary size
ls -lh build/moildev_app+DrpAiYolo
```

### 3. Test Production Build

Run thorough testing on development board:

```bash
# Performance test
./moildev_app+DrpAiYolo 0

# Check for memory leaks (run for extended period)
valgrind --leak-check=full ./moildev_app+DrpAiYolo 0

# Monitor resource usage
top -p $(pidof moildev_app+DrpAiYolo)
```

## Packaging

### Create Deployment Package

```bash
# Create deployment directory
mkdir -p deploy/moildev_app
cd deploy/moildev_app

# Copy executable
cp ../../build/moildev_app+DrpAiYolo .

# Copy model files
mkdir -p models
cp -r ../../unicornv8n models/
cp ../../unicornv8m.txt models/

# Copy camera parameters
mkdir -p config/parameter_camera
cp ../../asset/parameter_camera/*.txt config/parameter_camera/

# Copy GUI resources
cp ../../moildev_resources.xml config/

# Copy libraries (if not system-installed)
mkdir -p lib
cp ../../lib/*.a lib/  # libmoildevren.a

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
export DRP_EXE_PATH=$(pwd)
./moildev_app+DrpAiYolo 0
EOF
chmod +x run.sh

# Create README
cat > README.txt << 'EOF'
DRP-AI MOIL Application Deployment Package

Files:
- moildev_app+DrpAiYolo : Main executable
- models/               : YOLOv8 model files
- config/               : Configuration files
- lib/                  : Required libraries
- run.sh                : Launch script

To run:
  ./run.sh

Requirements:
- Renesas RZ/V2H or RZ/V2N board
- OpenCV 4.x with DRP-OCA support
- GTK+ 3.20+
- TVM Runtime
EOF

# Create tarball
cd ..
tar -czf moildev_app_deploy_$(date +%Y%m%d).tar.gz moildev_app/
cd ..
```

### Package Contents

```
moildev_app_deploy_YYYYMMDD.tar.gz
└── moildev_app/
    ├── moildev_app+DrpAiYolo    # Executable
    ├── run.sh                    # Launch script
    ├── README.txt
    ├── models/
    │   ├── unicornv8n/          # Model directory
    │   └── unicornv8m.txt       # Labels
    ├── config/
    │   ├── parameter_camera/
    │   └── moildev_resources.xml
    └── lib/
        └── libmoildevren.a
```

## Deployment

### 1. Transfer to Target Board

```bash
# Via SCP
scp moildev_app_deploy_YYYYMMDD.tar.gz root@<board-ip>:/home/root/

# Via USB
# Copy to USB drive, mount on board, copy files
```

### 2. Install on Board

```bash
# SSH to board
ssh root@<board-ip>

# Extract package
cd /home/root
tar -xzf moildev_app_deploy_YYYYMMDD.tar.gz
cd moildev_app

# Test run
./run.sh
```

### 3. System Integration

Install to system location (optional):

```bash
# Create application directory
sudo mkdir -p /opt/moildev_app

# Copy files
sudo cp -r moildev_app/* /opt/moildev_app/

# Create symlink for convenience
sudo ln -s /opt/moildev_app/run.sh /usr/local/bin/moildev_app
```

## System Configuration

### 1. DRP-AI Device Permissions

Ensure application can access DRP-AI device:

```bash
# Create udev rule
sudo cat > /etc/udev/rules.d/99-drpai.rules << 'EOF'
KERNEL=="drpai0", MODE="0666"
EOF

# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 2. Camera Permissions

For USB cameras:

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Or create udev rule
sudo cat > /etc/udev/rules.d/99-camera.rules << 'EOF'
SUBSYSTEM=="video4linux", MODE="0666"
EOF
```

### 3. Memory Configuration

For optimal performance, configure CMA (Contiguous Memory Allocator):

Edit `/boot/cmdline.txt` or kernel boot parameters:

```
cma=256M  # Adjust based on your needs
```

### 4. CPU Governor

Set performance mode for consistent FPS:

```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Auto-Start Setup

### Option 1: systemd Service

Create service file:

```bash
sudo cat > /etc/systemd/system/moildev-app.service << 'EOF'
[Unit]
Description=DRP-AI MOIL Application
After=network.target graphical.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/moildev_app
Environment="DISPLAY=:0"
Environment="TVM_HOME=/path/to/tvm"
ExecStart=/opt/moildev_app/run.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable moildev-app
sudo systemctl start moildev-app

# Check status
sudo systemctl status moildev-app

# View logs
sudo journalctl -u moildev-app -f
```

### Option 2: Desktop Auto-Start

For graphical login sessions:

```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/moildev-app.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=MOIL DRP-AI App
Exec=/opt/moildev_app/run.sh
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF
```

### Option 3: rc.local (Legacy)

```bash
# Edit /etc/rc.local
sudo nano /etc/rc.local

# Add before 'exit 0':
export TVM_HOME=/path/to/tvm
export DISPLAY=:0
cd /opt/moildev_app && ./run.sh &
```

## Performance Tuning

### Production Configuration

Edit `include/define.h` for production:

```cpp
// Optimize for your use case

// For vehicle dashboard (prioritize FPS)
#define DRPAI_IN_WIDTH (640)
#define DRPAI_IN_HEIGHT (480)
#define DRPAI_FREQ (2)       // Detect every 2 frames
#define TH_PROB (0.30f)      // Slightly higher threshold

// For high-accuracy (sacrifice some FPS)
#define DRPAI_IN_WIDTH (1280)
#define DRPAI_IN_HEIGHT (720)
#define DRPAI_FREQ (3)       // Detect every 3 frames
#define TH_PROB (0.25f)

// For embedded/resource-constrained
#define DRPAI_IN_WIDTH (320)
#define DRPAI_IN_HEIGHT (240)
#define DRPAI_FREQ (4)       // Detect every 4 frames
#define TH_PROB (0.35f)
```

### Runtime Optimization

```bash
# Set CPU affinity (run on specific cores)
taskset -c 0,1,2,3 ./moildev_app+DrpAiYolo 0

# Set process priority
nice -n -10 ./moildev_app+DrpAiYolo 0

# Combine both
nice -n -10 taskset -c 0,1,2,3 ./moildev_app+DrpAiYolo 0
```

## Monitoring

### Log Monitoring

```bash
# Real-time console output
./moildev_app+DrpAiYolo 0 2>&1 | tee app.log

# Or redirect to file
./moildev_app+DrpAiYolo 0 > app.log 2>&1 &

# Monitor log file
tail -f app.log
```

### Performance Monitoring

```bash
# CPU and memory usage
top -p $(pidof moildev_app+DrpAiYolo)

# Detailed stats
htop -p $(pidof moildev_app+DrpAiYolo)

# Continuous logging
while true; do
    date >> perf.log
    top -b -n 1 -p $(pidof moildev_app+DrpAiYolo) | head -20 >> perf.log
    sleep 5
done
```

### Temperature Monitoring

```bash
# Check board temperature
cat /sys/class/thermal/thermal_zone*/temp

# Monitor continuously
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
```

## Troubleshooting

### Application Crashes

```bash
# Enable core dumps
ulimit -c unlimited

# Run with gdb for debugging
gdb ./moildev_app+DrpAiYolo
(gdb) run 0
# After crash:
(gdb) backtrace
```

### Memory Issues

```bash
# Check available memory
free -h

# Monitor memory usage
watch -n 1 free -h

# Check for memory leaks (development)
valgrind --leak-check=full --show-leak-kinds=all ./moildev_app+DrpAiYolo 0
```

### DRP-AI Device Issues

```bash
# Check device exists
ls -l /dev/drpai0

# Check permissions
ls -l /dev/drpai0
# Should show: crw-rw-rw-

# Check if device is busy
lsof | grep drpai

# Reload DRP-AI driver (if available)
sudo modprobe -r drpai
sudo modprobe drpai
```

### Camera Issues

```bash
# List cameras
ls -l /dev/video*

# Check camera info
v4l2-ctl -d /dev/video0 --all

# Test camera capture
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=10

# Check permissions
ls -l /dev/video0
```

### Performance Issues

1. **Low FPS**:
   - Reduce input resolution
   - Increase `DRPAI_FREQ`
   - Check CPU frequency: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq`

2. **High CPU Usage**:
   - Verify DRP-OCA is activated (check console output)
   - Reduce resolution
   - Optimize map update frequency

3. **High Memory Usage**:
   - Check for leaks with valgrind
   - Verify buffer sizes in code
   - Monitor with: `pmap $(pidof moildev_app+DrpAiYolo)`

## Backup and Recovery

### Backup Configuration

```bash
# Backup critical files
tar -czf moildev_backup_$(date +%Y%m%d).tar.gz \
    /opt/moildev_app/config/ \
    /opt/moildev_app/models/ \
    /etc/systemd/system/moildev-app.service
```

### Recovery Procedure

```bash
# Stop application
sudo systemctl stop moildev-app

# Restore from backup
tar -xzf moildev_backup_YYYYMMDD.tar.gz -C /

# Restart
sudo systemctl start moildev-app
```

## Update Procedure

### Rolling Update

```bash
# 1. Stop current application
sudo systemctl stop moildev-app

# 2. Backup current version
sudo cp -r /opt/moildev_app /opt/moildev_app.backup

# 3. Deploy new version
sudo tar -xzf moildev_app_deploy_NEW.tar.gz -C /opt/

# 4. Test new version manually
cd /opt/moildev_app
./run.sh
# Press Ctrl+C after verifying it works

# 5. Start service
sudo systemctl start moildev-app

# 6. Verify
sudo systemctl status moildev-app
```

### Rollback

```bash
# If update fails, rollback:
sudo systemctl stop moildev-app
sudo rm -rf /opt/moildev_app
sudo mv /opt/moildev_app.backup /opt/moildev_app
sudo systemctl start moildev-app
```

## Production Checklist

Before going live:

- [ ] Application runs stably for 24+ hours
- [ ] Auto-start configured and tested
- [ ] Monitoring in place
- [ ] Logs are being collected
- [ ] Backup procedure tested
- [ ] Recovery procedure documented
- [ ] Temperature stays within limits under load
- [ ] All cameras and DRP-AI devices accessible
- [ ] Performance meets requirements
- [ ] Emergency contact information documented

## Support

For production issues:

1. Check logs: `sudo journalctl -u moildev-app`
2. Review monitoring data
3. Consult troubleshooting section
4. Contact Renesas technical support for hardware issues
5. Open GitHub issue for application bugs

---

**Document Version**: 1.0  
**Last Updated**: See git commit history
