#!/bin/bash

# Dependency Check Script for DRP-AI MOIL Application
# This script verifies that all required dependencies are available

echo "======================================"
echo "  DRP-AI MOIL Dependency Checker"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_count=0
pass_count=0
warn_count=0
fail_count=0

# Function to check if command exists
check_command() {
    local cmd=$1
    local desc=$2
    check_count=$((check_count + 1))
    
    if command -v $cmd &> /dev/null; then
        echo -e "${GREEN}✓${NC} $desc: $(which $cmd)"
        pass_count=$((pass_count + 1))
        return 0
    else
        echo -e "${RED}✗${NC} $desc: NOT FOUND"
        fail_count=$((fail_count + 1))
        return 1
    fi
}

# Function to check environment variable
check_env() {
    local var=$1
    local desc=$2
    check_count=$((check_count + 1))
    
    if [ -n "${!var}" ]; then
        echo -e "${GREEN}✓${NC} $desc: ${!var}"
        pass_count=$((pass_count + 1))
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $desc: NOT SET"
        warn_count=$((warn_count + 1))
        return 1
    fi
}

# Function to check pkg-config package
check_pkg() {
    local pkg=$1
    local desc=$2
    check_count=$((check_count + 1))
    
    if pkg-config --exists $pkg 2>/dev/null; then
        local ver=$(pkg-config --modversion $pkg 2>/dev/null)
        echo -e "${GREEN}✓${NC} $desc: $ver"
        pass_count=$((pass_count + 1))
        return 0
    else
        echo -e "${RED}✗${NC} $desc: NOT FOUND"
        fail_count=$((fail_count + 1))
        return 1
    fi
}

echo "=== Build Tools ==="
check_command cmake "CMake"
check_command make "Make"
check_command g++ "G++ Compiler"
check_command pkg-config "pkg-config"
echo ""

echo "=== SDK Environment ==="
check_env SDKTARGETSYSROOT "Poky SDK Sysroot"
check_env TVM_HOME "TVM Runtime Path"
echo ""

echo "=== Required Libraries ==="
check_pkg opencv4 "OpenCV 4.x"
check_pkg gtk+-3.0 "GTK+ 3.0"
echo ""

echo "=== Optional Tools ==="
check_command glib-compile-resources "GLib Resource Compiler"
check_command v4l2-ctl "V4L2 Utils"
check_command git "Git"
echo ""

echo "=== Device Checks ==="
check_count=$((check_count + 1))
if [ -e /dev/drpai0 ]; then
    echo -e "${GREEN}✓${NC} DRP-AI device: /dev/drpai0 exists"
    ls -l /dev/drpai0
    pass_count=$((pass_count + 1))
else
    echo -e "${YELLOW}⚠${NC} DRP-AI device: /dev/drpai0 not found (OK if checking on development machine)"
    warn_count=$((warn_count + 1))
fi

check_count=$((check_count + 1))
if ls /dev/video* 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Camera devices found:"
    ls -l /dev/video*
    pass_count=$((pass_count + 1))
else
    echo -e "${YELLOW}⚠${NC} No camera devices found (OK if checking on development machine)"
    warn_count=$((warn_count + 1))
fi
echo ""

echo "=== Project Files ==="
check_count=$((check_count + 1))
if [ -f "CMakeLists.txt" ]; then
    echo -e "${GREEN}✓${NC} CMakeLists.txt exists"
    pass_count=$((pass_count + 1))
else
    echo -e "${RED}✗${NC} CMakeLists.txt not found"
    fail_count=$((fail_count + 1))
fi

check_count=$((check_count + 1))
if [ -f "include/define.h" ]; then
    echo -e "${GREEN}✓${NC} include/define.h exists"
    pass_count=$((pass_count + 1))
else
    echo -e "${RED}✗${NC} include/define.h not found"
    fail_count=$((fail_count + 1))
fi

check_count=$((check_count + 1))
if [ -f "lib/libmoildevren.a" ]; then
    echo -e "${GREEN}✓${NC} libmoildevren.a library exists"
    pass_count=$((pass_count + 1))
else
    echo -e "${RED}✗${NC} lib/libmoildevren.a not found"
    fail_count=$((fail_count + 1))
fi
echo ""

echo "=== Model Files ==="
# Check for model directory (from define.h)
if [ -f "include/define.h" ]; then
    MODEL_DIR=$(grep 'const std::string model_dir' include/define.h | sed 's/.*"\(.*\)".*/\1/')
    check_count=$((check_count + 1))
    if [ -d "$MODEL_DIR" ]; then
        echo -e "${GREEN}✓${NC} Model directory '$MODEL_DIR' exists"
        pass_count=$((pass_count + 1))
        
        # Check for required model files
        check_count=$((check_count + 1))
        if [ -f "$MODEL_DIR/deploy.so" ] && [ -f "$MODEL_DIR/deploy.json" ] && [ -f "$MODEL_DIR/deploy.params" ]; then
            echo -e "${GREEN}✓${NC} Model files (deploy.so, deploy.json, deploy.params) exist"
            pass_count=$((pass_count + 1))
        else
            echo -e "${YELLOW}⚠${NC} Some model files missing in $MODEL_DIR"
            warn_count=$((warn_count + 1))
        fi
    else
        echo -e "${YELLOW}⚠${NC} Model directory '$MODEL_DIR' not found (you'll need to provide your model)"
        warn_count=$((warn_count + 1))
    fi
    
    # Check for label file
    LABEL_FILE=$(grep 'const std::string label_list' include/define.h | sed 's/.*"\(.*\)".*/\1/')
    check_count=$((check_count + 1))
    if [ -f "$LABEL_FILE" ]; then
        echo -e "${GREEN}✓${NC} Label file '$LABEL_FILE' exists"
        pass_count=$((pass_count + 1))
    else
        echo -e "${YELLOW}⚠${NC} Label file '$LABEL_FILE' not found (you'll need to create it)"
        warn_count=$((warn_count + 1))
    fi
fi
echo ""

echo "======================================"
echo "  Summary"
echo "======================================"
echo "Total checks: $check_count"
echo -e "${GREEN}Passed:  $pass_count${NC}"
echo -e "${YELLOW}Warnings: $warn_count${NC}"
echo -e "${RED}Failed:  $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    if [ $warn_count -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! Ready to build.${NC}"
        echo ""
        echo "Run './build.sh' to build the application."
        exit 0
    else
        echo -e "${YELLOW}⚠ Some warnings detected.${NC}"
        echo "You may need to provide model files or configure devices."
        echo ""
        echo "Run './build.sh' to build the application."
        exit 0
    fi
else
    echo -e "${RED}✗ Some critical dependencies are missing.${NC}"
    echo ""
    echo "Please install missing dependencies:"
    echo "  - Source Poky SDK: source /opt/poky/.../environment-setup-..."
    echo "  - Set TVM_HOME: export TVM_HOME=/path/to/tvm"
    echo "  - Install OpenCV: See README.md"
    echo "  - Install GTK+: See README.md"
    exit 1
fi
