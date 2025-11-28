#!/bin/bash

# Load Yocto environment
source /opt/poky/3.1.31/environment-setup-aarch64-poky-linux

CXX=aarch64-poky-linux-g++
SYSROOT=$SDKTARGETSYSROOT

# OpenCV libraries (sesuaikan jika perlu)
OPENCV_LIBS="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio"

echo "=== Compiling Wrapper (moildev_wrapper.cpp) ==="
$CXX -O3 -c moildev_wrapper.cpp -o moildev_wrapper.o \
    --sysroot="${SYSROOT}" \
    -I"${SYSROOT}/usr/include/opencv4" \
    -I"${SYSROOT}/usr/include/glib-2.0" \
    -I"${SYSROOT}/usr/lib64/glib-2.0/include" \
    -I"${SYSROOT}/usr/include/gtk-3.0" \
    -I"${SYSROOT}/usr/include/atk-1.0" \
    -I"${SYSROOT}/usr/include/pango-1.0" \
    -I"${SYSROOT}/usr/include/cairo" \
    -I"${SYSROOT}/usr/include/gdk-pixbuf-2.0" \
    || { echo "Wrapper Build Failed"; exit 1; }

echo "=== Compiling Main (main_drp_moildev_lib.cpp, dmabuf.cpp) & Linking ==="
$CXX main_drp_moildev_lib.cpp dmabuf.cpp moildev_wrapper.o -o moildev_app1 \
    --sysroot="${SYSROOT}" \
    -O3 -pthread -rdynamic \
    -I"${SYSROOT}/usr/include/opencv4" \
    -I"${SYSROOT}/usr/include/gtk-3.0" \
    -I"${SYSROOT}/usr/include/atk-1.0" \
    -I"${SYSROOT}/usr/include/pango-1.0" \
    -I"${SYSROOT}/usr/include/cairo" \
    -I"${SYSROOT}/usr/include/gdk-pixbuf-2.0" \
    -I"${SYSROOT}/usr/include/glib-2.0" \
    -I"${SYSROOT}/usr/lib64/glib-2.0/include" \
    -L"${SYSROOT}/usr/lib" \
    -L"${SYSROOT}/lib" \
    -L"./" -lmoildevren -lmmngr -lmmngrbuf \
    $OPENCV_LIBS \
    -lgtk-3 -lgdk-3 -lgobject-2.0 -latk-1.0 -lpango-1.0 -lcairo -lgdk_pixbuf-2.0 -lglib-2.0 \
    -ldl -lpthread -lrt -lm \
    || { echo "Main Build Failed"; exit 1; }

echo "Build Succeeded."
