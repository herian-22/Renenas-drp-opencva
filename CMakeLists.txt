cmake_minimum_required(VERSION 3.10)
project(moildev_app CXX)

# --- 1. Konfigurasi Compiler & Flags ---
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flag Optimasi & Threading (Sama seperti di build.sh)
add_compile_options(-O3 -pthread -fPIC)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

# --- 2. Deteksi Lingkungan Poky (Sysroot) ---
# Ini otomatis mengambil path dari environment variable SDKTARGETSYSROOT
if(DEFINED ENV{SDKTARGETSYSROOT})
    message(STATUS "Detected Cross-Compilation Environment")
    message(STATUS "SYSROOT: $ENV{SDKTARGETSYSROOT}")
    set(CMAKE_SYSROOT $ENV{SDKTARGETSYSROOT})
    
    # Tambahan path library sistem jika diperlukan
    link_directories(
        $ENV{SDKTARGETSYSROOT}/usr/lib 
        $ENV{SDKTARGETSYSROOT}/usr/lib64
    )
endif()

# --- 3. Cari Paket (PkgConfig) ---
find_package(PkgConfig REQUIRED)

# Cari OpenCV (Menggunakan pkg-config agar include path & libs otomatis benar)
# Ini akan menarik: core, imgproc, videoio, highgui, calib3d, features2d, flann, dll.
pkg_check_modules(OPENCV REQUIRED opencv4)

# Cari GTK+ 3.0 (Menggunakan pkg-config untuk cflags gtk, gdk, pango, atk, cairo)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# --- 4. Konfigurasi Include ---
include_directories(
    ${CMAKE_SOURCE_DIR}/include       # Header lokal (moildev.hpp, dmabuf.h, wrapper)
    ${OPENCV_INCLUDE_DIRS}            # Header OpenCV
    ${GTK3_INCLUDE_DIRS}              # Header GTK
)

# --- 5. Konfigurasi Library Path ---
link_directories(
    ${CMAKE_SOURCE_DIR}/lib           # Lokasi libmoildevren.a
)

# --- 6. Definisi Target Executable ---
# Gabungkan semua source code menjadi satu aplikasi
add_executable(moildev_app1
    src/main_drp_moildev.cpp
    src/dmabuf.cpp
    src/moildev_wrapper.cpp
)

# --- 7. Linking Library ---
target_link_libraries(moildev_app1
    PRIVATE
    moildevren                        # libmoildevren.a (Static)
    mmngr                             # Renesas Memory Manager
    mmngrbuf                          # Renesas Memory Buffer
    ${OPENCV_LIBRARIES}               # Semua modul OpenCV
    ${GTK3_LIBRARIES}                 # Semua modul GTK
    dl
    rt
    m
    pthread
)

# --- 8. Post-Build: Copy Asset ---
# Otomatis copy gui.glade ke folder build agar siap dijalankan
file(COPY ${CMAKE_SOURCE_DIR}/asset/gui.glade DESTINATION ${CMAKE_BINARY_DIR})

message(STATUS "Build Configured. Output will be in: ${CMAKE_BINARY_DIR}")