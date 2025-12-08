cmake_minimum_required(VERSION 3.10)
project(moildev_app+yolo CXX C)

# Update ke C++17 agar kompatibel dengan library DRP-AI/TVM modern
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flag Optimasi & OpenMP (Manual Force)
# Kita tambahkan -fopenmp di sini agar tidak perlu find_package yang sering gagal di Cross-Compile
add_compile_options(
    -O3
    -pthread
    -fPIC
    -fopenmp
    -mcpu=cortex-a55       # Spesifik prosesor (sudah otomatis aktifkan NEON)
    -mtune=cortex-a55      # Tuning instruksi
    -ffast-math            # PERCEPAT sin/cos (Wajib untuk performa matematika)
    -ftree-vectorize       # Paksa vektorisasi loop
)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -fopenmp")

# === FIX: MANUAL INCLUDE OPENMP (POKY SDK) ===
# Compiler tidak otomatis menemukan omp.h, jadi kita tambahkan manual path-nya.
# Path ini diambil dari hasil 'find' Anda: /opt/poky/.../include/omp.h
include_directories("/opt/poky/3.1.31/sysroots/aarch64-poky-linux/usr/lib64/aarch64-poky-linux/8.3.0/include")

# ==========================================
# 1. KONFIGURASI TVM & DRP-AI (BARU)
# ==========================================
set(TVM_ROOT $ENV{TVM_HOME})
if(NOT DEFINED ENV{TVM_HOME})
    message(WARNING "TVM_HOME environment variable not set. Please export TVM_HOME.")
endif()

# Lokasi Library Runtime TVM (Sesuaikan jika path build anda berbeda)
set(TVM_RUNTIME_LIB ${TVM_ROOT}/build_runtime/libtvm_runtime.so)

# Cek Versi SDK (V2H vs V2N/Lainnya) untuk DRP-AI
if(NOT EXISTS "$ENV{SDK}/sysroots/cortexa55-poky-linux")
    add_definitions(-DV2H) # Definisi untuk RZ/V2H
else()
    add_definitions(-DV2H -DV2N)
endif()

# ==========================================
# 2. KONFIGURASI UMUM (SAMA SEPERTI LAMA)
# ==========================================

# --- Deteksi Sysroot ---
if(DEFINED ENV{SDKTARGETSYSROOT})
    set(CMAKE_SYSROOT $ENV{SDKTARGETSYSROOT})
endif()

# --- Cari Paket ---
find_package(PkgConfig REQUIRED)
# HAPUS/COMMENT: find_package(OpenMP) sering gagal di sysroot. Kita sudah paksa flag di atas.
# find_package(OpenMP REQUIRED COMPONENTS CXX) 
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# --- KONFIGURASI RESOURCE (GResource) ---
find_program(GLIB_COMPILE_RESOURCES NAMES glib-compile-resources REQUIRED)

set(GRESOURCE_XML "${CMAKE_CURRENT_SOURCE_DIR}/moildev_resources.xml")
set(GRESOURCE_C "${CMAKE_CURRENT_BINARY_DIR}/moildev_resources.c")
set(GRESOURCE_H "${CMAKE_CURRENT_BINARY_DIR}/moildev_resources.h")

add_custom_command(
    OUTPUT ${GRESOURCE_C} ${GRESOURCE_H}
    COMMAND ${GLIB_COMPILE_RESOURCES}
            ${GRESOURCE_XML}
            --generate-source
            --target=${GRESOURCE_C}
            --c-name moildev_app
            --sourcedir ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GLIB_COMPILE_RESOURCES}
            ${GRESOURCE_XML}
            --generate-header
            --target=${GRESOURCE_H}
            --c-name moildev_app
            --sourcedir ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${GRESOURCE_XML}
    COMMENT "Generating Resource Files"
)

# ==========================================
# 3. INCLUDE DIRECTORIES (GABUNGAN)
# ==========================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OPENCV_INCLUDE_DIRS}
    ${GTK3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR} # Untuk header resource
    # --- Tambahan TVM ---
    ${TVM_ROOT}/include
    ${TVM_ROOT}/../apps
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${TVM_ROOT}/3rdparty/compiler-rt
)

# Direktori library lokal (libmoildevren.a)
link_directories(${CMAKE_SOURCE_DIR}/lib)

# ==========================================
# 4. TARGET EXECUTABLE (GABUNGAN SOURCE)
# ==========================================
add_executable(moildev_app+DrpAiYolo
    # File Utama & Moildev
    src/main_drp_moildev.cpp
    src/dmabuf.cpp
    src/moildev_wrapper.cpp
    
    # File DRP-AI / YOLO (BARU)
    src/drpai_yolo.cpp             # Wrapper Class AI
    src/MeraDrpRuntimeWrapper.cpp  # Library Runtime Renesas
    src/box.cpp                    # Helper NMS/Box
    
    # File Resource GUI
    ${GRESOURCE_C} 
)

# Tidak perlu target_link_libraries OpenMP::OpenMP_CXX karena sudah di-set di CMAKE_EXE_LINKER_FLAGS
# if(OpenMP_CXX_FOUND)
#    target_link_libraries(moildev_app+DrpAiYolo PUBLIC OpenMP::OpenMP_CXX)
# endif()

# ==========================================
# 5. LINKING LIBRARIES (GABUNGAN)
# ==========================================
target_link_libraries(moildev_app+DrpAiYolo
    PRIVATE
    moildevren          # Lib Moildev Fisheye
    mmngr               # Memory Manager DRP
    mmngrbuf
    ${OPENCV_LIBRARIES} # OpenCV
    ${GTK3_LIBRARIES}   # GTK GUI
    ${TVM_RUNTIME_LIB}  # Lib TVM DRP-AI (BARU)
    dl
    rt
    m
    pthread
)