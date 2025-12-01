cmake_minimum_required(VERSION 3.10)
project(moildev_app CXX C) # Tambahkan C karena kita compile resource .c

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flag Optimasi
add_compile_options(-O3 -pthread -fPIC)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

# --- Deteksi Sysroot (Cross-Compile) ---
# HAPUS link_directories manual untuk sistem lib agar tidak bentrok!
if(DEFINED ENV{SDKTARGETSYSROOT})
    set(CMAKE_SYSROOT $ENV{SDKTARGETSYSROOT})
endif()

# --- Cari Paket ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# --- KONFIGURASI RESOURCE (GResource) ---
find_program(GLIB_COMPILE_RESOURCES NAMES glib-compile-resources REQUIRED)

# Definisikan file resource
set(GRESOURCE_XML "${CMAKE_CURRENT_SOURCE_DIR}/moildev_resources.xml")
set(GRESOURCE_C "${CMAKE_CURRENT_BINARY_DIR}/moildev_resources.c")
set(GRESOURCE_H "${CMAKE_CURRENT_BINARY_DIR}/moildev_resources.h")

# Perintah untuk generate file .c dan .h dari .xml
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

# --- Include & Linking ---
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OPENCV_INCLUDE_DIRS}
    ${GTK3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR} # Agar header resource bisa di-include
)

# Hanya link direktori lokal (libmoildevren.a), JANGAN sistem library
link_directories(${CMAKE_SOURCE_DIR}/lib)

# --- Target Executable ---
# Masukkan ${GRESOURCE_C} agar ikut dikompilasi!
add_executable(moildev_app1
    src/main_drp_moildev.cpp
    src/dmabuf.cpp
    src/moildev_wrapper.cpp
    ${GRESOURCE_C} 
)

target_link_libraries(moildev_app1
    PRIVATE
    moildevren
    mmngr
    mmngrbuf
    ${OPENCV_LIBRARIES}
    ${GTK3_LIBRARIES}
    dl
    rt
    m
    pthread
)