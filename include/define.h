#pragma once

#include <string>

/* Model DRP-AI (Direktori tempat Anda meletakkan file .so, .json, .params) */

const std::string model_dir = "unicorn"; // Nama folder model Anda

/* File teks label (satu nama kelas per baris) */

const std::string label_list = "unicorn.txt";

/* Jumlah kelas yang dilatih model Anda */
#define NUM_CLASS (5) 

/* Dimensi Input Model */
#define MODEL_IN_W (640)
#define MODEL_IN_H (640)

/* Ambang batas (Thresholds) */

#define TH_PROB (0.25f) // Turunkan ke standar YOLO
#define TH_NMS (0.45f)  // Biarkan standar

/*****************************************
* Definisi Aplikasi Kamera & Tampilan
******************************************/

#define DRPAI_IN_WIDTH (640)  // Ubah dari 1280
#define DRPAI_IN_HEIGHT (480) // Ubah dari 720
#define IMAGE_OUTPUT_WIDTH (1280) // Display bisa tetap 1280x720
#define IMAGE_OUTPUT_HEIGHT (720)

/* Pengaturan Kamera (GStreamer/OpenCV) */
#define CAM_IMAGE_WIDTH (DRPAI_IN_WIDTH)
#define CAM_IMAGE_HEIGHT (DRPAI_IN_HEIGHT)
#define CAM_IMAGE_CHANNEL_BGR (3) 
#define IMAGE_OUTPUT_CHANNEL_BGRA (4) 
#define IMAGE_CHANNEL_BGR (3) 
#define CAPTURE_STABLE_COUNT (8) 

/* Nama Input Kamera (Akan dideteksi otomatis, tapi ini sbg placeholder) */
#define INPUT_CAM_NAME "/dev/video0"
#define DRPAI_FREQ (2) 

/* Pengaturan Teks & Kotak */
#define LINE_HEIGHT (30)
#define LINE_HEIGHT_OFFSET (20)
#define TEXT_WIDTH_OFFSET (10)
#define TEXT_START_X (10)
#define CHAR_SCALE_LARGE (0.8f)
#define CHAR_SCALE_SMALL (0.7f)
#define CHAR_SCALE_FONT (0.6f)
#define CHAR_SCALE_VERY_SMALL (0.5f)
#define CHAR_THICKNESS (2)
#define CHAR_THICKNESS_BOX (2)
#define BOX_LINE_SIZE (2)
#define BOX_HEIGHT_OFFSET (30)
#define BOX_TEXT_HEIGHT_OFFSET (5)
#define ALIGHN_LEFT (1)
#define ALIGHN_RIGHT (2)

#define BLACK_DATA (0x000000u)
#define WHITE_DATA (0xFFFFFFu)
#define GREEN_DATA (0x00FF00u)

/* Pengaturan Thread & Waktu */
#define WAIT_TIME (1000) // 1ms
#define TIME_COEF (1.0)
#define DISPLAY_THREAD_TIMEOUT (5)
#define CAPTURE_TIMEOUT (5)
#define AI_THREAD_TIMEOUT (5)
#define KEY_THREAD_TIMEOUT (5)
#define DRPAI_TIMEOUT (5)
#define IMAGE_THREAD_TIMEOUT (5)

/* Makro Helper */
#define SIZE_OF_ARRAY(array) (sizeof(array) / sizeof(array[0]))
#define NUM_BOX_COLORS 3
static uint32_t box_color[] =
{
    0xFF0000u, // Merah 
    0x00FF00u, // Hijau
    0x0000FFu, // Biru
};

/*****************************************
* Definisi Internal YOLOv8 (Dari file lama Anda)
******************************************/
#define STRIDE_0 (8)
#define STRIDE_1 (16)
#define STRIDE_2 (32)
const int strides[] = {STRIDE_0, STRIDE_1, STRIDE_2};

#define GRID_SIZE_0 (MODEL_IN_W / STRIDE_0)
#define GRID_SIZE_1 (MODEL_IN_W / STRIDE_1)
#define GRID_SIZE_2 (MODEL_IN_W / STRIDE_2)
const int grid_sizes[] = {GRID_SIZE_0, GRID_SIZE_1, GRID_SIZE_2};

#define NUM_COORD_BINS (16)
#define BOX_HEAD_CHANNELS (4 * NUM_COORD_BINS)

#define NUM_CLASS_YOLOV8 (NUM_CLASS) 

#define BOX_TENSOR_SIZE_0 (BOX_HEAD_CHANNELS * GRID_SIZE_0 * GRID_SIZE_0)
#define CLS_TENSOR_SIZE_0 (NUM_CLASS_YOLOV8 * GRID_SIZE_0 * GRID_SIZE_0)
#define BOX_TENSOR_SIZE_1 (BOX_HEAD_CHANNELS * GRID_SIZE_1 * GRID_SIZE_1)
#define CLS_TENSOR_SIZE_1 (NUM_CLASS_YOLOV8 * GRID_SIZE_1 * GRID_SIZE_1)
#define BOX_TENSOR_SIZE_2 (BOX_HEAD_CHANNELS * GRID_SIZE_2 * GRID_SIZE_2)
#define CLS_TENSOR_SIZE_2 (NUM_CLASS_YOLOV8 * GRID_SIZE_2 * GRID_SIZE_2)

#define TOTAL_OUTPUT_ELEMENTS (BOX_TENSOR_SIZE_0 + CLS_TENSOR_SIZE_0 + \
                             BOX_TENSOR_SIZE_1 + CLS_TENSOR_SIZE_1 + \
                             BOX_TENSOR_SIZE_2 + CLS_TENSOR_SIZE_2)

#define INF_OUT_SIZE (TOTAL_OUTPUT_ELEMENTS) // Ukuran output inferensi


/* Definisi lain  */
#define DEBUG_TIME_FLG
#define DISP_CAM_FRAME_RATE // Aktifkan tampilan FPS Kamera

// Definisi untuk DRP-AI Inference
#define DRPAI_OUT_WIDTH (IMAGE_OUTPUT_WIDTH)
#define DRPAI_OUT_HEIGHT (IMAGE_OUTPUT_HEIGHT)
#define NUM_INF_OUT_LAYER (3) 
#define NUM_BB (3) 
#define ANCHORS {10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326} 
const float anchors[] = ANCHORS;
const int num_grids[] = {GRID_SIZE_0, GRID_SIZE_1, GRID_SIZE_2}; 