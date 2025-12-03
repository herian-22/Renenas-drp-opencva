/**
 * DRP-AI YOLO Implementation
 * Uses configuration from include/define.h strictly.
 */
#include "drpai_yolo.h"
#include "define.h" // <--- WAJIB INCLUDE INI
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring> 
#include <chrono> 
#include <cstdio> 

using namespace std;
using namespace std::chrono; 

// Union untuk konversi bit float
union Float32Bits {
    float f;
    uint32_t u;
};

DrpAiYolo::DrpAiYolo() {
    // Menggunakan INF_OUT_SIZE dari define.h
    output_buffer = new float[INF_OUT_SIZE];
    // Menggunakan MODEL_IN_W/H dari define.h
    cpu_input_buffer = new float[MODEL_IN_W * MODEL_IN_H * 3];
}

DrpAiYolo::~DrpAiYolo() {
    delete[] output_buffer;
    delete[] cpu_input_buffer;
}

bool DrpAiYolo::init(const std::string& model_dir_path, uint64_t start_address) {
    // Parameter model_dir_path diambil dari argumen (yang berasal dari define.h di main)
    bool status = runtime.LoadModel(model_dir_path, start_address);
    if (!status) {
        cerr << "[DRPAI] Failed to load model from: " << model_dir_path << endl;
        return false;
    }
    cout << "[DRPAI] Model loaded: " << model_dir_path << endl;
    cout << "[DRPAI] Thresholds -> Prob: " << TH_PROB << " | NMS: " << TH_NMS << endl;
    return true;
}

std::vector<detection> DrpAiYolo::run_detection(const cv::Mat& input_image, AiStats& stats) {
    auto t_total_start = high_resolution_clock::now();

    // A. PRE-PROCESSING
    auto t_pre_start = high_resolution_clock::now();
    pre_process(input_image);
    auto t_pre_end = high_resolution_clock::now();

    // B. INFERENCE
    auto t_inf_start = high_resolution_clock::now();
    runtime.SetInput(0, cpu_input_buffer);
    runtime.Run(DRPAI_FREQ);
    auto t_inf_end = high_resolution_clock::now();

    // C. POST-PROCESSING
    auto t_post_start = high_resolution_clock::now();
    if (get_result_tensors() != 0) {
        return std::vector<detection>();
    }
    std::vector<detection> result = post_process(input_image.cols, input_image.rows);
    auto t_post_end = high_resolution_clock::now();
    auto t_total_end = high_resolution_clock::now();

    // ISI DATA STATISTIK (Tanpa Print)
    stats.pre   = duration<double, milli>(t_pre_end - t_pre_start).count();
    stats.inf   = duration<double, milli>(t_inf_end - t_inf_start).count();
    stats.post  = duration<double, milli>(t_post_end - t_post_start).count();
    stats.total = duration<double, milli>(t_total_end - t_total_start).count();
    stats.count = (int)result.size();

    return result;
}

void DrpAiYolo::pre_process(const cv::Mat& img) {
    // Padding color 114
    float pad_val = 114.0f / 255.0f;
    for (int i = 0; i < MODEL_IN_W * MODEL_IN_H * 3; i++) {
        cpu_input_buffer[i] = pad_val;
    }

    int w = img.cols;
    int h = img.rows;
    
    // Menggunakan MODEL_IN_W dan MODEL_IN_H dari define.h
    float scale = min((float)MODEL_IN_W / w, (float)MODEL_IN_H / h);
    int new_w = w * scale;
    int new_h = h * scale;
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    int pad_x = (MODEL_IN_W - new_w) / 2;
    int pad_y = (MODEL_IN_H - new_h) / 2;

    float* p_r = cpu_input_buffer;
    float* p_g = cpu_input_buffer + (MODEL_IN_W * MODEL_IN_H);
    float* p_b = cpu_input_buffer + (MODEL_IN_W * MODEL_IN_H * 2);

    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            int idx = (y + pad_y) * MODEL_IN_W + (x + pad_x);
            p_b[idx] = pixel[0] / 255.0f; 
            p_g[idx] = pixel[1] / 255.0f; 
            p_r[idx] = pixel[2] / 255.0f; 
        }
    }
}

int8_t DrpAiYolo::get_result_tensors() {
    int output_num = runtime.GetNumOutput();
    uint32_t size_count = 0;
    
    box_tensor_0 = cls_tensor_0 = nullptr;
    box_tensor_1 = cls_tensor_1 = nullptr;
    box_tensor_2 = cls_tensor_2 = nullptr;

    for (int i = 0; i < output_num; i++) {
        auto output = runtime.GetOutput(i);
        int64_t output_size = std::get<2>(output);
        float* current_ptr = output_buffer + size_count;

        if (std::get<0>(output) == InOutDataType::FLOAT16) {
            uint16_t* data = (uint16_t*)std::get<1>(output);
            for (int j = 0; j < output_size; j++) current_ptr[j] = float16_to_float32(data[j]);
        } else {
            memcpy(current_ptr, std::get<1>(output), output_size * sizeof(float));
        }

        // Mapping menggunakan konstanta SIZE dari define.h
        if (output_size == BOX_TENSOR_SIZE_0) box_tensor_0 = current_ptr;
        else if (output_size == CLS_TENSOR_SIZE_0) cls_tensor_0 = current_ptr;
        else if (output_size == BOX_TENSOR_SIZE_1) box_tensor_1 = current_ptr;
        else if (output_size == CLS_TENSOR_SIZE_1) cls_tensor_1 = current_ptr;
        else if (output_size == BOX_TENSOR_SIZE_2) box_tensor_2 = current_ptr;
        else if (output_size == CLS_TENSOR_SIZE_2) cls_tensor_2 = current_ptr;

        size_count += output_size;
    }
    return 0;
}

std::vector<detection> DrpAiYolo::post_process(int img_w, int img_h) {
    std::vector<detection> det_buff;
    float* boxes[] = { box_tensor_0, box_tensor_1, box_tensor_2 };
    float* classes[] = { cls_tensor_0, cls_tensor_1, cls_tensor_2 };

    for (int i = 0; i < 3; ++i) {
        // Menggunakan grid_sizes dari define.h
        int grid = grid_sizes[i];
        int stride = strides[i];
        int grid_sq = grid * grid;

        for (int y = 0; y < grid; ++y) {
            for (int x = 0; x < grid; ++x) {
                int idx = y * grid + x;
                
                float max_score = -FLT_MAX;
                int max_class = -1;
                // Menggunakan NUM_CLASS dari define.h
                for (int c = 0; c < NUM_CLASS; ++c) {
                    float score = 1.0f / (1.0f + exp(-classes[i][c * grid_sq + idx]));
                    if (score > max_score) { max_score = score; max_class = c; }
                }

                // Menggunakan TH_PROB dari define.h
                if (max_score < TH_PROB) continue;

                float dfl[4];
                for (int k=0; k<4; k++) {
                    float bins[NUM_COORD_BINS];
                    for (int b=0; b<NUM_COORD_BINS; b++) 
                        bins[b] = boxes[i][(k * NUM_COORD_BINS + b) * grid_sq + idx];
                    dfl[k] = dfl_decode(bins);
                }

                float cx = (x + 0.5f - dfl[0]) * stride;
                float cy = (y + 0.5f - dfl[1]) * stride;
                float cx2 = (x + 0.5f + dfl[2]) * stride;
                float cy2 = (y + 0.5f + dfl[3]) * stride;
                
                float center_x = (cx + cx2) / 2.0f;
                float center_y = (cy + cy2) / 2.0f;
                float w_box = cx2 - cx;
                float h_box = cy2 - cy;

                float scale = min((float)MODEL_IN_W / img_w, (float)MODEL_IN_H / img_h);
                int pad_x = (MODEL_IN_W - img_w * scale) / 2;
                int pad_y = (MODEL_IN_H - img_h * scale) / 2;

                center_x = (center_x - pad_x) / scale;
                center_y = (center_y - pad_y) / scale;
                w_box /= scale;
                h_box /= scale;

                Box b = {center_x, center_y, w_box, h_box};
                det_buff.push_back({b, max_class, max_score});
            }
        }
    }

    // Menggunakan TH_NMS dari define.h
    filter_boxes_nms(det_buff, det_buff.size(), TH_NMS);
    return det_buff;
}

float DrpAiYolo::float16_to_float32(uint16_t a) {
    uint16_t sign = (a & 0x8000) >> 15;
    uint16_t exp  = (a & 0x7C00) >> 10;
    uint16_t frac = (a & 0x03FF);

    Float32Bits result;
    
    if (exp == 0) {
        if (frac == 0) {
            result.u = (sign << 31);
        } else {
            float f = (float)frac * powf(2.0f, -24.0f);
            if (sign) f = -f;
            return f;
        }
    } else if (exp == 0x1F) {
        result.u = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        result.u = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    }

    return result.f;
}

float DrpAiYolo::dfl_decode(float* tensor) {
    float softmax[NUM_COORD_BINS], sum = 0, max_val = -FLT_MAX;
    for(int i=0; i<NUM_COORD_BINS; i++) if(tensor[i] > max_val) max_val = tensor[i];
    for(int i=0; i<NUM_COORD_BINS; i++) {
        softmax[i] = exp(tensor[i] - max_val);
        sum += softmax[i];
    }
    float result = 0;
    for(int i=0; i<NUM_COORD_BINS; i++) result += (softmax[i]/sum) * i;
    return result;
}