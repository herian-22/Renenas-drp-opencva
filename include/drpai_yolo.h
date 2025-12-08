#ifndef DRPAI_YOLO_H
#define DRPAI_YOLO_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MeraDrpRuntimeWrapper.h"
#include "box.h"     
#include "define.h"  

// Struktur untuk menyimpan detail waktu AI
struct AiStats {
    double pre;
    double inf;
    double post;
    double total;
    int count;
};

class DrpAiYolo {
public:
    DrpAiYolo();
    ~DrpAiYolo();

    bool init(const std::string& model_dir, uint64_t start_address = 0);
    
    // Update: Menambahkan parameter output AiStats
    std::vector<detection> run_detection(const cv::Mat& input_image, AiStats& stats);

private:
    MeraDrpRuntimeWrapper runtime;
    float* output_buffer; 
    float* cpu_input_buffer;

    float* box_tensor_0; float* cls_tensor_0;
    float* box_tensor_1; float* cls_tensor_1;
    float* box_tensor_2; float* cls_tensor_2;

    void pre_process(const cv::Mat& img);
    int8_t get_result_tensors();
    std::vector<detection> post_process(int img_w, int img_h);
    float dfl_decode(float* tensor);
    float float16_to_float32(uint16_t a);
};

#endif