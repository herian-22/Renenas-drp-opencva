#include "image.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm> // Untuk std::max, std::min
#include <cmath> // Untuk round

Image::Image() {}
Image::~Image() {}

uint8_t Image::init(uint32_t w, uint32_t h, uint32_t c, uint32_t ow, uint32_t oh, uint32_t oc) {
    img_w = w; img_h = h; img_c = c;
    out_w = ow; out_h = oh; out_c = oc;
    
    // Inisialisasi mat dengan ukuran output, 3 channel (BGR)
    // cv::imshow bekerja paling baik dengan 3-channel BGR
    image_mat = cv::Mat::zeros(oh, ow, CV_8UC3); 
    if (image_mat.empty()) {
        return 1; // Gagal menginisialisasi Mat
    }
    return 0;
}

void Image::set_mat(cv::Mat& mat) {
    // Salin data dari capture_image ke image_mat
    // Pastikan channel-nya konsisten (GStreamer biasanya BGR, 3 channel)
    if (mat.channels() == 3) {
        mat.copyTo(image_mat);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, image_mat, cv::COLOR_BGRA2BGR);
    } else {
        // Handle kasus lain jika perlu
        mat.copyTo(image_mat);
    }
}

cv::Mat& Image::get_mat() {
    return image_mat;
}

void Image::convert_size(int in_w_unused, int resize_w_unused, bool is_padding) {
    // Parameter yang dilewatkan (in_w, resize_w) diabaikan
    // Kita dapatkan dimensi asli langsung dari Mat
    int real_in_w = image_mat.cols; // Ini 640
    int real_in_h = image_mat.rows; // Ini 480

    if (real_in_w == 0 || real_in_h == 0) {
        // Pengaman jika Mat kosong
        return;
    }

    // Hitung rasio untuk lebar dan tinggi
    float r_w = (float)out_w / (float)real_in_w; // 1280 / 640 = 2.0
    float r_h = (float)out_h / (float)real_in_h; // 720 / 480 = 1.5

    // Ambil rasio terkecil agar gambar muat di dalam layar
    float ratio = std::min(r_w, r_h); // ratio akan menjadi 1.5

    // Hitung dimensi resize yang baru
    int resize_w = (int)round(real_in_w * ratio); // 640 * 1.5 = 960
    int resize_h = (int)round(real_in_h * ratio); // 480 * 1.5 = 720

    cv::Mat resized_img;
    // Resize gambar. Ganti INTER_NEAREST ke INTER_LINEAR untuk kualitas lebih baik
    if ( image_mat.cols != resize_w || image_mat.rows != resize_h ) {
        cv::resize(image_mat, resized_img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized_img = image_mat;
    }

    if ( is_padding ) {
        // Hitung padding (sekarang nilainya akan positif)
        uint32_t pad_top = (out_h - resize_h) / 2;     // (720 - 720) / 2 = 0
        uint32_t pad_bottom = out_h - resize_h - pad_top; // 0
        uint32_t pad_left = (out_w - resize_w) / 2;     // (1280 - 960) / 2 = 160
        uint32_t pad_right = out_w - resize_w - pad_left; // 160
        
        // Pad dengan border hitam (cv::Scalar(B, G, R))
        // Panggilan ini sekarang aman
        cv::copyMakeBorder(resized_img, image_mat, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        image_mat = resized_img;
    }
}

void Image::draw_rect(int32_t x, int32_t y, int32_t w, int32_t h, const char* str, uint32_t color, uint32_t label_color) {
    int32_t x_min = x - round(w / 2.);
    int32_t y_min = y - round(h / 2.);
    int32_t x_max = x + round(w / 2.) - 1;
    int32_t y_max = y + round(h / 2.) - 1;

    // Pastikan koordinat berada di dalam batas gambar
    x_min = std::max(0, x_min);
    y_min = std::max(0, y_min);
    x_max = std::min(image_mat.cols - 1, x_max);
    y_max = std::min(image_mat.rows - 1, y_max);

    // Konversi warna 0xRRGGBB ke cv::Scalar (BGR)
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;
    cv::Scalar box_scalar(b, g, r);

    // Gambar kotak
    cv::rectangle(image_mat, cv::Point(x_min, y_min), cv::Point(x_max, y_max), box_scalar, BOX_LINE_SIZE);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(str, cv::FONT_ITALIC, CHAR_SCALE_FONT, CHAR_THICKNESS_BOX, &baseline);
    
    // Pastikan kotak label tidak keluar dari atas gambar
    int label_y_min = std::max(y_min - text_size.height - BOX_TEXT_HEIGHT_OFFSET, 0);
    int label_y_max = std::max(y_min, text_size.height); // y_min

    // Gambar latar belakang teks
    cv::rectangle(image_mat, 
                  cv::Point(x_min, label_y_min), 
                  cv::Point(x_min + text_size.width, label_y_max), 
                  box_scalar, cv::FILLED);

    // Konversi warna label 0xRRGGBB ke cv::Scalar (BGR)
    uint8_t lr = (label_color >> 16) & 0xFF;
    uint8_t lg = (label_color >> 8) & 0xFF;
    uint8_t lb = label_color & 0xFF;
    cv::Scalar label_scalar(lb, lg, lr);

    // Gambar teks
    cv::putText(image_mat, str, 
                cv::Point(x_min, y_min - BOX_TEXT_HEIGHT_OFFSET), 
                cv::FONT_ITALIC, CHAR_SCALE_FONT, label_scalar, CHAR_THICKNESS_BOX);
}

void Image::write_string_rgb(std::string str, uint32_t align_type, uint32_t x, uint32_t y, float size, uint32_t color) {
    // Konversi warna 0xRRGGBB ke cv::Scalar (BGR)
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;
    cv::Scalar text_color(b, g, r);
    cv::Scalar shadow_color(0, 0, 0); // Hitam untuk bayangan

    int ptx = 0;
    int pty = 0;

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(str.c_str(), cv::FONT_HERSHEY_SIMPLEX, size, CHAR_THICKNESS, &baseline);
    
    if (align_type == ALIGHN_LEFT) { // Kiri
        ptx = x;
        pty = y;
    } else if (align_type == ALIGHN_RIGHT) { // Kanan
        ptx = out_w - (text_size.width + x);
        pty = y;
    } else { // Default ke Kiri
         ptx = x;
         pty = y;
    }

    // Gambar bayangan (outline) terlebih dahulu
    cv::putText(image_mat, str.c_str(), cv::Point(ptx, pty), cv::FONT_HERSHEY_SIMPLEX, size, shadow_color, CHAR_THICKNESS + 2);
    // Gambar teks di atasnya
    cv::putText(image_mat, str.c_str(), cv::Point(ptx, pty), cv::FONT_HERSHEY_SIMPLEX, size, text_color, CHAR_THICKNESS);
}