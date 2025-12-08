#pragma once

#include "define.h"
#include <opencv2/opencv.hpp>
#include <string>

/* Kelas Image ini dirancang untuk bekerja dengan cv::Mat,
 * menggantikan implementasi buffer manual/Wayland sebelumnya. */
class Image {
public:
    Image();
    ~Image();
    
    /**
     * @brief Inisialisasi properti gambar.
     * @param w Lebar input (kamera)
     * @param h Tinggi input (kamera)
     * @param c Channel input (kamera, misal BGR=3)
     * @param ow Lebar output (display)
     * @param oh Tinggi output (display)
     * @param oc Channel output (display)
     * @return 0 jika sukses
     */
    uint8_t init(uint32_t w, uint32_t h, uint32_t c, uint32_t ow, uint32_t oh, uint32_t oc);
    
    /**
     * @brief Menyalin cv::Mat eksternal (dari capture thread) ke buffer internal kelas ini.
     * @param mat Gambar input (biasanya dari cv::VideoCapture::read)
     */
    void set_mat(cv::Mat& mat);

    /**
     * @brief Mendapatkan referensi ke cv::Mat internal untuk diproses.
     * @return Referensi ke cv::Mat
     */
    cv::Mat& get_mat();
    
    /**
     * @brief Mengubah ukuran dan menambahkan padding ke gambar internal agar sesuai dengan resolusi output.
     * @param in_w Lebar gambar input (sebelum resize)
     * @param resize_w Lebar target untuk resize (sebelum padding)
     * @param is_padding Jika true, tambahkan padding hitam agar pas di output.
     */
    void convert_size(int in_w, int resize_w, bool is_padding);

    /**
     * @brief Menggambar kotak pembatas dan label di atas gambar internal (image_mat).
     * @param x Koordinat X tengah
     * @param y Koordinat Y tengah
     * @param w Lebar kotak
     * @param h Tinggi kotak
     * @param str Teks label
     * @param color Warna kotak (format 0xRRGGBB)
     * @param label_color Warna teks (format 0xRRGGBB)
     */
    void draw_rect(int32_t x, int32_t y, int32_t w, int32_t h, const char* str, uint32_t color, uint32_t label_color);

    /**
     * @brief Menulis teks pada gambar internal (image_mat).
     * @param str Teks
     * @param align_type Tipe perataan (1: Kiri, 2: Kanan)
     * @param x Koordinat X
     * @param y Koordinat Y (baseline)
     * @param size Ukuran font
     * @param color Warna teks (format 0xRRGGBB)
     */
    void write_string_rgb(std::string str, uint32_t align_type, uint32_t x, uint32_t y, float size, uint32_t color);

private:
    // Buffer gambar utama sekarang adalah objek cv::Mat
    cv::Mat image_mat; 
    
    uint32_t img_w, img_h, img_c;
    uint32_t out_w, out_h, out_c;
};