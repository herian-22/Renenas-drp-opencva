#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <chrono> 
#include <sys/ioctl.h>
#include <sys/mman.h> 
#include <linux/drpai.h>
#include <fstream> 
#include <signal.h>
#include <sys/stat.h>


#include "dmabuf.h" 
#include "drpai_yolo.h" 
#include "box.h"
#include "define.h"



extern "C" {
#include "moildev_resources.h" 
}
using namespace cv;
using namespace std;

// === KONFIGURASI INPUT ===
int DEFAULT_W = 800;
int DEFAULT_H = 600;
int IN_W = DEFAULT_W; 
int IN_H = DEFAULT_H;

int OUT_W = 1024, OUT_H = 600; 
int SEL_RES_IDX = 0;

struct Resolution { int w; int h; string name; };

const vector<Resolution> BENCHMARK_RES = {
    {640, 480,  "VGA (480p)"},
    {1024, 768,  "0.8 MP (HD)"},    
    {1600, 1200, "2.0 MP (FHD)"},   
    {1920, 1080, "2.0 MP (1080p)"}, 
    {2592, 1944, "5.0 MP (2K/5MP)"},
    
};

// Fisheye Params
const double PARA[] = {0.0, 0.0, -35.475, 73.455, -41.392, 499.33};
const double ORG_W = 2592.0, ORG_H = 1944.0; 
const double ORG_ICX = 1205.0, ORG_ICY = 966.0;

// State
enum ViewMode { MODE_MAIN, MODE_ORIGINAL, MODE_PANORAMA, MODE_GRID, MODE_SINGLE, MODE_TRIPLE };
atomic<ViewMode> current_mode(MODE_MAIN);
atomic<int> active_map_idx(0);

// --- VARIABEL PTZ & CONFIG ---
atomic<float> cur_alpha(0.0f), cur_beta(0.0f), cur_zoom(2.0f);
atomic<bool> cur_flip_h(false), cur_flip_v(false);

atomic<float> pano_alpha(110.0f), pano_beta(0.0f), pano_zoom(1.0f);
atomic<bool> pano_flip_h(false), pano_flip_v(false);

atomic<float> view1_alpha(-45.0f), view1_beta(0.0f), view1_zoom(3.5f); 
atomic<bool>  view1_flip_h(false), view1_flip_v(false);

atomic<float> view2_alpha(0.0f),   view2_beta(0.0f), view2_zoom(3.5f);
atomic<bool>  view2_flip_h(false), view2_flip_v(false);

atomic<float> view3_alpha(45.0f),  view3_beta(0.0f), view3_zoom(3.5f);
atomic<bool>  view3_flip_h(false), view3_flip_v(false);

atomic<int> active_view_id(0); 

atomic<bool> isRunning(true);
atomic<bool> is_paused(false);
atomic<bool> ui_update_pending(false);
atomic<bool> maps_ready(false);

GtkWidget *main_window, *img_display;

// Stats

atomic<double> stats_read_ms(0.0);   
atomic<double> stats_cvt_ms(0.0);    
atomic<double> stats_ai_ms(0.0);     
atomic<double> stats_remap_ms(0.0);  
atomic<double> stats_draw_ms(0.0);   
atomic<double> stats_fps(0.0);     
atomic<double> stats_map_ms(0.0); 

// Buffers
// BUFFER_SIZE = 3 untuk Double buffering yang aman
int BUFFER_SIZE = 3; 

// Input Pool: YUYV (2 Channel) dari Kamera
struct FrameBuffer { Mat data; dma_buffer dbuf; bool ready; };
vector<FrameBuffer> inputPool; // YUYV RAW

// Process Pool: BGR (3 Channel) hasil konversi OCA untuk AI & Remap
vector<FrameBuffer> processPool; 

// Output Pool: BGR (3 Channel) hasil Remap untuk Display
vector<FrameBuffer> outputPool;


const int MAP_BUFFER_COUNT = 2;
FrameBuffer map1_bufs[MAP_BUFFER_COUNT], map2_bufs[MAP_BUFFER_COUNT];

mutex mtxData; condition_variable cv_drp;
int w_idx_cap=0, r_idx_drp=0, w_idx_drp=0;

// AI
DrpAiYolo yolo;

#define LUT_SIZE 2000          // Resolusi tabel (semakin besar semakin presisi)
#define MAX_RADIAN 3.14159f


// Mouse
double last_mouse_x, last_mouse_y;
bool is_dragging = false;
bool IS_CAMERA_MODE = false;

// --- Helper Functions ---
uint64_t get_drpai_start_addr(int drpai_fd) {
    drpai_data_t drpai_data;
    errno = 0;
    int ret = ioctl(drpai_fd, DRPAI_GET_DRPAI_AREA, &drpai_data);
    if (-1 == ret) return 0;
    return drpai_data.address;
}

void allocateDRPBuffer(FrameBuffer &fb, int width, int height, int type) {
    size_t size = width * height * CV_ELEM_SIZE(type);
    if (buffer_alloc_dmabuf(&fb.dbuf, size) != 0) { 
        cerr << "[CRITICAL] Alloc Fail. Check CMA!" << endl; exit(1); 
    }
    fb.data = Mat(height, width, type, fb.dbuf.mem); 
    fb.ready = false;
}

void freeDRPBuffer(FrameBuffer &fb) {
    if (fb.dbuf.mem) {
        munmap(fb.dbuf.mem, fb.dbuf.size);
        fb.dbuf.mem = nullptr;
    }
}

void cleanup_resources() {
    cout << "\n[CLEANUP] Releasing DMA Buffers..." << endl;
    for(int i=0; i<inputPool.size(); i++) freeDRPBuffer(inputPool[i]);
    for(int i=0; i<processPool.size(); i++) freeDRPBuffer(processPool[i]); // Clean process pool
    for(int i=0; i<outputPool.size(); i++) freeDRPBuffer(outputPool[i]);
    for(int i=0; i<MAP_BUFFER_COUNT; i++) {
        freeDRPBuffer(map1_bufs[i]);
        freeDRPBuffer(map2_bufs[i]);
    }
    cout << "[CLEANUP] Done." << endl;
}

void signal_handler(int signum) {
    cout << "\n[SIGNAL] Caught Ctrl+C. Exiting..." << endl;
    isRunning = false;
    cv_drp.notify_all();
    gtk_main_quit();
}

// === AKTIVASI DRP-OCA (OPENCV ACCELERATOR) ===
void* lib_handle = nullptr;
typedef int (*OCA_Activate_Func)(unsigned long*);
void initDRP() {
    // 1. Setup Environment Variable
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
        setenv("DRP_EXE_PATH", cwd, 1);

    // 2. LOAD LIBRARY (GABUNGAN IF-ELSE)
    
    // Prioritas 1: Biarkan Linux mencari otomatis (Paling Aman)
    lib_handle = dlopen("libopencv_imgproc.so", RTLD_NOW | RTLD_GLOBAL);

    // Prioritas 2: Jika gagal, coba cari di folder library 64-bit (Standar Yocto/Renesas)
    if (!lib_handle) {
        lib_handle = dlopen("/usr/lib64/libopencv_imgproc.so", RTLD_NOW | RTLD_GLOBAL);
    }

    // Prioritas 3: Jika masih gagal, coba path hardcoded lama Anda (Ubuntu style)
    if (!lib_handle) {
        lib_handle = dlopen("/usr/lib/aarch64-linux-gnu/renesas/libopencv_imgproc.so.4.1.0", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    }

    // 3. CEK HASIL LOAD
    if (!lib_handle) {
        // Jika semua gagal
        cerr << "[WARN] OCA Library not found in any path! Running on CPU (Slower)." << endl;
        return; // Jangan exit(1) biar aplikasi tetap jalan meski lambat
    }

    // 4. LOAD SIMBOL AKTIVASI
    // Menggunakan nama simbol yang sudah kita validasi lewat 'strings' sebelumnya
    OCA_Activate_Func OCA_Activate_Ptr = (OCA_Activate_Func)dlsym(lib_handle, "_Z12OCA_ActivatePm");

    if (OCA_Activate_Ptr) {
        unsigned long OCA_list[17] = {0};
        OCA_list[0]  = 1;
        OCA_list[16] = 1;
        OCA_Activate_Ptr(OCA_list);
        cout << "[INIT] DRP-OCA Activated! (HW Acceleration ON)" << endl;
    } else {
        cerr << "[ERROR] OCA_Activate symbol not found in the loaded library!" << endl;
    }
}

// === FILE I/O ===
const string CONFIG_FILE = "moildev_config.txt";

void saveConfig() {
    ofstream file(CONFIG_FILE);
    if (file.is_open()) {
        file << "VIEW1 " << view1_alpha << " " << view1_beta << " " << view1_zoom << " " << view1_flip_h << " " << view1_flip_v << endl;
        file << "VIEW2 " << view2_alpha << " " << view2_beta << " " << view2_zoom << " " << view2_flip_h << " " << view2_flip_v << endl;
        file << "VIEW3 " << view3_alpha << " " << view3_beta << " " << view3_zoom << " " << view3_flip_h << " " << view3_flip_v << endl;
        file << "PANO " << pano_alpha << " " << pano_beta << " " << pano_zoom << " " << pano_flip_h << " " << pano_flip_v << endl;
        file << "SINGLE " << cur_alpha << " " << cur_beta << " " << cur_zoom << " " << cur_flip_h << " " << cur_flip_v << endl;
        file.close();
        cout << "[CONFIG] Configuration SAVED." << endl;
    }
}

void loadConfig() {
    ifstream file(CONFIG_FILE);
    if (file.is_open()) {
        string key;
        float a, b, z;
        bool fh, fv;
        while (file >> key >> a >> b >> z >> fh >> fv) {
            if (key == "VIEW1") { view1_alpha=a; view1_beta=b; view1_zoom=z; view1_flip_h=fh; view1_flip_v=fv; }
            else if (key == "VIEW2") { view2_alpha=a; view2_beta=b; view2_zoom=z; view2_flip_h=fh; view2_flip_v=fv; }
            else if (key == "VIEW3") { view3_alpha=a; view3_beta=b; view3_zoom=z; view3_flip_h=fh; view3_flip_v=fv; }
            else if (key == "PANO") { pano_alpha=a; pano_beta=b; pano_zoom=z; pano_flip_h=fh; pano_flip_v=fv; }
            else if (key == "SINGLE") { cur_alpha=a; cur_beta=b; cur_zoom=z; cur_flip_h=fh; cur_flip_v=fv; }
        }
        file.close();
    }
}

float rho_lut[LUT_SIZE];
float rad_step = MAX_RADIAN / (LUT_SIZE - 1);

void initRhoLUT() {
    for (int i = 0; i < LUT_SIZE; i++) {
        float a = i * rad_step;
        float p0 = (float)PARA[0], p1 = (float)PARA[1], p2 = (float)PARA[2], 
              p3 = (float)PARA[3], p4 = (float)PARA[4], p5 = (float)PARA[5];
        
        // Kalkulasi polinomial (Horner's Method) sekali saja saat startup
        rho_lut[i] = (((((p0 * a + p1) * a + p2) * a + p3) * a + p4) * a + p5) * a;
    }
    cout << "[INIT] Rho LUT Generated. Size: " << LUT_SIZE << " entries." << endl;
}

const string MAP_DIR = "maps_cache";
// Pastikan folder maps ada
void ensure_maps_folder() {
    struct stat st = {0};
    if (stat(MAP_DIR.c_str(), &st) == -1) {
        mkdir(MAP_DIR.c_str(), 0777);
    }
}

// Generate nama file unik berdasarkan parameter saat ini
string generate_map_filename(ViewMode mode, int w, int h) {
    stringstream ss;
    ss << MAP_DIR << "/map_m" << mode << "_" << w << "x" << h;

    // Helper lambda untuk membulatkan float agar nama file tidak aneh (presisi 1 desimal)
    auto fmt = [](float f) { return (int)(f * 10); };
    auto fmt_b = [](bool b) { return b ? 1 : 0; };

    if (mode == MODE_SINGLE) {
        ss << "_a" << fmt(cur_alpha) << "_b" << fmt(cur_beta) << "_z" << fmt(cur_zoom)
           << "_fh" << fmt_b(cur_flip_h) << "_fv" << fmt_b(cur_flip_v);
    } 
    else if (mode == MODE_PANORAMA) {
        ss << "_pa" << fmt(pano_alpha) << "_pb" << fmt(pano_beta)
           << "_pfh" << fmt_b(pano_flip_h) << "_pfv" << fmt_b(pano_flip_v);
    }
    else if (mode == MODE_MAIN || mode == MODE_TRIPLE) {
        // Untuk mode kompleks, kita hash semua view
        // Pano (Only for MAIN)
        if (mode == MODE_MAIN) {
             ss << "_P" << fmt(pano_alpha) << "_" << fmt(pano_beta);
        }
        // View 1
        ss << "_V1_" << fmt(view1_alpha) << "_" << fmt(view1_beta) << "_" << fmt(view1_zoom);
        // View 2
        ss << "_V2_" << fmt(view2_alpha) << "_" << fmt(view2_beta) << "_" << fmt(view2_zoom);
        // View 3
        ss << "_V3_" << fmt(view3_alpha) << "_" << fmt(view3_beta) << "_" << fmt(view3_zoom);
    }
    
    ss << ".bin"; // Binary file agar cepat load/save
    return ss.str();
}

// Simpan Mat ke File Binary (Sangat Cepat)
void save_map_to_disk(const string& filename, cv::Mat& mx, cv::Mat& my) {
    ofstream file(filename, ios::binary);
    if (file.is_open()) {
        size_t size = mx.total() * mx.elemSize();
        file.write((char*)mx.data, size);
        file.write((char*)my.data, size);
        file.close();
        // cout << "[CACHE] Saved new map: " << filename << endl;
    }
}

// Load Mat dari File Binary
bool load_map_from_disk(const string& filename, cv::Mat& mx, cv::Mat& my) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        size_t size = mx.total() * mx.elemSize();
        file.read((char*)mx.data, size);
        file.read((char*)my.data, size);
        file.close();
        // cout << "[CACHE] Loaded map from disk!" << endl;
        return true;
    }
    return false;
}

// --- MOIL MATH & MAP UPDATE ---
class MoilNative {
public:
    double iCx, iCy, scale_x, scale_y;
    MoilNative() {
        scale_x = (double)IN_W / ORG_W; scale_y = (double)IN_H / ORG_H;
        iCx = ORG_ICX * scale_x; iCy = ORG_ICY * scale_y;
    }
    inline float getRhoFast(float alpha_rad) {
        // Proteksi jika input di luar range
        if (alpha_rad <= 0) return 0.0f;
        if (alpha_rad >= MAX_RADIAN) return rho_lut[LUT_SIZE - 1];

        // Hitung indeks berdasarkan rasio
        int idx = (int)(alpha_rad / rad_step);
        return rho_lut[idx];
    }

    void fillRegion(Mat &mx, Mat &my, Rect r, float alpha, float beta, float zoom, bool isPano, bool flip_h, bool flip_v) {
        // 1. Definisikan PI_F agar tidak error 'not declared'
        const float PI_F = 3.1415926535f;
        
        // 2. PRE-CALCULATE KONSTANTA (Sangat penting di luar loop)
        float ar = alpha * PI_F / 180.0f;
        float br = beta * PI_F / 180.0f;
        float cos_ar = cosf(ar);
        float sin_ar = sinf(ar);
        float cos_br = cosf(br);
        float sin_br = sinf(br);
        
        // Hitung konstanta f (focal length virtual)
        float f = r.width / (2.0f * tanf(30.0f * PI_F / 180.0f / zoom));

        #pragma omp parallel for collapse(2)
        for(int y=0; y<r.height; y++) {
            for(int x=0; x<r.width; x++) {
                float rho, theta;
                
                if(isPano) {
                    float nx = (float)x / r.width;
                    float ny = (float)y / r.height;
                    theta = (nx - 0.5f) * 2.0f * PI_F + br;
                    
                    // Menggunakan LUT: ny * ar adalah sudut alpha dalam radian
                    rho = getRhoFast(ny * ar); 
                } else {
                    float cx = (float)x - r.width / 2.0f;
                    float cy = (float)y - r.height / 2.0f;
                    
                    // Transformasi koordinat 3D ke 2D Fisheye
                    float x1 = cx;
                    float y1 = cy * cos_ar - f * sin_ar;
                    float z1 = cy * sin_ar + f * cos_ar;
                    
                    float x2 = x1 * cos_br - y1 * sin_br;
                    float y2 = x1 * sin_br + y1 * cos_br;
                    float z2 = z1;
                    
                    // Menggunakan LUT untuk mendapatkan rho dari sudut incident
                    rho = getRhoFast(atan2f(sqrtf(x2*x2 + y2*y2), z2)); 
                    theta = atan2f(y2, x2);
                }

                // Tentukan posisi target dengan mempertimbangkan Flip
                int target_x = flip_h ? (r.width - 1 - x) : x;
                int target_y = flip_v ? (r.height - 1 - y) : y;
                int py = r.y + target_y;
                int px = r.x + target_x;

                // Mapping ke koordinat asal kamera fisheye
                if(px >= 0 && px < mx.cols && py >= 0 && py < mx.rows) {
                    // iCx, iCy, scale_x, scale_y diambil dari class MoilNative
                    mx.at<float>(py, px) = (float)(iCx + (rho * (float)scale_x) * cosf(theta));
                    my.at<float>(py, px) = (float)(iCy + (rho * (float)scale_y) * sinf(theta));
                }
            }
        }
    }

    void fillStandard(Mat &mx, Mat &my, Rect r) {
        float sx = (float)IN_W / r.width; float sy = (float)IN_H / r.height;
        #pragma omp parallel for collapse(2)
        for(int y=0; y<r.height; y++) {
            for(int x=0; x<r.width; x++) {
                int py=r.y+y, px=r.x+x;
                if(px>=0 && px<mx.cols && py>=0 && py<mx.rows) {
                    mx.at<float>(py,px)=x*sx; my.at<float>(py,px)=y*sy;
                }
            }
        }
    }
};

void mapUpdateThread() {
    this_thread::sleep_for(chrono::milliseconds(800));

    ensure_maps_folder();

    MoilNative moil;
    
    // === OPTIMASI DOWNSCALED MAP ===
    // Menghitung Map di resolusi setengah (Scale 0.5) untuk meringankan CPU 4x lipat.
    // Hasilnya akan di-resize kembali ke Full HD agar gambar tetap tajam.
    int CALC_W = OUT_W / 2;
    int CALC_H = OUT_H / 2;
    
    // Buffer Kecil untuk kalkulasi matematika berat
    Mat mx_small(CALC_H, CALC_W, CV_32F);
    Mat my_small(CALC_H, CALC_W, CV_32F);

    // Buffer Akhir (Full HD) untuk dikirim ke DRP
    Mat mx(OUT_H, OUT_W, CV_32F);
    Mat my(OUT_H, OUT_W, CV_32F);

    ViewMode last_mode = MODE_SINGLE; 
    
    // Cache variables (Logic "Optimization Stability")
    float l_a=-99, l_b=-99, l_z=-99; bool l_fh=false, l_fv=false;
    float l_pa=-99, l_pb=-99; bool l_pfh=false, l_pfv=false;
    float l_v1a=-99, l_v1b=-99, l_v1z=-99; bool l_v1fh=false, l_v1fv=false;
    float l_v2a=-99, l_v2b=-99, l_v2z=-99; bool l_v2fh=false, l_v2fv=false;
    float l_v3a=-99, l_v3b=-99, l_v3z=-99; bool l_v3fh=false, l_v3fv=false;

    while(isRunning) {
        if (is_paused.load()) { this_thread::sleep_for(chrono::milliseconds(100)); continue; }

        ViewMode mode = current_mode.load();
        bool changed = false;

        // --- DETEKSI PERUBAHAN ---
        // (Logika ini sama persis dengan kode asli Anda, tidak perlu diubah)
        if(mode != last_mode) changed = true;
        else if (mode == MODE_SINGLE) {
            if(cur_alpha!=l_a || cur_beta!=l_b || cur_zoom!=l_z || cur_flip_h!=l_fh || cur_flip_v!=l_fv) changed = true;
        } 
        else if (mode == MODE_PANORAMA) {
            if(pano_alpha!=l_pa || pano_beta!=l_pb || pano_flip_h!=l_pfh || pano_flip_v!=l_pfv) changed = true;
        }
        else if (mode == MODE_MAIN) {
            if(view1_alpha!=l_v1a || view1_beta!=l_v1b || view1_zoom!=l_v1z || view1_flip_h!=l_v1fh || view1_flip_v!=l_v1fv ||
               view2_alpha!=l_v2a || view2_beta!=l_v2b || view2_zoom!=l_v2z || view2_flip_h!=l_v2fh || view2_flip_v!=l_v2fv ||
               view3_alpha!=l_v3a || view3_beta!=l_v3b || view3_zoom!=l_v3z || view3_flip_h!=l_v3fh || view3_flip_v!=l_v3fv ||
               pano_alpha!=l_pa || pano_beta!=l_pb || pano_flip_h!=l_pfh || pano_flip_v!=l_pfv) changed = true;
        }
        else if (mode == MODE_TRIPLE) {
            if(view1_alpha!=l_v1a || view1_beta!=l_v1b || view1_zoom!=l_v1z || view1_flip_h!=l_v1fh || view1_flip_v!=l_v1fv ||
               view2_alpha!=l_v2a || view2_beta!=l_v2b || view2_zoom!=l_v2z || view2_flip_h!=l_v2fh || view2_flip_v!=l_v2fv ||
               view3_alpha!=l_v3a || view3_beta!=l_v3b || view3_zoom!=l_v3z || view3_flip_h!=l_v3fh || view3_flip_v!=l_v3fv) changed = true;
        }

        // JIKA TIDAK ADA PERUBAHAN, TIDUR (Hemat CPU)
        if (maps_ready && !changed) { 
            this_thread::sleep_for(chrono::milliseconds(20)); continue; 
        }

        // === MULAI PROSES PEMBUATAN MAP ===
        auto t1 = chrono::steady_clock::now();
        int back_idx = (active_map_idx.load() + 1) % MAP_BUFFER_COUNT;

        // 1. Generate Nama File Unik berdasarkan parameter saat ini
        string map_fname = generate_map_filename(mode, CALC_W, CALC_H);

        // 2. Cek apakah file sudah ada di Cache?
        bool loaded_from_cache = load_map_from_disk(map_fname, mx_small, my_small);

        if (loaded_from_cache) {
            // [JALUR CEPAT] Load dari disk (IO bound, CPU ringan)
            // Tidak perlu update variabel cache l_a, l_b dll karena kita tidak menghitungnya.
            // Tapi kita perlu update agar loop berikutnya mendeteksi 'tidak berubah'
            // (Disederhanakan: kita update variabel cache di bawah)
        } else {
            // [JALUR LAMBAT] Kalkulasi Matematika (CPU bound)
            // Reset buffer
            mx_small.setTo(Scalar(-1)); my_small.setTo(Scalar(-1));

            switch(mode) {
                case MODE_MAIN: {
                    int split_y = CALC_H / 2;
                    int third_w = CALC_W / 3;
                    moil.fillRegion(mx_small, my_small, Rect(0, 0, CALC_W, split_y), pano_alpha.load(), pano_beta.load(), 0, true, pano_flip_h, pano_flip_v);
                    moil.fillRegion(mx_small, my_small, Rect(0, split_y, third_w, split_y), view1_alpha, view1_beta, view1_zoom, false, view1_flip_h, view1_flip_v);
                    moil.fillRegion(mx_small, my_small, Rect(third_w, split_y, third_w, split_y), view2_alpha, view2_beta, view2_zoom, false, view2_flip_h, view2_flip_v);
                    moil.fillRegion(mx_small, my_small, Rect(2*third_w, split_y, third_w, split_y), view3_alpha, view3_beta, view3_zoom, false, view3_flip_h, view3_flip_v);
                    break;
                }
                case MODE_TRIPLE: {
                    int sidebar_w = CALC_W / 3;
                    int main_w = CALC_W - sidebar_w;
                    int half_h = CALC_H / 2;
                    moil.fillRegion(mx_small, my_small, Rect(0, 0, sidebar_w, half_h), view1_alpha, view1_beta, view1_zoom, false, view1_flip_h, view1_flip_v);
                    moil.fillRegion(mx_small, my_small, Rect(0, half_h, sidebar_w, half_h), view2_alpha, view2_beta, view2_zoom, false, view2_flip_h, view2_flip_v);
                    moil.fillRegion(mx_small, my_small, Rect(sidebar_w, 0, main_w, CALC_H), view3_alpha, view3_beta, view3_zoom, false, view3_flip_h, view3_flip_v);
                    break;
                }
                case MODE_ORIGINAL: 
                    moil.fillStandard(mx_small, my_small, Rect(0, 0, CALC_W, CALC_H)); 
                    break;
                case MODE_PANORAMA: {
                    int split_y = CALC_H / 2; 
                    int start_y = CALC_H / 4;
                    moil.fillRegion(mx_small, my_small, Rect(0, start_y, CALC_W, split_y), pano_alpha.load(), pano_beta.load(), 0, true, pano_flip_h, pano_flip_v);
                    break;
                }
                case MODE_GRID: {
                    // Grid biasanya statis, tapi tidak apa-apa di-cache juga
                    int bw = CALC_W/2, bh = CALC_H/2;
                    moil.fillRegion(mx_small, my_small, Rect(0, 0, bw, bh), 0, 0, 2.0, false, false, false);
                    moil.fillRegion(mx_small, my_small, Rect(bw, 0, bw, bh), 45, 0, 2.0, false, false, false);
                    moil.fillRegion(mx_small, my_small, Rect(0, bh, bw, bh), 45, -90, 2.0, false, false, false);
                    moil.fillRegion(mx_small, my_small, Rect(bw, bh, bw, bh), 45, 90, 2.0, false, false, false);
                    break;
                }
                case MODE_SINGLE: {
                    moil.fillRegion(mx_small, my_small, Rect(0, 0, CALC_W, CALC_H), cur_alpha, cur_beta, cur_zoom, false, cur_flip_h, cur_flip_v);
                    break;
                }
            }

            // SIMPAN HASIL KALKULASI KE DISK AGAR NANTI CEPAT
            save_map_to_disk(map_fname, mx_small, my_small);
        }
        
        l_a = cur_alpha; l_b = cur_beta; l_z = cur_zoom; l_fh = cur_flip_h; l_fv = cur_flip_v;
        l_pa = pano_alpha; l_pb = pano_beta; l_pfh = pano_flip_h; l_pfv = pano_flip_v;
        l_v1a = view1_alpha; l_v1b = view1_beta; l_v1z = view1_zoom; l_v1fh = view1_flip_h; l_v1fv = view1_flip_v;
        l_v2a = view2_alpha; l_v2b = view2_beta; l_v2z = view2_zoom; l_v2fh = view2_flip_h; l_v2fv = view2_flip_v;
        l_v3a = view3_alpha; l_v3b = view3_beta; l_v3z = view3_zoom; l_v3fh = view3_flip_h; l_v3fv = view3_flip_v;

        // --- PROSES RESIZE (UPSCALING MAP) ---
        // Membesarkan Map Kecil -> Map Besar
        // INTER_LINEAR wajib dipakai agar hasil lengkungannya halus (tidak kotak-kotak)
        resize(mx_small, mx, Size(OUT_W, OUT_H), 0, 0, INTER_LINEAR);
        resize(my_small, my, Size(OUT_W, OUT_H), 0, 0, INTER_LINEAR);

        auto t2 = chrono::steady_clock::now();
        stats_map_ms.store(chrono::duration_cast<chrono::milliseconds>(t2 - t1).count());

        // Konversi ke format Fixed Point untuk DRP
        convertMaps(mx, my, map1_bufs[back_idx].data, map2_bufs[back_idx].data, CV_16SC2, false);
        buffer_flush_dmabuf(map1_bufs[back_idx].dbuf.idx, map1_bufs[back_idx].dbuf.size);
        buffer_flush_dmabuf(map2_bufs[back_idx].dbuf.idx, map2_bufs[back_idx].dbuf.size);
        active_map_idx.store(back_idx);
        if (!maps_ready) maps_ready = true;
        last_mode = mode;
    }
}

/**
 * @brief Draws bounding boxes on the remapped output image based on fisheye detections.
 * Uses a reverse-lookup strategy via map1 to translate coordinates.
 * * @param out   The destination image (Remapped/Dewarped view).
 * @param map1  The X-coordinate map used for cv::remap (contains source X coordinates).
 * @param dets  Raw detections from YOLO (coordinates relative to the original fisheye image).
 * @param mode  Current display mode (Main, Triple, etc.) to determine scanning regions.
 */
void drawBBoxesOnOutput(Mat &out, Mat &map1, const vector<detection>& dets, ViewMode mode) {
    if (dets.empty()) return;

    // --- STEP 1: Filter Detections ---
    // Filter out low-probability detections to reduce processing overhead.
    vector<detection> valid_dets;
    valid_dets.reserve(dets.size());
    for (const auto& d : dets) {
        if (d.prob > 0.50) { // Confidence threshold > 50%
            valid_dets.push_back(d);
        }
    }
    if (valid_dets.empty()) return;

    // --- STEP 2: Define Search Regions (Optimization) ---
    // Instead of scanning the whole image, we define specific Regions of Interest (ROI)
    // based on the current ViewMode layout.
    vector<Rect> regions;
    int split_y = OUT_H / 2;
    int third_w = OUT_W / 3;
    int start_y = OUT_H / 4; // For center-aligned modes
    if (mode == MODE_MAIN) {
        // Layout: Top half is Panorama, Bottom half is split into 3 views

        regions.push_back(Rect(0, 0, OUT_W, split_y));            // Top Panorama
        regions.push_back(Rect(0, split_y, third_w, split_y));    // Bottom Left
        regions.push_back(Rect(third_w, split_y, third_w, split_y)); // Bottom Center
        regions.push_back(Rect(2*third_w, split_y, third_w, split_y)); // Bottom Right
    } else if (mode == MODE_TRIPLE) {
        // === MODE TRIPLE (LAYOUT BARU) ===
        int sidebar_w = OUT_W / 3;       // Lebar Sidebar Kiri
        int main_w = OUT_W - sidebar_w;  // Lebar Utama Kanan
        int half_h = OUT_H / 2;          // Tinggi separuh

        // Region 1: Kiri Atas
        regions.push_back(Rect(0, 0, sidebar_w, half_h)); 
        
        // Region 2: Kiri Bawah
        regions.push_back(Rect(0, half_h, sidebar_w, half_h)); 
        
        // Region 3: Kanan Full (Besar)
        regions.push_back(Rect(sidebar_w, 0, main_w, OUT_H));

    } else if (mode == MODE_PANORAMA) {
         // === MODE PANORAMA (CENTER) ===
         // Y dimulai dari start_y (tengah)
         regions.push_back(Rect(0, start_y, OUT_W, split_y));

    } else {
        // Full screen modes (Original, Single, Panorama Full)
        regions.push_back(Rect(0, 0, OUT_W, OUT_H));
    }

    // Structure to track min/max coordinates of the projected box on the Output screen
    struct BoxCoords { int min_x=9999, min_y=9999, max_x=-1, max_y=-1; bool found=false; };
    vector<BoxCoords> out_boxes(valid_dets.size());

    // --- STEP 3: Reverse Mapping (Scanning) ---
    // Scan step size. Higher value = Faster CPU performance but lower box precision.
    // We skip pixels to reduce the loop count (CPU optimization).
    int step = 20; 

    for (const auto& roi : regions) {
        // Reset box coordinates for the current region
        for(auto &b : out_boxes) { b.found = false; b.min_x=9999; b.min_y=9999; b.max_x=-1; b.max_y=-1; }

        // Loop through the Output pixels within the ROI
        for (int y = roi.y; y < roi.y + roi.height; y += step) {
            short* ptr_map = map1.ptr<short>(y); // Access the X-map row
            
            for (int x = roi.x; x < roi.x + roi.width; x += step) {
                // Get the Source Coordinate (Fisheye Input) from the map
                // map1 contains the 'src_x' and 'src_y' for every pixel (x,y)
                int idx = x * 2;
                int src_x = ptr_map[idx];
                int src_y = ptr_map[idx+1];

                if (src_x <= 0) continue; // Skip invalid mapping points (black areas)

                // Check if this source pixel belongs to any detected object
                for (size_t i = 0; i < valid_dets.size(); ++i) {
                    const auto& d = valid_dets[i];
                    int dx = d.bbox.x - d.bbox.w/2; // BBox Left
                    int dy = d.bbox.y - d.bbox.h/2; // BBox Top
                    
                    // Logic: If the source pixel is inside the YOLO box...
                    if (src_x >= dx && src_x < dx + d.bbox.w) {
                        if (src_y >= dy && src_y < dy + d.bbox.h) {
                            // ...then update the Output Box boundaries.
                            auto& box = out_boxes[i];
                            if (x < box.min_x) box.min_x = x;
                            if (y < box.min_y) box.min_y = y;
                            if (x > box.max_x) box.max_x = x;
                            if (y > box.max_y) box.max_y = y;
                            box.found = true;
                            break; // Object found, move to next pixel
                        }
                    }
                }
            }
        }

        // --- STEP 4: Rendering ---
        // Draw the calculated boxes on the output frame
        for (size_t i = 0; i < valid_dets.size(); ++i) {
            if (out_boxes[i].found) {
                int bx = out_boxes[i].min_x;
                int by = out_boxes[i].min_y;
                int bw = out_boxes[i].max_x - out_boxes[i].min_x;
                int bh = out_boxes[i].max_y - out_boxes[i].min_y;
                
                // Add padding to compensate for the scanning 'step' (20px)
                int pad = step / 2;
                bx = max(roi.x, bx - pad);
                by = max(roi.y, by - pad);
                bw += step;
                bh += step;

                // Boundary checks
                if (bx + bw > roi.x + roi.width) bw = (roi.x + roi.width) - bx;
                if (by + bh > roi.y + roi.height) bh = (roi.y + roi.height) - by;

                // Only draw if the resulting box is big enough
                if (bw > 10 && bh > 10) {
                    rectangle(out, Rect(bx, by, bw, bh), Scalar(0, 255, 0), 2);
                    
                    // Draw Label (Percentage)
                    putText(out, to_string((int)(valid_dets[i].prob*100))+"%", 
                            Point(bx, by > 20 ? by-5 : by+15), 
                            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
                }
            }
        }
    }
}

void captureThread(string src) {
    string pipe;
    bool is_file = !isdigit(src[0]);

    if (!is_file) {
        // === MODE KAMERA (SAFE MODE) ===
        // HAPUS 'framerate=30/1' yang memaksa. 
        // Biarkan GStreamer negosiasi sendiri speed terbaik kamera.
        pipe = "v4l2src device=/dev/video" + src +
               " ! image/jpeg, width=" + to_string(IN_W) + 
               ", height=" + to_string(IN_H) + 
               // ", framerate=30/1" <--- INI BIANG KEROKNYA (HAPUS/KOMENTARI)
               " ! jpegdec"                 
               " ! videoconvert"            
               " ! video/x-raw, format=YUY2"
               " ! queue max-size-buffers=2 leaky=downstream" 
               " ! appsink drop=true sync=false";
               
        cout << "[CAPTURE] Pipeline Safe Mode: " << pipe << endl;
    } else {
        // Mode File
        pipe = "filesrc location=" + src +
               " ! decodebin ! queue"
               " ! videoconvert ! video/x-raw,format=YUY2"
               " ! appsink max-buffers=2 drop=true sync=true";
    }
    VideoCapture *cap = new VideoCapture(pipe, CAP_GSTREAMER);
    if (!cap->isOpened()) { 
        cerr << "[ERROR] Gagal membuka source: " << src << endl;
        isRunning = false; 
        return; 
    }

    Mat temp;
    while (isRunning) {

        auto t_read_start = chrono::steady_clock::now();

        if (!cap->read(temp) || temp.empty()) { 
            if (is_file) {
                cout << "[CAPTURE] Video Ended. Replaying..." << endl;
                cap->set(CAP_PROP_POS_FRAMES, 0); 
            }
            continue; 
        }
        auto t_read_end = chrono::steady_clock::now();
        // Simpan ke variabel global
        std::chrono::duration<double, std::milli> ms_read = t_read_end - t_read_start;
        stats_read_ms.store(ms_read.count());
        if (temp.cols != IN_W || temp.rows != IN_H) {
            resize(temp, temp, Size(IN_W, IN_H));
        }

        int idx = w_idx_cap;
        size_t frame_size = IN_W * IN_H * 2; // YUY2 = 2 bytes per pixel

        if (inputPool[idx].data.data && temp.data) {
            std::memcpy(inputPool[idx].data.data, temp.data, frame_size);
        }

        buffer_flush_dmabuf(inputPool[idx].dbuf.idx, inputPool[idx].dbuf.size);

        {
            lock_guard<mutex> lock(mtxData);
            inputPool[idx].ready = true;
            w_idx_cap = (w_idx_cap + 1) % BUFFER_SIZE;
            r_idx_drp = idx;
        }
        cv_drp.notify_one();
    }
    
    if (cap->isOpened()) cap->release();
    delete cap;
}

void drawDetectionsOnInput(Mat &img, const vector<detection>& dets) {
    for (const auto& d : dets) {
        if (d.prob == 0) continue;
        int x = d.bbox.x - d.bbox.w / 2;
        int y = d.bbox.y - d.bbox.h / 2;
        rectangle(img, Rect(x, y, d.bbox.w, d.bbox.h), Scalar(0, 255, 0), 4);
        string label = to_string((int)(d.prob * 100)) + "%";
        putText(img, label, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 3);
    }
}

void drawInfoBox(Mat &img) {}

gboolean update_gui_image(gpointer user_data)
{
    int idx = (int)(intptr_t)user_data;
    if (idx < 0 || idx >= BUFFER_SIZE) {
        ui_update_pending = false;
        return FALSE;
    }

    Mat &src = outputPool[idx].data;
    
    // 1. Ambil ukuran widget yang tersedia
    int avail_w = gtk_widget_get_allocated_width(img_display);
    int avail_h = gtk_widget_get_allocated_height(img_display);
    
    // 2. Jika widget belum valid, gunakan ukuran layar sebagai fallback
    if (avail_w < 1 || avail_h < 1) {
        GdkScreen *screen = gdk_screen_get_default();
        avail_w = gdk_screen_get_width(screen);
        avail_h = gdk_screen_get_height(screen);
    }
    
    // 3. Tentukan area maksimal untuk gambar
    //    Kurangi dengan lebar sidebar/panel kontrol jika ada
    const int SIDEBAR_WIDTH = 320;
    int max_w = avail_w - SIDEBAR_WIDTH;
    int max_h = avail_h;
    
    // 4. Pastikan area tidak terlalu kecil
    //    Minimal 800px untuk lebar, dan 600px untuk tinggi
    if (max_w < 800) max_w = 800;
    if (max_h < 600) max_h = 600;
    
    // 5. Hitung skala agar gambar fit di area (jaga aspect ratio)
    float scale_w = (float)max_w / src.cols;
    float scale_h = (float)max_h / src.rows;
    
    // Pilih skala terkecil agar gambar tidak keluar dari area
    float scale = (scale_w < scale_h) ? scale_w : scale_h;
    
    // 6. TAMBAHAN: Pastikan skala tidak terlalu kecil untuk gambar kecil
    //    Jika gambar sumber kecil (misal 480p), berikan skala minimum
    float min_scale = 1.0f; // Minimal 1:1 untuk gambar kecil
    if (src.cols <= 640) { // Jika lebar gambar <= 640px
        min_scale = 1.5f; // Perbesar sedikit agar tidak terlalu kecil
    }
    
    // Terapkan batasan skala minimum
    if (scale < min_scale) scale = min_scale;
    
    // 7. Hitung ukuran akhir
    int dest_w = (int)(src.cols * scale);
    int dest_h = (int)(src.rows * scale);
    
    // 8. Pastikan tidak melebihi area maksimal
    if (dest_w > max_w) {
        dest_w = max_w;
        dest_h = (int)(src.rows * ((float)max_w / src.cols));
    }
    
    if (dest_h > max_h) {
        dest_h = max_h;
        dest_w = (int)(src.cols * ((float)max_h / src.rows));
    }

    // Buat pixbuf dari data gambar
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
        src.data,
        GDK_COLORSPACE_RGB,
        FALSE,
        8,
        src.cols,
        src.rows,
        src.step,
        NULL,
        NULL
    );

    if (pixbuf) {
        // Cek apakah resize diperlukan
        if (dest_w == src.cols && dest_h == src.rows) {
            // Ukuran sama, langsung tampilkan
            gtk_image_set_from_pixbuf(GTK_IMAGE(img_display), pixbuf);
        } else {
            // Resize dengan interpolasi yang lebih baik untuk pembesaran
            GdkInterpType interp = GDK_INTERP_NEAREST;
            if (scale > 1.0f) {
                // Jika memperbesar, gunakan bilinear agar lebih smooth
                interp = GDK_INTERP_BILINEAR;
            }
            
            GdkPixbuf *scaled = gdk_pixbuf_scale_simple(
                pixbuf,
                dest_w,
                dest_h,
                interp
            );
            gtk_image_set_from_pixbuf(GTK_IMAGE(img_display), scaled);
            g_object_unref(scaled);
        }
        g_object_unref(pixbuf);
    }

    outputPool[idx].ready = false;
    ui_update_pending = false;
    return FALSE;
}



void pipelineThread() {
    int frame_count = 0; 
    auto last_fps_time = chrono::steady_clock::now();

    vector<detection> cached_results;
    int ai_skip_counter = 0; 

    while(isRunning) {
        if (is_paused.load()) {
            unique_lock<mutex> lock(mtxData);
            if (inputPool[r_idx_drp].ready) inputPool[r_idx_drp].ready = false; 
            lock.unlock();
            this_thread::sleep_for(chrono::milliseconds(100));
            continue;
        }

        int idx_in = -1;
        { 
            unique_lock<mutex> lock(mtxData); 
            if(cv_drp.wait_for(lock, chrono::milliseconds(1000), []{ return inputPool[r_idx_drp].ready || !isRunning; })) {
                if(!isRunning) break; 
                idx_in = r_idx_drp; inputPool[idx_in].ready = false; 
            } else {
                continue;
            }
        }
        
        int idx_out = w_idx_drp;
        int map_idx = active_map_idx.load();
        
        AiStats ai_stats;
        vector<detection> results;
        
        // === 1. Convert YUYV -> BGR (KOTAK MERAH 2) ===
        auto t_cvt_start = chrono::steady_clock::now();
        
        cv::cvtColor(inputPool[idx_in].data, processPool[idx_in].data, COLOR_YUV2RGB_YUY2);
        buffer_flush_dmabuf(processPool[idx_in].dbuf.idx, processPool[idx_in].dbuf.size);
        
        auto t_cvt_end = chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> ms_cvt = t_cvt_end - t_cvt_start;
        stats_cvt_ms.store(ms_cvt.count());

        
        // === 2. Run DRP-AI YOLO (KOTAK HIJAU 1) ===
        bool run_inference = true;
        if (IS_CAMERA_MODE) {
            if (ai_skip_counter % 5 != 0) run_inference = false;
            ai_skip_counter++;
        } else {
            run_inference = true;
        }

        if (run_inference) {
            results = yolo.run_detection(processPool[idx_in].data, ai_stats);
            cached_results = results; 
            stats_ai_ms.store(ai_stats.total);
        } else {
            results = cached_results;
            ai_stats.total = 0; 
        }

        // === 3. Remap Fisheye (KOTAK HIJAU 2) ===
        auto t_remap_start = chrono::steady_clock::now();
        if (maps_ready && !map1_bufs[map_idx].data.empty()) {
            cv::remap(processPool[idx_in].data, outputPool[idx_out].data, map1_bufs[map_idx].data, map2_bufs[map_idx].data, INTER_NEAREST, BORDER_CONSTANT);
        } else {
            processPool[idx_in].data.copyTo(outputPool[idx_out].data); 
        }
        auto t_remap_end = chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> ms_remap = t_remap_end - t_remap_start;
        stats_remap_ms.store(ms_remap.count());

        
        // === 4. Draw Boxes (KOTAK MERAH 3) ===
        auto t_draw_start = chrono::steady_clock::now();
        if (maps_ready && !map1_bufs[map_idx].data.empty()) {
            drawBBoxesOnOutput(outputPool[idx_out].data, map1_bufs[map_idx].data, results, current_mode.load());
        }
        // (Opsional: Jika Anda mengaktifkan cvtColor BGR->RGB untuk display, masukkan ke sini juga)
        
        auto t_draw_end = chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> ms_draw = t_draw_end - t_draw_start;
        stats_draw_ms.store(ms_draw.count());


        // --- FPS & LOGGING (FULL CALCULATION) ---
        frame_count++;
        auto now = chrono::steady_clock::now();
        double elapsed_ms = chrono::duration_cast<chrono::milliseconds>(now - last_fps_time).count();
        if(elapsed_ms >= 1000.0) { 
            double current_fps = (frame_count * 1000.0) / elapsed_ms;
            stats_fps.store(current_fps);
            frame_count = 0; last_fps_time = now;
            
            string mode_name = "Unknown";
            switch(current_mode.load()) {
                case MODE_MAIN: mode_name = "Main"; break;
                case MODE_ORIGINAL: mode_name = "Original"; break;
                case MODE_PANORAMA: mode_name = "Panorama"; break;
                case MODE_GRID: mode_name = "Grid"; break;
                case MODE_SINGLE: mode_name = "Single"; break;
                case MODE_TRIPLE: mode_name = "Triple"; break;
            }

            // --- PERHITUNGAN TOTAL SEMUA KOTAK ---
            double t_read = stats_read_ms.load();
            double t_cvt  = stats_cvt_ms.load();
            double t_ai   = stats_ai_ms.load();
            double t_remap= stats_remap_ms.load();
            double t_draw = stats_draw_ms.load();

            double total_pipeline_latency = t_read + t_cvt + t_ai + t_remap + t_draw;

            printf("\n=================[ FULL PERFORMANCE LOG ]=================\n");
            printf(" [GENERAL]\n");
            printf("  > Source    : %s\n", IS_CAMERA_MODE ? "Camera" : "Video File");
            printf("  > Mode      : %s\n", mode_name.c_str());
            printf("  > FPS       : %d\n", (int)current_fps);
            printf("  > Input     : %dx%d\n", IN_W, IN_H);
            printf("  > Output    : %dx%d\n", OUT_W, OUT_H);
            printf(" ---------------------------------------------------------\n");
            printf(" [LATENCY BREAKDOWN (RED + GREEN SCOPES)]\n");
            printf("  1. Cam Read : %6.2f ms  [Red]\n", t_read);
            printf("  2. CvtColor : %6.2f ms  [Red]\n", t_cvt);
            printf("  3. YOLO AI  : %6.2f ms  [Green]\n", t_ai);
            printf("  4. Remap    : %6.2f ms  [Green]\n", t_remap);
            printf("  5. Drawing  : %6.2f ms  [Red]\n", t_draw);
            printf("  --------------------------------------\n");
            printf("  > GRAND TOTAL: %6.2f ms\n", total_pipeline_latency);
            printf(" ---------------------------------------------------------\n");
            fflush(stdout);
        }

        buffer_flush_dmabuf(outputPool[idx_out].dbuf.idx, outputPool[idx_out].dbuf.size);
        { 
            lock_guard<mutex> lock(mtxData); 
            outputPool[idx_out].ready = true; 
            w_idx_drp = (w_idx_drp + 1) % BUFFER_SIZE; 
            if (!ui_update_pending.load()) { 
                ui_update_pending = true; 
                g_idle_add(update_gui_image, (gpointer)(intptr_t)idx_out); 
            }
        }
    }
}

extern "C" {
    void on_window_destroy(GtkWidget *object, gpointer user_data) { 
        isRunning = false; 
        gtk_main_quit(); 
    }
    gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data) { 
        if (event->keyval == GDK_KEY_Escape) { 
            isRunning = false; 
            gtk_main_quit(); 
            return TRUE; 
        } 
        return FALSE; 
    }
    void on_btn_main_clicked(GtkButton *b) { current_mode = MODE_MAIN; }
    void on_btn_orig_clicked(GtkButton *b) { current_mode = MODE_ORIGINAL; }
    // void on_btn_pano_clicked(GtkButton *b) { current_mode = MODE_PANORAMA; }
    void on_btn_grid_clicked(GtkButton *b) { current_mode = MODE_GRID; }
    void on_btn_ptz_clicked(GtkButton *b)  { current_mode = MODE_SINGLE; }
    void on_btn_triple_clicked(GtkButton *b) { current_mode = MODE_TRIPLE; }

    void on_btn_flip_h_clicked(GtkButton *b) { 
        if (current_mode == MODE_SINGLE) cur_flip_h = !cur_flip_h;
        else if (current_mode == MODE_PANORAMA) pano_flip_h = !pano_flip_h;
        else if (active_view_id == 0) view1_flip_h = !view1_flip_h;
        else if (active_view_id == 1) view2_flip_h = !view2_flip_h;
        else if (active_view_id == 2) view3_flip_h = !view3_flip_h;
        else if (active_view_id == 3) pano_flip_h = !pano_flip_h; 
    }

    void on_btn_flip_v_clicked(GtkButton *b) { 
        if (current_mode == MODE_SINGLE) cur_flip_v = !cur_flip_v;
        else if (current_mode == MODE_PANORAMA) pano_flip_v = !pano_flip_v;
        else if (active_view_id == 0) view1_flip_v = !view1_flip_v;
        else if (active_view_id == 1) view2_flip_v = !view2_flip_v;
        else if (active_view_id == 2) view3_flip_v = !view3_flip_v;
        else if (active_view_id == 3) pano_flip_v = !pano_flip_v; 
    }

    void on_btn_pause_toggled(GtkToggleButton *b) { is_paused = gtk_toggle_button_get_active(b); }
    void on_btn_save_clicked(GtkButton *b) { saveConfig(); }
    void on_btn_pano_clicked(GtkButton *b) { 
            current_mode = MODE_PANORAMA;
    }
    // === MOUSE DETEKSI ===
    gboolean on_mouse_down(GtkWidget *widget, GdkEventButton *event, gpointer user_data) { 
        if (event->button == 1) { 
            is_dragging = true; 
            last_mouse_x = event->x; last_mouse_y = event->y;
            
            int h = gtk_widget_get_allocated_height(GTK_WIDGET(widget));
            int w = gtk_widget_get_allocated_width(GTK_WIDGET(widget));
            
            if (current_mode == MODE_MAIN) {
                if (event->y < h/2) {
                     active_view_id = 3; // Pano
                } else {
                    if (event->x < w/3) active_view_id = 0;      // View 1 (Kiri)
                    else if (event->x < (2*w)/3) active_view_id = 1; // View 2 (Tengah)
                    else active_view_id = 2;                     // View 3 (Kanan)
                }
            } else if (current_mode == MODE_TRIPLE) {
                if (event->x < w/3) active_view_id = 0;
                else if (event->x < (2*w)/3) active_view_id = 1;
                else active_view_id = 2;
            }
        }
        else if (event->button == 3) { // Reset Logic
            if (current_mode == MODE_MAIN) {
                if (active_view_id == 0) { view1_zoom = 3.5f; view1_flip_h=false; }
                else if (active_view_id == 1) { view2_zoom = 3.5f; view2_flip_h=false; }
                else if (active_view_id == 2) { view3_zoom = 3.5f; view3_flip_h=false; }
                else if (active_view_id == 3) { pano_alpha = 110.0f; pano_beta = 0.0f; pano_flip_h = false; }
            } else if (current_mode == MODE_TRIPLE) {
                if (active_view_id == 0) view1_zoom = 3.5f;
                else if (active_view_id == 1) view2_zoom = 3.5f;
                else view3_zoom = 3.5f;
            } else if (current_mode == MODE_SINGLE) {
                cur_zoom = 2.0f; cur_flip_h = false;
            } else if (current_mode == MODE_PANORAMA) {
                pano_alpha = 110.0f; pano_beta = 0.0f; pano_flip_h = false;
            }
        } 
        return TRUE; 
    }
    gboolean on_mouse_up(GtkWidget *widget, GdkEventButton *event, gpointer user_data) { is_dragging = false; return TRUE; }
    gboolean on_mouse_move(GtkWidget *widget, GdkEventMotion *event, gpointer user_data) { 
        if (is_dragging) { 
            float dx = (event->x - last_mouse_x) * 0.2f; 
            float dy = (event->y - last_mouse_y) * 0.2f; 
            
            if (current_mode == MODE_SINGLE) {
                cur_alpha = cur_alpha + dx; cur_beta = cur_beta + dy; 
            } 
            else if (current_mode == MODE_PANORAMA) {
                pano_beta = pano_beta + dx; pano_alpha = pano_alpha + dy; 
            }
            else if (current_mode == MODE_MAIN) {
                if (active_view_id == 3) {
                     pano_beta = pano_beta + dx; pano_alpha = pano_alpha + dy; 
                } else if (active_view_id == 0) { 
                    view1_alpha = view1_alpha + dx; view1_beta = view1_beta + dy; 
                } else if (active_view_id == 1) { 
                    view2_alpha = view2_alpha + dx; view2_beta = view2_beta + dy; 
                } else if (active_view_id == 2) { 
                    view3_alpha = view3_alpha + dx; view3_beta = view3_beta + dy; 
                }
            } 
            else if (current_mode == MODE_TRIPLE) {
                if (active_view_id == 0) { view1_alpha = view1_alpha + dx; view1_beta = view1_beta + dy; }
                else if (active_view_id == 1) { view2_alpha = view2_alpha + dx; view2_beta = view2_beta + dy; }
                else { view3_alpha = view3_alpha + dx; view3_beta = view3_beta + dy; }
            }
            last_mouse_x = event->x; last_mouse_y = event->y; 
        } 
        return TRUE; 
    }

    #define UPDATE_PTZ(VAR, VAL) \
        if (current_mode == MODE_SINGLE) cur_##VAR = cur_##VAR + VAL; \
        else if (current_mode == MODE_PANORAMA) pano_##VAR = pano_##VAR + VAL; \
        else if (active_view_id == 0) view1_##VAR = view1_##VAR + VAL; \
        else if (active_view_id == 1) view2_##VAR = view2_##VAR + VAL; \
        else if (active_view_id == 2) view3_##VAR = view3_##VAR + VAL; \
        else if (active_view_id == 3) pano_##VAR = pano_##VAR + VAL;

    void on_btn_alpha_inc_clicked(GtkButton *b) { UPDATE_PTZ(alpha, 1.0f); }
    void on_btn_alpha_dec_clicked(GtkButton *b) { UPDATE_PTZ(alpha, -1.0f); }
    void on_btn_beta_inc_clicked(GtkButton *b)  { UPDATE_PTZ(beta, 1.0f); }
    void on_btn_beta_dec_clicked(GtkButton *b)  { UPDATE_PTZ(beta, -1.0f); }
    void on_btn_zoom_in_clicked(GtkButton *b)   { UPDATE_PTZ(zoom, 0.1f); }
    void on_btn_zoom_out_clicked(GtkButton *b)  { UPDATE_PTZ(zoom, -0.1f); }
}

int main(int argc, char** argv) {
    try {
        signal(SIGINT, signal_handler);
        putenv((char*)"GST_DEBUG=*:0");
        moildev_app_get_resource();
        
        if(argc < 2) { cout << "./app <file> <res_id (0-4)>" << endl; return -1; }
        if (argc >= 3) SEL_RES_IDX = atoi(argv[2]);
        if (SEL_RES_IDX < 0 || SEL_RES_IDX >= BENCHMARK_RES.size()) SEL_RES_IDX = 0;
        if(SEL_RES_IDX == 3) BUFFER_SIZE = 2; 
        
        OUT_W = BENCHMARK_RES[SEL_RES_IDX].w; OUT_H = BENCHMARK_RES[SEL_RES_IDX].h;

        string src_param = argv[1];
        if (!isdigit(src_param[0])) {
            // Input adalah FILE VIDEO (String path)
            IS_CAMERA_MODE = false; 
            
            VideoCapture probe(src_param);
            if(probe.isOpened()) {
                IN_W = (int)probe.get(CAP_PROP_FRAME_WIDTH); IN_H = (int)probe.get(CAP_PROP_FRAME_HEIGHT);
                probe.release();
            } else { IN_W = DEFAULT_W; IN_H = DEFAULT_H; }
        } else { 
            // Input adalah KAMERA (Angka index, misal 0, 1)
            IS_CAMERA_MODE = true; 
            
            IN_W = OUT_W; IN_H = OUT_H; 
        }

        gtk_init(&argc, &argv);
        GtkBuilder *builder = gtk_builder_new();
        if (!gtk_builder_add_from_resource(builder, "/moildev/app/gui.glade", NULL)) return 1;
        main_window = GTK_WIDGET(gtk_builder_get_object(builder, "main_window"));
        img_display = GTK_WIDGET(gtk_builder_get_object(builder, "img_display"));
        gtk_window_fullscreen(GTK_WINDOW(main_window));
        gtk_builder_connect_signals(builder, NULL);
        g_object_unref(builder);

        loadConfig();

        unsetenv("LD_LIBRARY_PATH"); initDRP();
        
        // Alokasi Buffer Pools
        inputPool.resize(BUFFER_SIZE); 
        processPool.resize(BUFFER_SIZE); // Buffer Tengah (BGR)
        outputPool.resize(BUFFER_SIZE);
        
        for(int i=0; i<BUFFER_SIZE; i++) { 
            // Input Pool: YUYV (2 Byte/Pixel)
            allocateDRPBuffer(inputPool[i], IN_W, IN_H, CV_8UC2); 
            // Process Pool: BGR (3 Byte/Pixel) untuk input AI & Remap
            allocateDRPBuffer(processPool[i], IN_W, IN_H, CV_8UC3);
            // Output Pool: BGR (3 Byte/Pixel) untuk Display
            allocateDRPBuffer(outputPool[i], OUT_W, OUT_H, CV_8UC3); 
        }
        for(int i=0; i<MAP_BUFFER_COUNT; i++) { 
            allocateDRPBuffer(map1_bufs[i], OUT_W, OUT_H, CV_16SC2); 
            allocateDRPBuffer(map2_bufs[i], OUT_W, OUT_H, CV_16UC1); 
        }

        cout << "[INIT] Initializing DRP-AI YOLO... (Please Wait)" << endl;
        int drpai_fd = open("/dev/drpai0", O_RDWR);
        if (drpai_fd >= 0) {
            uint64_t drp_addr = get_drpai_start_addr(drpai_fd);
            close(drpai_fd);
            if (!yolo.init(model_dir, drp_addr)) {
                cerr << "[ERROR] Failed to init YOLO model!" << endl;
                return -1;
            }
        } else {
            cerr << "[ERROR] Failed to open /dev/drpai0" << endl;
            return -1;
        }
        cout << "[INIT] YOLO Ready!" << endl;
        
        gtk_widget_show_all(main_window);
        initRhoLUT();

        thread t1(captureThread, argv[1]); 
        this_thread::sleep_for(chrono::milliseconds(200)); 
        
        thread t2(mapUpdateThread); 
        thread t3(pipelineThread); 

        gtk_main();
        
        isRunning = false; 
        cv_drp.notify_all(); 
        if(t1.joinable()) t1.join(); 
        if(t2.joinable()) t2.join(); 
        if(t3.joinable()) t3.join(); 
        
        cleanup_resources();
        return 0;

    } catch (const std::exception& e) { return -1; }
}