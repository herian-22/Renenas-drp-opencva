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
int DEFAULT_W = 1280;
int DEFAULT_H = 960;
int IN_W = DEFAULT_W; 
int IN_H = DEFAULT_H;

int OUT_W = 1024, OUT_H = 768; 
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
atomic<double> stats_map_ms(0.0);
atomic<double> stats_ai_ms(0.0);
atomic<double> stats_remap_ms(0.0);
atomic<double> stats_fps(0.0);

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
    char cwd[PATH_MAX]; 
    if(getcwd(cwd, sizeof(cwd)) != NULL) setenv("DRP_EXE_PATH", cwd, 1);
    // Load library OCA Renesas
    lib_handle = dlopen("/usr/lib/aarch64-linux-gnu/renesas/libopencv_imgproc.so.4.1.0", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!lib_handle) {
        cerr << "[ERROR] Failed to load libopencv_imgproc.so (OCA)! Check SDK." << endl;
        exit(1);
    }
    OCA_Activate_Func OCA_Activate_Ptr = (OCA_Activate_Func)dlsym(lib_handle, "_Z12OCA_ActivatePm");
    if (OCA_Activate_Ptr) {
        unsigned long OCA_list[17] = {0}; OCA_list[0]=1; OCA_list[16]=1;
        OCA_Activate_Ptr(OCA_list);
        cout << "[INIT] DRP-OCA Activated! cv::cvt & cv::remap will be HW Accelerated." << endl;
    } else {
        cerr << "[ERROR] OCA_Activate symbol not found!" << endl;
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

// --- MOIL MATH & MAP UPDATE ---
class MoilNative {
public:
    double iCx, iCy, scale_x, scale_y;
    MoilNative() {
        scale_x = (double)IN_W / ORG_W; scale_y = (double)IN_H / ORG_H;
        iCx = ORG_ICX * scale_x; iCy = ORG_ICY * scale_y;
    }
    double getRho(double alpha) {
        double p0=PARA[0], p1=PARA[1], p2=PARA[2], p3=PARA[3], p4=PARA[4], p5=PARA[5];
        return (((((p0*alpha+p1)*alpha+p2)*alpha+p3)*alpha+p4)*alpha+p5)*alpha;
    }

    void fillRegion(Mat &mx, Mat &my, Rect r, float alpha, float beta, float zoom, bool isPano, bool flip_h, bool flip_v) {
        const double PI = 3.14159265358979323846;
        #pragma omp parallel for collapse(2)
        for(int y=0; y<r.height; y++) {
            for(int x=0; x<r.width; x++) {
                double rho, theta;
                if(isPano) {
                    double nx=(double)x/r.width, ny=(double)y/r.height;
                    theta=(nx-0.5)*2.0*PI + (beta * PI / 180.0);
                    rho=getRho(ny*(alpha*PI/180.0));
                } else {
                    double f=r.width/(2.0*tan(30.0*PI/180.0/zoom));
                    double ar=alpha*PI/180.0, br=beta*PI/180.0;
                    double cx=x-r.width/2.0, cy=y-r.height/2.0;
                    double x1=cx, y1=cy*cos(ar)-f*sin(ar), z1=cy*sin(ar)+f*cos(ar);
                    double x2=x1*cos(br)-y1*sin(br), y2=x1*sin(br)+y1*cos(br), z2=z1;
                    rho=getRho(atan2(sqrt(x2*x2+y2*y2),z2)); theta=atan2(y2,x2);
                }

                int target_x = flip_h ? (r.width - 1 - x) : x;
                int target_y = flip_v ? (r.height - 1 - y) : y;
                int py = r.y + target_y;
                int px = r.x + target_x;

                if(px>=0 && px<mx.cols && py>=0 && py<mx.rows) {
                    mx.at<float>(py,px)=(float)(iCx+(rho*scale_x)*cos(theta));
                    my.at<float>(py,px)=(float)(iCy+(rho*scale_y)*sin(theta));
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

        // Cek apakah ada perubahan posisi/mode (Logic sama seperti original)
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

        // Jika tidak ada perubahan, tidur sebentar (Hemat CPU)
        if (maps_ready && !changed) { 
            this_thread::sleep_for(chrono::milliseconds(20)); continue; 
        }

        auto t1 = chrono::steady_clock::now();
        int back_idx = (active_map_idx.load() + 1) % MAP_BUFFER_COUNT;
        
        // Reset buffer kecil
        mx_small.setTo(Scalar(-1)); my_small.setTo(Scalar(-1));

        // Kalkulasi di koordinat KECIL (CALC_W x CALC_H)
        switch(mode) {
            case MODE_MAIN: {
                int split_y = CALC_H / 2;
                int third_w = CALC_W / 3;
                
                moil.fillRegion(mx_small, my_small, Rect(0, 0, CALC_W, split_y), pano_alpha.load(), pano_beta.load(), 0, true, pano_flip_h, pano_flip_v);
                moil.fillRegion(mx_small, my_small, Rect(0, split_y, third_w, split_y), view1_alpha, view1_beta, view1_zoom, false, view1_flip_h, view1_flip_v);
                moil.fillRegion(mx_small, my_small, Rect(third_w, split_y, third_w, split_y), view2_alpha, view2_beta, view2_zoom, false, view2_flip_h, view2_flip_v);
                moil.fillRegion(mx_small, my_small, Rect(2*third_w, split_y, third_w, split_y), view3_alpha, view3_beta, view3_zoom, false, view3_flip_h, view3_flip_v);
                
                // Update Cache
                l_v1a = view1_alpha; l_v1b = view1_beta; l_v1z = view1_zoom; l_v1fh = view1_flip_h; l_v1fv = view1_flip_v;
                l_v2a = view2_alpha; l_v2b = view2_beta; l_v2z = view2_zoom; l_v2fh = view2_flip_h; l_v2fv = view2_flip_v;
                l_v3a = view3_alpha; l_v3b = view3_beta; l_v3z = view3_zoom; l_v3fh = view3_flip_h; l_v3fv = view3_flip_v;
                l_pa = pano_alpha; l_pb = pano_beta; l_pfh = pano_flip_h; l_pfv = pano_flip_v;
                break;
            }
            case MODE_TRIPLE: {
                int split_w = CALC_W / 3; 
                moil.fillRegion(mx_small, my_small, Rect(0, 0, split_w, CALC_H), view1_alpha, view1_beta, view1_zoom, false, view1_flip_h, view1_flip_v);
                moil.fillRegion(mx_small, my_small, Rect(split_w, 0, split_w, CALC_H), view2_alpha, view2_beta, view2_zoom, false, view2_flip_h, view2_flip_v);
                moil.fillRegion(mx_small, my_small, Rect(2*split_w, 0, split_w, CALC_H), view3_alpha, view3_beta, view3_zoom, false, view3_flip_h, view3_flip_v);
                
                l_v1a = view1_alpha; l_v1b = view1_beta; l_v1z = view1_zoom; l_v1fh = view1_flip_h; l_v1fv = view1_flip_v;
                l_v2a = view2_alpha; l_v2b = view2_beta; l_v2z = view2_zoom; l_v2fh = view2_flip_h; l_v2fv = view2_flip_v;
                l_v3a = view3_alpha; l_v3b = view3_beta; l_v3z = view3_zoom; l_v3fh = view3_flip_h; l_v3fv = view3_flip_v;
                break;
            }
            case MODE_ORIGINAL: 
                moil.fillStandard(mx_small, my_small, Rect(0, 0, CALC_W, CALC_H)); 
                break;
            case MODE_PANORAMA: {
                moil.fillRegion(mx_small, my_small, Rect(0, 0, CALC_W, CALC_H), pano_alpha.load(), pano_beta.load(), 0, true, pano_flip_h, pano_flip_v);
                l_pa = pano_alpha; l_pb = pano_beta; l_pfh = pano_flip_h; l_pfv = pano_flip_v;
                break;
            }
            case MODE_GRID: {
                int bw = CALC_W/2, bh = CALC_H/2;
                moil.fillRegion(mx_small, my_small, Rect(0, 0, bw, bh), 0, 0, 2.0, false, false, false);
                moil.fillRegion(mx_small, my_small, Rect(bw, 0, bw, bh), 45, 0, 2.0, false, false, false);
                moil.fillRegion(mx_small, my_small, Rect(0, bh, bw, bh), 45, -90, 2.0, false, false, false);
                moil.fillRegion(mx_small, my_small, Rect(bw, bh, bw, bh), 45, 90, 2.0, false, false, false);
                break;
            }
            case MODE_SINGLE: {
                // Kalkulasi di buffer kecil saja agar konsisten
                moil.fillRegion(mx_small, my_small, Rect(0, 0, CALC_W, CALC_H), cur_alpha, cur_beta, cur_zoom, false, cur_flip_h, cur_flip_v);
                l_a = cur_alpha; l_b = cur_beta; l_z = cur_zoom; l_fh = cur_flip_h; l_fv = cur_flip_v;
                break;
            }
        }
        
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
    if (mode == MODE_MAIN) {
        // Layout: Top half is Panorama, Bottom half is split into 3 views
        int split_y = OUT_H / 2;
        int third_w = OUT_W / 3;
        regions.push_back(Rect(0, 0, OUT_W, split_y));            // Top Panorama
        regions.push_back(Rect(0, split_y, third_w, split_y));    // Bottom Left
        regions.push_back(Rect(third_w, split_y, third_w, split_y)); // Bottom Center
        regions.push_back(Rect(2*third_w, split_y, third_w, split_y)); // Bottom Right
    } else if (mode == MODE_TRIPLE) {
        // Layout: Screen split vertically into 3 columns
        int w = OUT_W / 3;
        regions.push_back(Rect(0, 0, w, OUT_H)); 
        regions.push_back(Rect(w, 0, w, OUT_H)); 
        regions.push_back(Rect(2*w, 0, w, OUT_H));
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
    pipe = "v4l2src device=/dev/video" + src +
           " ! image/jpeg, width=" + to_string(IN_W) + ", height=" + to_string(IN_H) + 
           " ! jpegdec"                 // Decode JPEG
           " ! videoconvert"            // <--- WAJIB: Konverter warna otomatis
           " ! video/x-raw, format=YUY2"// <--- WAJIB: Paksa output jadi YUY2 (sesuai code C++)
           " ! queue max-size-buffers=2"
           " ! appsink drop=true sync=false";
           
    cout << "[CAPTURE] Mode: Camera Live (MJPEG Corrected)" << endl;
    } else {
        pipe = "filesrc location=" + src +
               " ! decodebin ! queue"
               " ! videoconvert ! video/x-raw,format=YUY2"
               " ! appsink max-buffers=2 drop=true sync=true";
        cout << "[CAPTURE] Mode: Video File (Follow Original FPS)" << endl;
    }

    VideoCapture *cap = new VideoCapture(pipe, CAP_GSTREAMER);
    if (!cap->isOpened()) { 
        cerr << "[ERROR] Gagal membuka source: " << src << endl;
        isRunning = false; 
        return; 
    }

    Mat temp;
    while (isRunning) {
        if (!cap->read(temp) || temp.empty()) { 
            if (is_file) {
                cout << "[CAPTURE] Video Ended. Replaying..." << endl;
                cap->set(CAP_PROP_POS_FRAMES, 0); 
            }
            continue; 
        }

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

    // Ambil ukuran widget
    int avail_w = gtk_widget_get_allocated_width(img_display);
    int avail_h = gtk_widget_get_allocated_height(img_display);

    // Ambil ukuran layar asli
    GdkScreen *screen = gdk_screen_get_default();
    int screen_w = gdk_screen_get_width(screen);
    int screen_h = gdk_screen_get_height(screen);

    // // Log semua ukuran
    // std::cout << "\n=== DISPLAY SIZE INFO ===" << std::endl;
    // std::cout << "Screen size  : " << screen_w << " x " << screen_h << std::endl;
    // std::cout << "Widget size  : " << avail_w << " x " << avail_h << std::endl;
    // std::cout << "Source (Mat) : " << src.cols << " x " << src.rows << std::endl;

    // Kalau widget belum valid, pakai ukuran layar
    if (avail_w < 1) avail_w = screen_w;
    if (avail_h < 1) avail_h = screen_h;

    int dest_w = avail_w;   // FULL LEBAR
    int dest_h = 1150;       // TINGGI DIKUNCI

    // std::cout << "Render size  : " << dest_w << " x " << dest_h << std::endl;
    // std::cout << "===========================\n" << std::endl;

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
        GdkPixbuf *scaled = gdk_pixbuf_scale_simple(
            pixbuf,
            dest_w,
            dest_h,
            GDK_INTERP_NEAREST
        );

        gtk_image_set_from_pixbuf(GTK_IMAGE(img_display), scaled);
        g_object_unref(scaled);
        g_object_unref(pixbuf);
    }

    outputPool[idx].ready = false;
    ui_update_pending = false;
    return FALSE;
}



void pipelineThread() {
    int frame_count = 0; 
    auto last_fps_time = chrono::steady_clock::now();

    // Variable untuk menyimpan hasil deteksi frame sebelumnya (untuk mode Kamera)
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
        
        // === 1. Convert YUYV (Cam) -> BGR (Process Buffer) ===
        // Selalu dijalankan karena dibutuhkan untuk Remap (Display) dan AI
        auto t_conv_start = chrono::steady_clock::now();
        cv::cvtColor(inputPool[idx_in].data, processPool[idx_in].data, COLOR_YUV2BGR_YUY2);
        buffer_flush_dmabuf(processPool[idx_in].dbuf.idx, processPool[idx_in].dbuf.size);
        
        // === 2. Run DRP-AI YOLO (LOGIKA BARU) ===
        bool run_inference = true;

        if (IS_CAMERA_MODE) {
            // --- MODE KAMERA (LIGHTER MODE) ---
            // Kita skip AI inference untuk mengejar 30 FPS.
            // Jalankan AI hanya setiap 3 frame (Frame 0: Run, Frame 1: Skip, Frame 2: Skip)
            if (ai_skip_counter % 3 != 0) {
                run_inference = false;
            }
            ai_skip_counter++;
        } else {
            // --- MODE VIDEO FILE (HEAVY MODE) ---
            // Jalankan AI di setiap frame seperti kode asli
            run_inference = true;
        }

        if (run_inference) {
            // Berat: Jalankan DRP-AI
            results = yolo.run_detection(processPool[idx_in].data, ai_stats);
            
            // Simpan hasil untuk frame berikutnya yang di-skip
            cached_results = results; 
            stats_ai_ms.store(ai_stats.total);
        } else {
            // Ringan: Pakai hasil frame sebelumnya (tanpa membebani DRP-AI)
            results = cached_results;
            
            // Set stats 0 atau nilai terakhir agar log tidak bingung
            ai_stats.total = 0; 
        }

        // === 3. Remap Fisheye -> Output Buffer (Display) ===
        auto t_remap_start = chrono::steady_clock::now();
        if (maps_ready && !map1_bufs[map_idx].data.empty()) {
            cv::remap(processPool[idx_in].data, outputPool[idx_out].data, map1_bufs[map_idx].data, map2_bufs[map_idx].data, INTER_NEAREST, BORDER_CONSTANT);
        } else {
            processPool[idx_in].data.copyTo(outputPool[idx_out].data); 
        }
        auto t_remap_end = chrono::steady_clock::now();
        stats_remap_ms.store(chrono::duration_cast<chrono::milliseconds>(t_remap_end - t_remap_start).count());

        
        // === 4. Draw Boxes (CPU - Ringan) ===
        if (maps_ready && !map1_bufs[map_idx].data.empty()) {
            // Gambar kotak (baik hasil AI baru maupun hasil cache)
            drawBBoxesOnOutput(outputPool[idx_out].data, map1_bufs[map_idx].data, results, current_mode.load());
        }

        // Convert ke RGB untuk Display GTK
        cv::cvtColor(outputPool[idx_out].data, outputPool[idx_out].data, COLOR_BGR2RGB);
        
        // --- FPS COUNTER ---
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

            printf("\n=================[ PERFORMANCE LOG ]=================\n");
            printf(" [GENERAL]\n");
            printf("  > Source    : %s\n", IS_CAMERA_MODE ? "Camera (Lighter Mode)" : "Video File (Full AI)");
            printf("  > Mode      : %s\n", mode_name.c_str());
            printf("  > FPS       : %d\n", (int)current_fps);
            printf("  > Input     : %dx%d\n", IN_W, IN_H);
            printf("  > Output    : %dx%d\n", OUT_W, OUT_H);
            printf(" ----------------------------------------------------\n");
            printf(" [PIPELINE LATENCY]\n");
            printf("  > Read      : %.2f ms\n", stats_read_ms.load());
            printf("  > Map       : %.2f ms\n", stats_map_ms.load());
            printf("  > Remap     : %.2f ms\n", stats_remap_ms.load());
            // Total waktu sedikit bias jika AI diskip, tapi ini memberi gambaran latency per frame display
            printf("  > Total     : %.2f ms\n", stats_read_ms.load() + stats_ai_ms.load() + stats_remap_ms.load());
            printf(" ----------------------------------------------------\n");
            printf(" [AI DETAILS]\n");
            if (IS_CAMERA_MODE && !run_inference) {
                 printf("  > Status    : CACHED (Skipped for Performance)\n");
            } else {
                 printf("  > Status    : RUNNING\n");
                 printf("  > AI Total  : %.2f ms\n", ai_stats.total);
                 printf("  > Objects   : %d\n", ai_stats.count);
            }
            printf("=====================================================\n");
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
    void on_btn_pano_clicked(GtkButton *b) { current_mode = MODE_PANORAMA; }
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