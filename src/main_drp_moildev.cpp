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
#include <linux/drpai.h>
#include <fstream> 

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
    {1024, 768,  "0.8 MP (HD)"},    
    {1600, 1200, "2.0 MP (FHD)"},   
    {1920, 1080, "2.0 MP (1080p)"}, 
    {2592, 1944, "5.0 MP (2K/5MP)"},
    {1440, 480,  "Triple Strip (3x480)"}
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

// 1. Single View
atomic<float> cur_alpha(0.0f), cur_beta(0.0f), cur_zoom(2.0f);
atomic<bool> cur_flip_h(false), cur_flip_v(false);

// 2. Panorama 
atomic<float> pano_alpha(110.0f), pano_beta(0.0f), pano_zoom(1.0f);
atomic<bool> pano_flip_h(false), pano_flip_v(false);

// 3. Multi View (Anypoint 1, 2, 3)
atomic<float> view1_alpha(-45.0f), view1_beta(0.0f), view1_zoom(3.5f); // Kiri
atomic<bool>  view1_flip_h(false), view1_flip_v(false);

atomic<float> view2_alpha(0.0f),   view2_beta(0.0f), view2_zoom(3.5f); // Tengah
atomic<bool>  view2_flip_h(false), view2_flip_v(false);

atomic<float> view3_alpha(45.0f),  view3_beta(0.0f), view3_zoom(3.5f); // Kanan
atomic<bool>  view3_flip_h(false), view3_flip_v(false);

// ID View Aktif: 
// 0=View1, 1=View2, 2=View3
// 3=Panorama (Saat di Mode Main bagian atas diklik)
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
struct FrameBuffer { Mat data; dma_buffer dbuf; bool ready; };
int BUFFER_SIZE = 3; 
vector<FrameBuffer> inputPool;
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

void* lib_handle = nullptr;
typedef int (*OCA_Activate_Func)(unsigned long*);
void initDRP() {
    char cwd[PATH_MAX]; 
    if(getcwd(cwd, sizeof(cwd)) != NULL) setenv("DRP_EXE_PATH", cwd, 1);
    lib_handle = dlopen("/usr/lib/aarch64-linux-gnu/renesas/libopencv_imgproc.so.4.1.0", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!lib_handle) exit(1);
    OCA_Activate_Func OCA_Activate_Ptr = (OCA_Activate_Func)dlsym(lib_handle, "_Z12OCA_ActivatePm");
    unsigned long OCA_list[17] = {0}; OCA_list[0]=1; OCA_list[16]=1;
    OCA_Activate_Ptr(OCA_list);
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
        cout << "[CONFIG] Configuration SAVED to " << CONFIG_FILE << endl;
    } else {
        cerr << "[CONFIG] Error saving config file!" << endl;
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
        cout << "[CONFIG] Configuration LOADED from " << CONFIG_FILE << endl;
    } else {
        cout << "[CONFIG] No config file found. Using defaults." << endl;
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

                // Logic Flip
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
    MoilNative moil;
    Mat mx(OUT_H, OUT_W, CV_32F);
    Mat my(OUT_H, OUT_W, CV_32F);
    ViewMode last_mode = MODE_SINGLE; 
    
    // Cache variables
    float l_a=-99, l_b=-99, l_z=-99; bool l_fh=false, l_fv=false;
    float l_pa=-99, l_pb=-99; bool l_pfh=false, l_pfv=false;
    float l_v1a=-99, l_v1b=-99, l_v1z=-99; bool l_v1fh=false, l_v1fv=false;
    float l_v2a=-99, l_v2b=-99, l_v2z=-99; bool l_v2fh=false, l_v2fv=false;
    float l_v3a=-99, l_v3b=-99, l_v3z=-99; bool l_v3fh=false, l_v3fv=false;

    while(isRunning) {
        if (is_paused.load()) { this_thread::sleep_for(chrono::milliseconds(100)); continue; }

        ViewMode mode = current_mode.load();
        bool changed = false;

        if(mode != last_mode) changed = true;
        else if (mode == MODE_SINGLE) {
            if(cur_alpha!=l_a || cur_beta!=l_b || cur_zoom!=l_z || cur_flip_h!=l_fh || cur_flip_v!=l_fv) changed = true;
        } 
        else if (mode == MODE_PANORAMA) {
            if(pano_alpha!=l_pa || pano_beta!=l_pb || pano_flip_h!=l_pfh || pano_flip_v!=l_pfv) changed = true;
        }
        else if (mode == MODE_MAIN) {
            // Cek 3 View + Pano
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

        if (maps_ready && !changed) { 
            this_thread::sleep_for(chrono::milliseconds(20)); continue; 
        }

        auto t1 = chrono::steady_clock::now();
        int back_idx = (active_map_idx.load() + 1) % MAP_BUFFER_COUNT;
        mx.setTo(Scalar(-1)); my.setTo(Scalar(-1));

        switch(mode) {
            case MODE_MAIN: {
                // Layout: Atas Pano, Bawah 3 Anypoint
                int split_y = OUT_H / 2;
                int third_w = OUT_W / 3; // Bagi lebar jadi 3

                // Pano Atas
                moil.fillRegion(mx, my, Rect(0, 0, OUT_W, split_y), pano_alpha.load(), pano_beta.load(), 0, true, pano_flip_h, pano_flip_v);
                
                // View 1 (Kiri Bawah)
                moil.fillRegion(mx, my, Rect(0, split_y, third_w, split_y), view1_alpha, view1_beta, view1_zoom, false, view1_flip_h, view1_flip_v);
                
                // View 2 (Tengah Bawah)
                moil.fillRegion(mx, my, Rect(third_w, split_y, third_w, split_y), view2_alpha, view2_beta, view2_zoom, false, view2_flip_h, view2_flip_v);
                
                // View 3 (Kanan Bawah) - Baru
                moil.fillRegion(mx, my, Rect(2*third_w, split_y, third_w, split_y), view3_alpha, view3_beta, view3_zoom, false, view3_flip_h, view3_flip_v);

                // Update Cache
                l_v1a = view1_alpha; l_v1b = view1_beta; l_v1z = view1_zoom; l_v1fh = view1_flip_h; l_v1fv = view1_flip_v;
                l_v2a = view2_alpha; l_v2b = view2_beta; l_v2z = view2_zoom; l_v2fh = view2_flip_h; l_v2fv = view2_flip_v;
                l_v3a = view3_alpha; l_v3b = view3_beta; l_v3z = view3_zoom; l_v3fh = view3_flip_h; l_v3fv = view3_flip_v;
                l_pa = pano_alpha; l_pb = pano_beta; l_pfh = pano_flip_h; l_pfv = pano_flip_v;
                break;
            }

            case MODE_TRIPLE: {
                int split_w = OUT_W / 3; 
                moil.fillRegion(mx, my, Rect(0, 0, split_w, OUT_H), view1_alpha, view1_beta, view1_zoom, false, view1_flip_h, view1_flip_v);
                moil.fillRegion(mx, my, Rect(split_w, 0, split_w, OUT_H), view2_alpha, view2_beta, view2_zoom, false, view2_flip_h, view2_flip_v);
                moil.fillRegion(mx, my, Rect(2*split_w, 0, split_w, OUT_H), view3_alpha, view3_beta, view3_zoom, false, view3_flip_h, view3_flip_v);
                
                l_v1a = view1_alpha; l_v1b = view1_beta; l_v1z = view1_zoom; l_v1fh = view1_flip_h; l_v1fv = view1_flip_v;
                l_v2a = view2_alpha; l_v2b = view2_beta; l_v2z = view2_zoom; l_v2fh = view2_flip_h; l_v2fv = view2_flip_v;
                l_v3a = view3_alpha; l_v3b = view3_beta; l_v3z = view3_zoom; l_v3fh = view3_flip_h; l_v3fv = view3_flip_v;
                break;
            }

            case MODE_ORIGINAL: 
                moil.fillStandard(mx, my, Rect(0, 0, OUT_W, OUT_H)); 
                break;

            case MODE_PANORAMA: {
                moil.fillRegion(mx, my, Rect(0, 0, OUT_W, OUT_H), pano_alpha.load(), pano_beta.load(), 0, true, pano_flip_h, pano_flip_v);
                l_pa = pano_alpha; l_pb = pano_beta; l_pfh = pano_flip_h; l_pfv = pano_flip_v;
                break;
            }

            case MODE_GRID: {
                int bw = OUT_W/2, bh = OUT_H/2;
                moil.fillRegion(mx, my, Rect(0, 0, bw, bh), 0, 0, 2.0, false, false, false);
                moil.fillRegion(mx, my, Rect(bw, 0, bw, bh), 45, 0, 2.0, false, false, false);
                moil.fillRegion(mx, my, Rect(0, bh, bw, bh), 45, -90, 2.0, false, false, false);
                moil.fillRegion(mx, my, Rect(bw, bh, bw, bh), 45, 90, 2.0, false, false, false);
                break;
            }

            case MODE_SINGLE: {
                int scale = 4; int sw = OUT_W / scale; int sh = OUT_H / scale;
                Mat smx(sh, sw, CV_32F); Mat smy(sh, sw, CV_32F);
                moil.fillRegion(smx, smy, Rect(0, 0, sw, sh), cur_alpha, cur_beta, cur_zoom, false, cur_flip_h, cur_flip_v);
                resize(smx, mx, Size(OUT_W, OUT_H), 0, 0, INTER_LINEAR);
                resize(smy, my, Size(OUT_W, OUT_H), 0, 0, INTER_LINEAR);
                l_a = cur_alpha; l_b = cur_beta; l_z = cur_zoom; l_fh = cur_flip_h; l_fv = cur_flip_v;
                break;
            }
        }
        
        auto t2 = chrono::steady_clock::now();
        stats_map_ms.store(chrono::duration_cast<chrono::milliseconds>(t2 - t1).count());

        convertMaps(mx, my, map1_bufs[back_idx].data, map2_bufs[back_idx].data, CV_16SC2, false);
        buffer_flush_dmabuf(map1_bufs[back_idx].dbuf.idx, map1_bufs[back_idx].dbuf.size);
        buffer_flush_dmabuf(map2_bufs[back_idx].dbuf.idx, map2_bufs[back_idx].dbuf.size);
        active_map_idx.store(back_idx);
        if (!maps_ready) maps_ready = true;
        last_mode = mode;
    }
}

void captureThread(string src) {
    string pipe;

    if (isdigit(src[0])) {
        // === KAMERA (/dev/videoX) ===
        // Kita paksa resolusi kamera agar sesuai dengan IN_W dan IN_H
        // yang sudah diset di main() berdasarkan resolusi benchmark yang dipilih.
        pipe = "v4l2src device=/dev/video" + src +
                " ! image/jpeg, width=" + to_string(IN_W) +
                ", height=" + to_string(IN_H) +
                " ! jpegdec ! videoconvert ! video/x-raw,format=BGR"
                " ! appsink drop=true sync=false";
        
        cout << "[CAPTURE] Camera Force Resolution: " << IN_W << "x" << IN_H << endl;
    } 
    else {
        // === VIDEO FILE ===
        // Decodebin otomatis menyesuaikan resolusi file asli
        pipe = "filesrc location=" + src +
               " ! decodebin ! queue"
               " ! videoconvert ! video/x-raw,format=BGR"
               " ! appsink max-buffers=2 drop=true sync=false";
    }

    cout << "[CAPTURE] Pipeline: " << pipe << endl;
    VideoCapture *cap = new VideoCapture(pipe, CAP_GSTREAMER);

    if (!cap->isOpened()) { 
        cerr << "[ERROR] Gagal membuka kamera/file: " << src << endl;
        isRunning = false; 
        return; 
    }

    // Cek resolusi aktual yang didapat (kadang kamera menolak jika hardware tidak support)
    int actual_w = (int)cap->get(CAP_PROP_FRAME_WIDTH);
    int actual_h = (int)cap->get(CAP_PROP_FRAME_HEIGHT);
    cout << "[INFO] Source Opened Resolution: " << actual_w << " x " << actual_h << endl;

    Mat temp;
    while(isRunning) {
        auto t1 = chrono::steady_clock::now();
        
        if (!cap->read(temp) || temp.empty()) {
             // Jika file habis, ulang dari awal
             if (!isdigit(src[0])) cap->set(CAP_PROP_POS_FRAMES, 0);
             continue;
        }

        auto t2 = chrono::steady_clock::now();
        stats_read_ms.store(chrono::duration_cast<chrono::milliseconds>(t2 - t1).count());

        // Jika resolusi kamera yang didapat ternyata beda (hardware limitation),
        // Kita resize paksa di sini agar tidak crash di DRP-AI
        if (temp.cols != IN_W || temp.rows != IN_H) {
            resize(temp, temp, Size(IN_W, IN_H));
        }

        int idx = w_idx_cap;
        temp.copyTo(inputPool[idx].data);
        
        buffer_flush_dmabuf(inputPool[idx].dbuf.idx, inputPool[idx].dbuf.size);

        {
            lock_guard<mutex> lock(mtxData);
            inputPool[idx].ready = true;
            w_idx_cap = (w_idx_cap + 1) % BUFFER_SIZE;
            r_idx_drp = idx;
        }

        cv_drp.notify_one();
    }

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

void drawInfoBox(Mat &img) {
    // int start_y = 30;
    // int line_h  = 25;

    // // vector<string> lines;
    // // if (is_paused) 
    // //     lines.push_back("MODE: PAUSED");
    // // else 
    // //     lines.push_back("MODE: Running");

    // int box_w = 240;
    // int box_h = lines.size() * line_h + 10;

    // Mat roi = img(Rect(10, 5, box_w, box_h));
    // Mat color(roi.size(), CV_8UC3, Scalar(0, 0, 0));
    // addWeighted(color, 0.6, roi, 0.4, 0.0, roi);

    // // for (size_t i = 0; i < lines.size(); i++) {
    // //     putText(img, lines[i], Point(21, start_y + i*line_h + 1),
    // //             FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 2);
    // //     putText(img, lines[i], Point(20, start_y + i*line_h),
    // //             FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 1.5);
    // // }
}


gboolean update_gui_image(gpointer user_data) {
    int idx = (int)(intptr_t)user_data;
    if(idx < 0 || idx >= BUFFER_SIZE) { ui_update_pending = false; return FALSE; }
    Mat &src = outputPool[idx].data;
    
    int avail_w = gtk_widget_get_allocated_width(img_display);
    int avail_h = gtk_widget_get_allocated_height(img_display);
    if (avail_w < 100) avail_w = 1024;
    if (avail_h < 100) avail_h = 768;
    int dest_w, dest_h; float ratio = (float)OUT_W / (float)OUT_H;
    
    if ((float)avail_w / avail_h > ratio) { dest_h = avail_h; dest_w = (int)(dest_h * ratio); } 
    else { dest_w = avail_w; dest_h = (int)(dest_w / ratio); }

    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(src.data, GDK_COLORSPACE_RGB, FALSE, 8, src.cols, src.rows, src.step, NULL, NULL);
    if (pixbuf) { 
        GdkPixbuf *scaled = gdk_pixbuf_scale_simple(pixbuf, dest_w, dest_h, GDK_INTERP_NEAREST);
        gtk_image_set_from_pixbuf(GTK_IMAGE(img_display), scaled); 
        g_object_unref(scaled); g_object_unref(pixbuf); 
    }
    outputPool[idx].ready = false; ui_update_pending = false; 
    return FALSE;
}

void pipelineThread() {
    int drpai_fd = open("/dev/drpai0", O_RDWR);
    uint64_t drp_addr = get_drpai_start_addr(drpai_fd);
    if (drp_addr == 0 || !yolo.init(model_dir, drp_addr)) return;
    close(drpai_fd);

    int frame_count = 0; auto last_fps_time = chrono::steady_clock::now();

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
            cv_drp.wait(lock, []{ return inputPool[r_idx_drp].ready || !isRunning; }); 
            if(!isRunning) break; 
            idx_in = r_idx_drp; inputPool[idx_in].ready = false; 
        }
        
        int idx_out = w_idx_drp;
        int map_idx = active_map_idx.load();
        
        AiStats ai_stats;
        vector<detection> results = yolo.run_detection(inputPool[idx_in].data, ai_stats);
        stats_ai_ms.store(ai_stats.total);
        drawDetectionsOnInput(inputPool[idx_in].data, results);
        
        auto t_remap_start = chrono::steady_clock::now();
        if (maps_ready && !map1_bufs[map_idx].data.empty()) {
            cv::remap(inputPool[idx_in].data, outputPool[idx_out].data, map1_bufs[map_idx].data, map2_bufs[map_idx].data, INTER_NEAREST, BORDER_CONSTANT);
        } else {
            cv::resize(inputPool[idx_in].data, outputPool[idx_out].data, Size(OUT_W, OUT_H));
        }
        auto t_remap_end = chrono::steady_clock::now();
        stats_remap_ms.store(chrono::duration_cast<chrono::milliseconds>(t_remap_end - t_remap_start).count());

        cv::cvtColor(outputPool[idx_out].data, outputPool[idx_out].data, COLOR_BGR2RGB);
        drawInfoBox(outputPool[idx_out].data);

        frame_count++;
        auto now = chrono::steady_clock::now();
        double elapsed_ms = chrono::duration_cast<chrono::milliseconds>(now - last_fps_time).count();
        if(elapsed_ms >= 1000.0) {
            stats_fps.store((frame_count * 1000.0) / elapsed_ms);
            frame_count = 0; last_fps_time = now;
            printf("FPS: %d\n", (int)stats_fps.load());
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
    void on_window_destroy(GtkWidget *object, gpointer user_data) { isRunning = false; gtk_main_quit(); }
    gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data) { if (event->keyval == GDK_KEY_Escape) { isRunning = false; gtk_main_quit(); return TRUE; } return FALSE; }

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

    // === MOUSE DETEKSI: 3 VIEW DI BAWAH MODE MAIN ===
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
                    // Di bawah: Split 3
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

    void on_btn_alpha_inc_clicked(GtkButton *b) { UPDATE_PTZ(alpha, 10.0f); }
    void on_btn_alpha_dec_clicked(GtkButton *b) { UPDATE_PTZ(alpha, -10.0f); }
    void on_btn_beta_inc_clicked(GtkButton *b)  { UPDATE_PTZ(beta, 10.0f); }
    void on_btn_beta_dec_clicked(GtkButton *b)  { UPDATE_PTZ(beta, -10.0f); }
    void on_btn_zoom_in_clicked(GtkButton *b)   { UPDATE_PTZ(zoom, 0.5f); }
    void on_btn_zoom_out_clicked(GtkButton *b)  { UPDATE_PTZ(zoom, -0.5f); }
}

int main(int argc, char** argv) {
    try {
        putenv((char*)"GST_DEBUG=*:0");
        moildev_app_get_resource();
        
        if(argc < 2) { cout << "./app <file> <res_id (0-4)>" << endl; return -1; }
        if (argc >= 3) SEL_RES_IDX = atoi(argv[2]);
        if (SEL_RES_IDX < 0 || SEL_RES_IDX >= BENCHMARK_RES.size()) SEL_RES_IDX = 0;
        if(SEL_RES_IDX == 3) BUFFER_SIZE = 2; 
        
        OUT_W = BENCHMARK_RES[SEL_RES_IDX].w; OUT_H = BENCHMARK_RES[SEL_RES_IDX].h;

        string src_param = argv[1];
        if (!isdigit(src_param[0])) {
            VideoCapture probe(src_param);
            if(probe.isOpened()) {
                IN_W = (int)probe.get(CAP_PROP_FRAME_WIDTH); IN_H = (int)probe.get(CAP_PROP_FRAME_HEIGHT);
                probe.release();
            } else { IN_W = DEFAULT_W; IN_H = DEFAULT_H; }
        } else { IN_W = OUT_W; IN_H = OUT_H; }

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
        
        inputPool.resize(BUFFER_SIZE); outputPool.resize(BUFFER_SIZE);
        for(int i=0; i<BUFFER_SIZE; i++) { 
            allocateDRPBuffer(inputPool[i], IN_W, IN_H, CV_8UC3); 
            allocateDRPBuffer(outputPool[i], OUT_W, OUT_H, CV_8UC3); 
        }
        for(int i=0; i<MAP_BUFFER_COUNT; i++) { 
            allocateDRPBuffer(map1_bufs[i], OUT_W, OUT_H, CV_16SC2); 
            allocateDRPBuffer(map2_bufs[i], OUT_W, OUT_H, CV_16UC1); 
        }
        
        thread t1(captureThread, argv[1]); 
        thread t2(mapUpdateThread); 
        thread t3(pipelineThread); 

        gtk_widget_show_all(main_window);
        gtk_main();
        
        isRunning = false; cv_drp.notify_all(); 
        t1.join(); t2.join(); t3.join(); 
        return 0;

    } catch (const std::exception& e) { return -1; }
}