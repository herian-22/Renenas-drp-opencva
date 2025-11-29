/**
 * DRP-AI MOILDEV GUI CONTROLLER - FINAL ROBUST VERSION
 * Menggunakan Wrapper C-Style (MoilHandle) untuk stabilitas ABI.
 */

#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <dlfcn.h>
#include <unistd.h>
#include <pthread.h>
#include <chrono>
#include <sys/stat.h>

// HEADER LOKAL
#include "dmabuf.h" 
#include "moildev_wrapper.h" // Menggunakan WRAPPER
#include "define.h"   

using namespace cv;
using namespace std;

// === KONFIGURASI ===
const int PROC_W = 640;   
const int PROC_H = 480;
int OUT_W  = 1280; 
int OUT_H  = 960;

// Parameter Fisheye
const double PARA[] = {0.0, 0.0, -35.475, 73.455, -41.392, 499.33};
const double ORG_W = 2592.0, ORG_H = 1944.0; 
const double ORG_ICX = 1205.0, ORG_ICY = 966.0;
const double SENSOR_W = 1.4; 
const double SENSOR_H = 1.4;

// === GLOBAL STATE ===
enum ViewMode { MODE_MAIN, MODE_ORIGINAL, MODE_PANORAMA, MODE_GRID, MODE_SINGLE };
atomic<ViewMode> current_mode(MODE_MAIN);
atomic<int> active_map_idx(0);
atomic<float> cur_alpha(0.0f), cur_beta(0.0f), cur_zoom(2.0f);

double last_mouse_x, last_mouse_y;
bool is_dragging = false;
atomic<bool> ui_update_pending(false); 

GtkWidget *main_window = nullptr;
GtkWidget *img_display = nullptr;

struct FrameBuffer { Mat data; dma_buffer dbuf; bool ready; };
const int BUFFER_SIZE = 3; 
vector<FrameBuffer> inputPool(BUFFER_SIZE);
vector<FrameBuffer> outputPool(BUFFER_SIZE);
const int MAP_BUFFER_COUNT = 2;
FrameBuffer map1_bufs[MAP_BUFFER_COUNT], map2_bufs[MAP_BUFFER_COUNT];

int w_idx_cap=0, r_idx_drp=0, w_idx_drp=0;
mutex mtxData; 
condition_variable cv_drp;
atomic<bool> isRunning(true);
string stats_log = "Init...";

// --- DMABUF ---
void allocateDRPBuffer(FrameBuffer &fb, int width, int height, int type) {
    size_t size = width * height * CV_ELEM_SIZE(type);
    if (buffer_alloc_dmabuf(&fb.dbuf, size) != 0) { cerr << "Alloc Fail!" << endl; exit(1); }
    if (!fb.dbuf.mem) { cerr << "Memory NULL!" << endl; exit(1); }
    fb.data = Mat(height, width, type, fb.dbuf.mem); 
    fb.ready = false;
}

// --- DRP INIT ---
void* lib_handle = nullptr;
typedef int (*OCA_Activate_Func)(unsigned long*);
OCA_Activate_Func OCA_Activate_Ptr = nullptr;

void initDRP() {
    cout << "[DRP] Init..." << endl;
    char cwd[PATH_MAX]; if (getcwd(cwd, sizeof(cwd))) setenv("DRP_EXE_PATH", cwd, 1);
    
    const char* lib_path = "/usr/lib/aarch64-linux-gnu/renesas/libopencv_imgproc.so.4.1.0";
    lib_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    
    if (lib_handle) {
        OCA_Activate_Ptr = (OCA_Activate_Func)dlsym(lib_handle, "_Z12OCA_ActivatePm");
        if (OCA_Activate_Ptr) {
            unsigned long OCA_list[17] = {0}; OCA_list[0]=1; OCA_list[16]=1;
            OCA_Activate_Ptr(OCA_list);
            cout << "[DRP] Activated." << endl;
        }
    } else {
        cout << "[DRP] Hardware accel lib not found. Using Software fallback." << endl;
    }
}

// --- MAP UPDATE (VIA WRAPPER) ---
void mapUpdateThread() {
    cout << "[MAP] Thread Start." << endl;
    
    // PENTING: Gunakan Wrapper
    MoilHandle moil = Moil_Create();
    
    // Config via Wrapper (Aman)
    Moil_Config(moil, "camera", SENSOR_W, SENSOR_H, ORG_ICX, ORG_ICY, 1.0, ORG_W, ORG_H, 1.0, PARA[0], PARA[1], PARA[2], PARA[3], PARA[4], PARA[5]);
    cout << "[MAP] Configured." << endl;

    Mat mx(OUT_H, OUT_W, CV_32F);
    Mat my(OUT_H, OUT_W, CV_32F);
    
    // Init Identity
    for(int y=0; y<OUT_H; y++) for(int x=0; x<OUT_W; x++) {
        mx.at<float>(y,x) = (float)x * PROC_W / OUT_W;
        my.at<float>(y,x) = (float)y * PROC_H / OUT_H;
    }

    ViewMode last_mode = MODE_ORIGINAL;
    float last_a=-999, last_b=-999, last_z=-999;

    while(isRunning) {
        float a = cur_alpha.load(); 
        float b = cur_beta.load(); 
        float z = cur_zoom.load();
        ViewMode mode = current_mode.load();

        if (mode != last_mode || abs(a-last_a)>0.1 || abs(b-last_b)>0.1 || abs(z-last_z)>0.1) {
            
            // Generate Maps berdasarkan Mode via Wrapper
            if (mode == MODE_PANORAMA || mode == MODE_MAIN) {
                 Moil_MapsPanorama(moil, (float*)mx.data, (float*)my.data, 110, 0, 0);
            } 
            else if (mode == MODE_SINGLE || mode == MODE_GRID) {
                 Moil_AnyPoint(moil, (float*)mx.data, (float*)my.data, a, b, z);
            }
            else { // Original / Default
                for(int y=0; y<OUT_H; y++) for(int x=0; x<OUT_W; x++) {
                    mx.at<float>(y,x) = (float)x * PROC_W / OUT_W;
                    my.at<float>(y,x) = (float)y * PROC_H / OUT_H;
                }
            }

            int back_idx = (active_map_idx.load() + 1) % MAP_BUFFER_COUNT;
            convertMaps(mx, my, map1_bufs[back_idx].data, map2_bufs[back_idx].data, CV_16SC2, false);
            
            buffer_flush_dmabuf(map1_bufs[back_idx].dbuf.idx, map1_bufs[back_idx].dbuf.size);
            buffer_flush_dmabuf(map2_bufs[back_idx].dbuf.idx, map2_bufs[back_idx].dbuf.size);
            
            active_map_idx.store(back_idx);
            last_a=a; last_b=b; last_z=z; last_mode=mode;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    Moil_Destroy(moil);
}

// --- CAPTURE ---
void captureThread(string src) {
    cout << "[CAP] Connecting to: " << src << endl;
    string pipe = (src.length()==1 && isdigit(src[0])) ? 
        "v4l2src device=/dev/video"+src+" ! image/jpeg, width=640, height=480, framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false" :
        "filesrc location="+src+" ! decodebin ! videoconvert ! video/x-raw, format=BGR ! videoscale ! video/x-raw, width=640, height=480 ! appsink drop=true sync=false";
    
    VideoCapture cap(pipe, CAP_GSTREAMER);
    if(!cap.isOpened()) { cerr << "[CAP] Failed to open pipeline." << endl; isRunning=false; return; }
    
    Mat temp;
    while(isRunning) {
        if(!cap.read(temp) || temp.empty()) {
             if(src.length()>1) cap.set(CAP_PROP_POS_FRAMES, 0);
             else this_thread::sleep_for(chrono::milliseconds(100));
             continue;
        }
        int idx = w_idx_cap;
        temp.copyTo(inputPool[idx].data);
        buffer_flush_dmabuf(inputPool[idx].dbuf.idx, inputPool[idx].dbuf.size);
        { lock_guard<mutex> lock(mtxData); inputPool[idx].ready=true; w_idx_cap=(w_idx_cap+1)%BUFFER_SIZE; r_idx_drp=idx; }
        cv_drp.notify_one();
        if(src.length()>1) this_thread::sleep_for(chrono::milliseconds(33));
    }
}

// --- GUI ---
gboolean update_gui_image(gpointer user_data) {
    if(!isRunning || !img_display) return FALSE;
    int idx = (int)(intptr_t)user_data;
    if(idx < 0 || idx >= BUFFER_SIZE) { ui_update_pending = false; return FALSE; }
    
    Mat &src = outputPool[idx].data;
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(src.data, GDK_COLORSPACE_RGB, FALSE, 8, src.cols, src.rows, src.step, NULL, NULL);
    if (pixbuf) { gtk_image_set_from_pixbuf(GTK_IMAGE(img_display), pixbuf); g_object_unref(pixbuf); }
    outputPool[idx].ready = false; ui_update_pending = false;
    return FALSE;
}

// --- DRP PROCESS ---
void drpThread() {
    setThreadPriority(99); 

    int frames = 0; 
    auto start = chrono::steady_clock::now();
    while(isRunning) {
        int idx_in = -1;
        { unique_lock<mutex> lock(mtxData); cv_drp.wait(lock, []{ return inputPool[r_idx_drp].ready || !isRunning; }); if(!isRunning) break; idx_in = r_idx_drp; inputPool[idx_in].ready = false; }
        
        int idx_out = w_idx_drp;
        int map_idx = active_map_idx.load();
        
        // Remap (INTER_LINEAR untuk kualitas, akselerasi DRP)
        cv::remap(inputPool[idx_in].data, outputPool[idx_out].data, map1_bufs[map_idx].data, map2_bufs[map_idx].data, INTER_LINEAR, BORDER_CONSTANT);
        cv::cvtColor(outputPool[idx_out].data, outputPool[idx_out].data, COLOR_BGR2RGB);

        frames++;
        if(frames % 30 == 0) {
            auto now = chrono::steady_clock::now();
            double fps = 30000.0 / (chrono::duration_cast<chrono::milliseconds>(now - start).count() + 1.0);
            start = now; stats_log = "FPS: " + to_string((int)fps);
        }
        putText(outputPool[idx_out].data, stats_log, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        buffer_flush_dmabuf(outputPool[idx_out].dbuf.idx, outputPool[idx_out].dbuf.size);
        
        { lock_guard<mutex> lock(mtxData); outputPool[idx_out].ready = true; w_idx_drp = (w_idx_drp + 1) % BUFFER_SIZE; 
        if (!ui_update_pending.load()) { ui_update_pending = true; g_idle_add(update_gui_image, (gpointer)(intptr_t)idx_out); }}
    }
}

extern "C" {
    void on_window_destroy(GtkWidget *o, gpointer u) { isRunning = false; gtk_main_quit(); }
    gboolean on_key_press(GtkWidget *w, GdkEventKey *e, gpointer u) { if (e->keyval == GDK_KEY_Escape) { isRunning = false; gtk_main_quit(); return TRUE; } return FALSE; }
    void on_btn_main_clicked(GtkButton *b) { current_mode = MODE_MAIN; }
    void on_btn_orig_clicked(GtkButton *b) { current_mode = MODE_ORIGINAL; }
    void on_btn_pano_clicked(GtkButton *b) { current_mode = MODE_PANORAMA; }
    void on_btn_grid_clicked(GtkButton *b) { current_mode = MODE_GRID; }
    void on_btn_ptz_clicked(GtkButton *b)  { current_mode = MODE_SINGLE; }
    
    gboolean on_mouse_down(GtkWidget *w, GdkEventButton *e, gpointer u) { 
        if (e->button == 1) { is_dragging = true; last_mouse_x = e->x; last_mouse_y = e->y; } 
        if (e->button == 3) { cur_zoom.store(2.0f); }
        return TRUE; 
    }
    gboolean on_mouse_up(GtkWidget *w, GdkEventButton *e, gpointer u) { is_dragging = false; return TRUE; }
    gboolean on_mouse_move(GtkWidget *w, GdkEventMotion *e, gpointer u) { 
        if (is_dragging && current_mode == MODE_SINGLE) { 
            float dx = (e->x - last_mouse_x) * 0.1f; float dy = (e->y - last_mouse_y) * 0.1f;
            float a = cur_alpha.load(); cur_alpha.store(a + dx);
            float b = cur_beta.load(); cur_beta.store(b + dy);
            last_mouse_x = e->x; last_mouse_y = e->y; 
        } 
        return TRUE; 
    }
}
void setThreadPriority(int priority) {
    sched_param sch_params;
    sch_params.sched_priority = priority;
    if(pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch_params)) {
        cerr << "[WARN] Failed to set Thread Priority. Run with sudo?" << endl;
    } else {
        cout << "[SYS] Thread Priority set to Real-Time: " << priority << endl;
    }
}

int main(int argc, char** argv) {
    if(argc < 2) { cout << "Usage: ./app <video>" << endl; return -1; }
    gtk_init(&argc, &argv);
    GtkBuilder *builder = gtk_builder_new();
    if (!gtk_builder_add_from_file(builder, "gui.glade", NULL)) return 1;
    main_window = GTK_WIDGET(gtk_builder_get_object(builder, "main_window"));
    img_display = GTK_WIDGET(gtk_builder_get_object(builder, "img_display"));
    gtk_window_fullscreen(GTK_WINDOW(main_window));
    gtk_builder_connect_signals(builder, NULL);
    g_signal_connect(main_window, "key-press-event", G_CALLBACK(on_key_press), NULL);
    g_object_unref(builder);

    for(int i=0; i<BUFFER_SIZE; i++) { allocateDRPBuffer(inputPool[i], PROC_W, PROC_H, CV_8UC3); allocateDRPBuffer(outputPool[i], OUT_W, OUT_H, CV_8UC3); }
    for(int i=0; i<MAP_BUFFER_COUNT; i++) { allocateDRPBuffer(map1_bufs[i], OUT_W, OUT_H, CV_16SC2); allocateDRPBuffer(map2_bufs[i], OUT_W, OUT_H, CV_16UC1); }

    unsetenv("LD_LIBRARY_PATH"); initDRP();
    thread t1(captureThread, argv[1]); thread t2(mapUpdateThread); thread t3(drpThread);
    gtk_widget_show_all(main_window);
    gtk_main();

    isRunning = false; cv_drp.notify_all(); t1.join(); t2.join(); t3.join();
    for(int i=0; i<BUFFER_SIZE; i++) { buffer_free_dmabuf(&inputPool[i].dbuf); buffer_free_dmabuf(&outputPool[i].dbuf); }
    return 0;
}
