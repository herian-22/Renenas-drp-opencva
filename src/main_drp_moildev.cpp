/**
 * DRP GUI CONTROLLER - EMBEDDED RESOURCE VERSION
 * Revised for Stability: 1080p Resolution & Resource Loading
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
#include "dmabuf.h" 

// --- [PENTING] Header Resource Hasil Generate CMake ---
extern "C" {
#include "moildev_resources.h" 
}
using namespace cv;
using namespace std;

// === KONFIGURASI RESOLUSI (SAFE SPOT: 1080p) ===
// Kita gunakan 1080p agar CPU lancar men-decode MJPEG, dan DRP ringan meremap.
const int PROC_W = 1280;   
const int PROC_H = 960;
int OUT_W  = 1280; 
int OUT_H  = 960;

// Fisheye Params (Sesuaikan jika perlu)
const double PARA[] = {0.0, 0.0, -35.475, 73.455, -41.392, 499.33};
const double ORG_W = 2592.0, ORG_H = 1944.0; 
const double ORG_ICX = 1205.0, ORG_ICY = 966.0;

// === GLOBAL STATE ===
enum ViewMode { MODE_MAIN, MODE_ORIGINAL, MODE_PANORAMA, MODE_GRID, MODE_SINGLE };
atomic<ViewMode> current_mode(MODE_MAIN);
atomic<int> active_map_idx(0);
atomic<float> cur_alpha(0.0f), cur_beta(0.0f), cur_zoom(2.0f);

// Mouse State
double last_mouse_x, last_mouse_y;
bool is_dragging = false;

// GUI State
atomic<bool> ui_update_pending(false); 

// GTK Widgets
GtkWidget *main_window;
GtkWidget *img_display;

// Buffers (Menggunakan 3 Buffer agar smooth)
struct FrameBuffer { Mat data; dma_buffer dbuf; bool ready; };
const int BUFFER_SIZE = 3; 
vector<FrameBuffer> inputPool(BUFFER_SIZE);
vector<FrameBuffer> outputPool(BUFFER_SIZE);
const int MAP_BUFFER_COUNT = 2;
FrameBuffer map1_bufs[MAP_BUFFER_COUNT], map2_bufs[MAP_BUFFER_COUNT];

int w_idx_cap=0, r_idx_drp=0, w_idx_drp=0, r_idx_dsp=0;
mutex mtxData; condition_variable cv_drp;
atomic<bool> isRunning(true);
string stats_log = "Init...";

// --- DMABUF HELPER ---
void allocateDRPBuffer(FrameBuffer &fb, int width, int height, int type) {
    size_t size = width * height * CV_ELEM_SIZE(type);
    if (buffer_alloc_dmabuf(&fb.dbuf, size) != 0) { 
        cerr << "[DMABUF] ERROR: Alloc Fail! (Check CMA Size)" << endl; 
        exit(1); 
    }
    fb.data = Mat(height, width, type, fb.dbuf.mem); 
    fb.ready = false;
    printf("[DMABUF] Allocated: %dx%d | IDX: %d\n", width, height, fb.dbuf.idx);
}

// --- DRP INIT ---
void* lib_handle = nullptr;
typedef int (*OCA_Activate_Func)(unsigned long*);
OCA_Activate_Func OCA_Activate_Ptr = nullptr;

void initDRP() {
    cout << "[DRP] Init..." << endl;
    char cwd[PATH_MAX]; getcwd(cwd, sizeof(cwd)); setenv("DRP_EXE_PATH", cwd, 1);
    lib_handle = dlopen("/usr/lib/aarch64-linux-gnu/renesas/libopencv_imgproc.so.4.1.0", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!lib_handle) { cerr << "Lib Fail" << endl; exit(1); }
    OCA_Activate_Ptr = (OCA_Activate_Func)dlsym(lib_handle, "_Z12OCA_ActivatePm");
    unsigned long OCA_list[17] = {0}; OCA_list[0]=1; OCA_list[16]=1;
    OCA_Activate_Ptr(OCA_list);
}

// --- MOIL MATH ---
class MoilNative {
public:
    double iCx, iCy, scale_x, scale_y;
    MoilNative() {
        scale_x = (double)PROC_W / ORG_W; scale_y = (double)PROC_H / ORG_H;
        iCx = ORG_ICX * scale_x; iCy = ORG_ICY * scale_y;
    }
    double getRho(double alpha) {
        double p0=PARA[0], p1=PARA[1], p2=PARA[2], p3=PARA[3], p4=PARA[4], p5=PARA[5];
        return (((((p0*alpha+p1)*alpha+p2)*alpha+p3)*alpha+p4)*alpha+p5)*alpha;
    }
    void fillRegion(Mat &mx, Mat &my, Rect r, float alpha, float beta, float zoom, bool isPano) {
        const double PI = 3.14159265358979323846;
        #pragma omp parallel for collapse(2)
        for(int y=0; y<r.height; y++) {
            for(int x=0; x<r.width; x++) {
                double rho, theta;
                if(isPano) {
                    double nx=(double)x/r.width, ny=(double)y/r.height;
                    theta=(nx-0.5)*2.0*PI; rho=getRho(ny*(alpha*PI/180.0));
                } else {
                    double f=r.width/(2.0*tan(30.0*PI/180.0/zoom));
                    double ar=alpha*PI/180.0, br=beta*PI/180.0;
                    double cx=x-r.width/2.0, cy=y-r.height/2.0;
                    double x1=cx, y1=cy*cos(ar)-f*sin(ar), z1=cy*sin(ar)+f*cos(ar);
                    double x2=x1*cos(br)-y1*sin(br), y2=x1*sin(br)+y1*cos(br), z2=z1;
                    rho=getRho(atan2(sqrt(x2*x2+y2*y2),z2)); theta=atan2(y2,x2);
                }
                int py=r.y+y, px=r.x+x;
                if(px>=0 && px<mx.cols && py>=0 && py<mx.rows) {
                    mx.at<float>(py,px)=(float)(iCx+(rho*scale_x)*cos(theta));
                    my.at<float>(py,px)=(float)(iCy+(rho*scale_y)*sin(theta));
                }
            }
        }
    }
    void fillStandard(Mat &mx, Mat &my, Rect r) {
        float sx = (float)PROC_W / r.width; float sy = (float)PROC_H / r.height;
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

// --- THREAD MAP UPDATE ---
void mapUpdateThread() {
    MoilNative moil;
    Mat mx(OUT_H, OUT_W, CV_32F);
    Mat my(OUT_H, OUT_W, CV_32F);
    float last_a=-999, last_b=-999, last_z=-999; ViewMode last_mode = MODE_SINGLE; 

    while(isRunning) {
        float a = cur_alpha.load(); float b = cur_beta.load(); float z = cur_zoom.load();
        ViewMode mode = current_mode.load();
        if (mode == last_mode && a == last_a && b == last_b && z == last_z) {
            this_thread::sleep_for(chrono::milliseconds(15)); continue;
        }
        int back_idx = (active_map_idx.load() + 1) % MAP_BUFFER_COUNT;
        mx.setTo(Scalar(-1)); my.setTo(Scalar(-1));

        switch(mode) {
            case MODE_MAIN: { 
                int split = OUT_H / 2;
                moil.fillRegion(mx, my, Rect(0, 0, OUT_W, split), 110, 0, 0, true);
                int bw = OUT_W/4, bh = OUT_H - split;
                moil.fillRegion(mx, my, Rect(0, split, bw, bh), 0, 0, 2.0, false);
                moil.fillRegion(mx, my, Rect(bw, split, bw, bh), 45, 0, 2.0, false);
                moil.fillRegion(mx, my, Rect(bw*2, split, bw, bh), 45, -90, 2.0, false);
                moil.fillRegion(mx, my, Rect(bw*3, split, bw, bh), 45, 90, 2.0, false);
                break; 
            }
            case MODE_ORIGINAL: moil.fillStandard(mx, my, Rect(0, 0, OUT_W, OUT_H)); break;
            case MODE_PANORAMA: moil.fillRegion(mx, my, Rect(0, 0, OUT_W, OUT_H), 110, 0, 0, true); break;
            case MODE_GRID: {
                int bw = OUT_W/2, bh = OUT_H/2;
                moil.fillRegion(mx, my, Rect(0, 0, bw, bh), 0, 0, 2.0, false);
                moil.fillRegion(mx, my, Rect(bw, 0, bw, bh), 45, 0, 2.0, false);
                moil.fillRegion(mx, my, Rect(0, bh, bw, bh), 45, -90, 2.0, false);
                moil.fillRegion(mx, my, Rect(bw, bh, bw, bh), 45, 90, 2.0, false);
                break;
            }
            case MODE_SINGLE: moil.fillRegion(mx, my, Rect(0, 0, OUT_W, OUT_H), a, b, z, false); break;
        }

        convertMaps(mx, my, map1_bufs[back_idx].data, map2_bufs[back_idx].data, CV_16SC2, false);
        buffer_flush_dmabuf(map1_bufs[back_idx].dbuf.idx, map1_bufs[back_idx].dbuf.size);
        buffer_flush_dmabuf(map2_bufs[back_idx].dbuf.idx, map2_bufs[back_idx].dbuf.size);
        active_map_idx.store(back_idx);
        last_a=a; last_b=b; last_z=z; last_mode=mode;
    }
}

// --- THREAD CAPTURE ---
void captureThread(string src) {
    string pipe;
    bool is_camera = isdigit(src[0]);

    // PIPELINE OPTIMIZATION 1080p
    // Menggunakan jpegdec (Software MJPEG) karena v4l2jpegdec (Hardware) tidak tersedia
    if(is_camera) {
        pipe = "v4l2src device=/dev/video" + src + " ! image/jpeg, width=1280, height=960, framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false";
    } else {
        // pipe = "filesrc location=" + src + " ! decodebin ! videoconvert ! video/x-raw, format=BGR ! videoscale ! video/x-raw, width=1280, height=960 ! appsink drop=true sync=false";
        pipe = "filesrc location=" + src +
                " ! decodebin"
                " ! videorate"
                " ! videoconvert"
                " ! video/x-raw,format=BGR,framerate=30/1"
                " ! videoscale"
                " ! video/x-raw,width=1280,height=960"
                " ! appsink max-buffers=1 drop=true sync=false";

    }

    VideoCapture *cap = new VideoCapture(pipe, CAP_GSTREAMER);

    if (!cap->isOpened() && is_camera) {
        cerr << "[CAPTURE] MJPG Failed, trying RAW..." << endl;
        delete cap;
        // Fallback pipeline (slow but works)
        pipe = "v4l2src device=/dev/video" + src + " ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false";
        cap = new VideoCapture(pipe, CAP_GSTREAMER);
    }

    if (!cap->isOpened()) {
        cerr << "[CAPTURE] FATAL: Cannot open source." << endl;
        isRunning = false; return;
    }

    cout << "[CAPTURE] Pipeline OK: " << src << endl;

    Mat temp;
    auto next_frame = chrono::steady_clock::now();
    const auto frame_duration = chrono::milliseconds(33); // Target 30 FPS

    while(isRunning) {
        next_frame += frame_duration;
        
        if (!cap->read(temp) || temp.empty()) {
             if(is_camera) {
                 // Auto-reconnect camera
                 delete cap; this_thread::sleep_for(chrono::milliseconds(500)); 
                 cap = new VideoCapture(pipe, CAP_GSTREAMER);
             }
             continue;
        }

        int idx = w_idx_cap;
        // Resize jika input fallback (640x480) tidak sama dengan PROC_W (1920)
        if (temp.cols != PROC_W || temp.rows != PROC_H) {
            cv::resize(temp, inputPool[idx].data, Size(PROC_W, PROC_H));
        } else {
            temp.copyTo(inputPool[idx].data);
        }

        buffer_flush_dmabuf(inputPool[idx].dbuf.idx, inputPool[idx].dbuf.size);
        
        { lock_guard<mutex> lock(mtxData); inputPool[idx].ready = true; w_idx_cap = (w_idx_cap + 1) % BUFFER_SIZE; r_idx_drp = idx; }
        cv_drp.notify_one();

        if (!is_camera) this_thread::sleep_until(next_frame);
    }
    if (cap) delete cap;
}

// --- GTK CALLBACKS ---
extern "C" {
    void on_window_destroy(GtkWidget *object, gpointer user_data) { isRunning = false; gtk_main_quit(); }
    gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data) { if (event->keyval == GDK_KEY_Escape) { isRunning = false; gtk_main_quit(); return TRUE; } return FALSE; }
    void on_btn_main_clicked(GtkButton *b) { current_mode = MODE_MAIN; }
    void on_btn_orig_clicked(GtkButton *b) { current_mode = MODE_ORIGINAL; }
    void on_btn_pano_clicked(GtkButton *b) { current_mode = MODE_PANORAMA; }
    void on_btn_grid_clicked(GtkButton *b) { current_mode = MODE_GRID; }
    void on_btn_ptz_clicked(GtkButton *b)  { current_mode = MODE_SINGLE; }
    gboolean on_mouse_down(GtkWidget *widget, GdkEventButton *event, gpointer user_data) { if (event->button == 1 && current_mode == MODE_SINGLE) { is_dragging = true; last_mouse_x = event->x; last_mouse_y = event->y; } else if (event->button == 3) { cur_zoom = 1.0f; } return TRUE; }
    gboolean on_mouse_up(GtkWidget *widget, GdkEventButton *event, gpointer user_data) { is_dragging = false; return TRUE; }
    gboolean on_mouse_move(GtkWidget *widget, GdkEventMotion *event, gpointer user_data) { if (is_dragging && current_mode == MODE_SINGLE) { float dx = (event->x - last_mouse_x) * 0.2f; float dy = (event->y - last_mouse_y) * 0.2f; cur_alpha = cur_alpha + dx; cur_beta = cur_beta + dy; last_mouse_x = event->x; last_mouse_y = event->y; } return TRUE; }
}

gboolean update_gui_image(gpointer user_data) {
    int idx = (int)(intptr_t)user_data;
    if(idx < 0 || idx >= BUFFER_SIZE) { ui_update_pending = false; return FALSE; }
    Mat &src = outputPool[idx].data;
    // Konversi Mat ke Pixbuf untuk ditampilkan di GTK
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(src.data, GDK_COLORSPACE_RGB, FALSE, 8, src.cols, src.rows, src.step, NULL, NULL);
    if (pixbuf) { 
        // Resize agar pas di layar laptop jika outputnya besar
        GdkPixbuf *scaled = gdk_pixbuf_scale_simple(pixbuf, 1280, 720, GDK_INTERP_BILINEAR);
        gtk_image_set_from_pixbuf(GTK_IMAGE(img_display), scaled); 
        g_object_unref(scaled);
        g_object_unref(pixbuf); 
    }
    outputPool[idx].ready = false; ui_update_pending = false; 
    return FALSE;
}

// --- THREAD DRP (PROCESSING) ---
void drpThread() {
    int frames = 0; 
    auto start = chrono::steady_clock::now();
    cout << "[DRP] Waiting..." << endl;

    while(isRunning) {
        int idx_in = -1;
        { unique_lock<mutex> lock(mtxData); cv_drp.wait(lock, []{ return inputPool[r_idx_drp].ready || !isRunning; }); if(!isRunning) break; idx_in = r_idx_drp; inputPool[idx_in].ready = false; }
        
        int idx_out = w_idx_drp;
        int map_idx = active_map_idx.load();
        
        int interp = INTER_NEAREST; // Tercepat
        
        auto t1 = chrono::steady_clock::now();
        // Proses Remap di Hardware DRP
        cv::remap(inputPool[idx_in].data, outputPool[idx_out].data, map1_bufs[map_idx].data, map2_bufs[map_idx].data, interp, BORDER_CONSTANT);
        auto t2 = chrono::steady_clock::now();
        double remap_ms = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

        cv::cvtColor(outputPool[idx_out].data, outputPool[idx_out].data, COLOR_BGR2RGB);

        frames++;
        if(frames % 30 == 0) {
            auto now = chrono::steady_clock::now();
            double total_ms = chrono::duration_cast<chrono::milliseconds>(now - start).count();
            double fps = 30000.0 / (total_ms + 1.0); start = now;
            stats_log = "FPS: " + to_string((int)fps);
            printf("[DIAG] FPS: %d | Remap Time: %.1f ms | Res: %dx%d\n", (int)fps, remap_ms, OUT_W, OUT_H);
        }
        
        putText(outputPool[idx_out].data, stats_log, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        buffer_flush_dmabuf(outputPool[idx_out].dbuf.idx, outputPool[idx_out].dbuf.size);
        
        { lock_guard<mutex> lock(mtxData); outputPool[idx_out].ready = true; w_idx_drp = (w_idx_drp + 1) % BUFFER_SIZE; 
        if (!ui_update_pending.load()) { ui_update_pending = true; g_idle_add(update_gui_image, (gpointer)(intptr_t)idx_out); }}
    }
}

// --- MAIN FUNCTION (RESOURCE FIXED) ---
int main(int argc, char** argv) {
    putenv((char*)"GST_DEBUG=1");
    
    // [PENTING] Init Resource agar tidak dibuang oleh linker
    moildev_app_get_resource();

    if(argc < 2) { cout << "./app <video_id>" << endl; return -1; }
    
    gtk_init(&argc, &argv);
    GtkBuilder *builder = gtk_builder_new();
    GError *error = NULL;

    // [PENTING] Memuat UI dari Resource Memory (bukan file)
    if (!gtk_builder_add_from_resource(builder, "/moildev/app/gui.glade", &error)) {
        g_printerr("Error loading UI resource: %s\n", error->message);
        g_clear_error(&error);
        return 1;
    }

    main_window = GTK_WIDGET(gtk_builder_get_object(builder, "main_window"));
    img_display = GTK_WIDGET(gtk_builder_get_object(builder, "img_display"));
    gtk_window_fullscreen(GTK_WINDOW(main_window));
    gtk_builder_connect_signals(builder, NULL);
    g_signal_connect(main_window, "key-press-event", G_CALLBACK(on_key_press), NULL);
    g_object_unref(builder);

    unsetenv("LD_LIBRARY_PATH"); initDRP();
    
    // Alokasi Buffer Memory
    for(int i=0; i<BUFFER_SIZE; i++) { allocateDRPBuffer(inputPool[i], PROC_W, PROC_H, CV_8UC3); allocateDRPBuffer(outputPool[i], OUT_W, OUT_H, CV_8UC3); }
    for(int i=0; i<MAP_BUFFER_COUNT; i++) { allocateDRPBuffer(map1_bufs[i], OUT_W, OUT_H, CV_16SC2); allocateDRPBuffer(map2_bufs[i], OUT_W, OUT_H, CV_16UC1); }
    
    thread t1(captureThread, argv[1]); 
    thread t2(mapUpdateThread); 
    thread t3(drpThread);
    
    gtk_widget_show_all(main_window);
    gtk_main();
    
    isRunning = false; cv_drp.notify_all(); t1.join(); t2.join(); t3.join();
    return 0;
}