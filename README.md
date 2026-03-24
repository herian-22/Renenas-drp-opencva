# Renenas DRP-OAI + Moildev (OpenCVA)

Dokumentasi ini merangkum seluruh fungsi penting, rumus yang dipakai, serta cara build dan menjalankan aplikasi pendeteksi objek berbasis DRP-AI YOLO dengan dukungan koreksi fisheye Moildev dan akselerasi OpenCV Accelerator (OCA) di SoC Renesas RZ/V2H.

## Ringkasan Fitur
- **Inferensi YOLOv8 on-device** menggunakan DRP-AI TVM runtime (float16/float32 auto-handling).
- **Koreksi fisheye & PTZ virtual** memakai `Moildev` (AnyPoint & Panorama) dengan akselerasi DRP-OCA untuk `cvtColor`/`remap`.
- **Manajemen DMA buffer** untuk alur kamera → AI → tampilan GTK.
- **Statistik performa per frame** (pre, inferensi, post, total) yang dikembalikan lewat `AiStats`.
- **Non-Maximum Suppression (NMS)** lengkap dengan rumus IoU dan penanganan kelas.

## Prasyarat Sistem
| Komponen | Keterangan |
| --- | --- |
| Hardware | Renesas RZ/V2H (atau keluarga RZ/V yang mendukung DRP-AI) |
| SDK & Toolchain | Poky SDK dengan OpenCV 4.1, GTK3, dan DRP-OCA (libopencv_imgproc.so.4.1.0) |
| TVM Runtime | Variabel `TVM_HOME` harus menunjuk ke root build runtime TVM. |
| Model & Label | Folder model (default `unicorn/`) berisi `.so`, `.json`, `.params`; file label teks (default `unicorn.txt`). |

> Pastikan `SDK`, `SDKTARGETSYSROOT`, dan `TVM_HOME` sudah diekspor sebelum build.

## Struktur Direktori
```
include/      Header (box, drpai_yolo, moildev_wrapper, define, dll.)
src/          Implementasi utama (YOLO, Moil, DMA buffer, main GTK)
asset/        Resource GUI (di-build via glib-compile-resources)
lib/          Library statis moildev (`libmoildevren.a`)
build.sh      Skrip singkat cmake && make
moildev_resources.xml  Definisi resource GTK
```

## Cara Build Cepat
```bash
export TVM_HOME=/path/to/tvm
export SDK=/path/to/sdk              # diperlukan agar deteksi V2H/V2N berjalan
export SDKTARGETSYSROOT=/path/to/sysroot

cd /home/runner/work/Renenas-drp-opencva/Renenas-drp-opencva
./build.sh
```
Hasil binaan: `build/moildev_app+DrpAiYolo`.

## Menjalankan Aplikasi
1. Salin folder model (`unicorn/`) dan `unicorn.txt` ke direktori kerja berdampingan dengan biner.
2. Pastikan kamera tersedia di `INPUT_CAM_NAME` (default `/dev/video0`).
3. Jalankan:
   ```bash
   ./build/moildev_app+DrpAiYolo
   ```
4. Tampilan GTK memuat mode panorama / PTZ / grid; inferensi berjalan otomatis di thread AI.

## Parameter & Konfigurasi Penting (`include/define.h`)
| Nama | Nilai Default | Fungsi |
| --- | --- | --- |
| `model_dir` | `"unicorn"` | Folder model DRP-AI |
| `label_list` | `"unicorn.txt"` | File label satu-per-baris |
| `NUM_CLASS` | 5 | Jumlah kelas deteksi |
| `MODEL_IN_W`, `MODEL_IN_H` | 640 x 640 | Ukuran input YOLO (letterbox) |
| `TH_PROB` | 0.25 | Ambang probabilitas sebelum NMS |
| `TH_NMS` | 0.45 | Ambang IoU untuk NMS |
| `DRPAI_FREQ` | 2 | Frekuensi run DRP-AI (Hz) |
| `grid_sizes` | 80 / 40 / 20 | Grid YOLOv8 (stride 8/16/32) |
| `NUM_COORD_BINS` | 16 | Bin Distance Focal Loss (DFL) untuk tiap sisi bbox |

## Alur AI YOLOv8 (DRP-AI)
1. **Pre-proses (letterbox)** – `DrpAiYolo::pre_process`  
   - Resize proporsional, pad dengan warna 114/255.  
   - Pisah kanal menjadi NCHW (R,G,B) float32.
2. **Inferensi** – `runtime.SetInput` → `runtime.Run(DRPAI_FREQ)` menggunakan TVM runtime.
3. **Post-proses** – `get_result_tensors` + `post_process`  
   - Dekode DFL, sigmoid skor kelas, skala kembali ke citra asli, lalu `filter_boxes_nms`.
4. **Statistik** – `AiStats` mengembalikan waktu (ms) pre/inf/post/total dan jumlah deteksi.

## Rumus Bounding Box & Post-Proses
| Fungsi | Rumus / Logika | Berkas |
| --- | --- | --- |
| `overlap(x1,w1,x2,w2)` | `left = max(x1 - w1/2, x2 - w2/2)`; `right = min(x1 + w1/2, x2 + w2/2)`; `return right - left` | `src/box.cpp` |
| `box_intersection(a,b)` | `w = overlap(ax,aw,bx,bw)`; `h = overlap(ay,ah,by,bh)`; `area = (w<0 || h<0) ? 0 : w*h` | `src/box.cpp` |
| `box_union(a,b)` | `u = a.w*a.h + b.w*b.h - box_intersection(a,b)` | `src/box.cpp` |
| `box_iou(a,b)` | `IoU = box_intersection(a,b) / box_union(a,b)` | `src/box.cpp` |
| `filter_boxes_nms(det,th)` | Supresi kotak dengan kelas sama bila `IoU > th` **atau** tumpang tindih hampir penuh. Kotak prob lebih kecil diset `prob=0`. | `src/box.cpp` |
| `sigmoid(x)` | `1 / (1 + exp(-x))` diterapkan pada logit kelas | `src/drpai_yolo.cpp` |
| `DFL decode` | `softmax(bin_k) = exp(bin_k - max) / Σ exp(bin_i - max)`; `offset = Σ softmax[i] * i` untuk tiap sisi (left,top,right,bottom) | `DrpAiYolo::dfl_decode` |
| Konversi bbox ke piksel | `cx = (x+0.5 - dfl_left)*stride`, `cy = (y+0.5 - dfl_top)*stride`, `cx2 = (x+0.5 + dfl_right)*stride`, `cy2 = (y+0.5 + dfl_bottom)*stride` → skala balik: `(cx - pad_x)/scale` | `DrpAiYolo::post_process` |

## Tabel Fungsi Utama
### Deteksi YOLO (`src/drpai_yolo.cpp`)
| Fungsi | Deskripsi Ringkas |
| --- | --- |
| `DrpAiYolo::init(model_dir, start_addr)` | Memuat model DRP-AI TVM; menampilkan ambang `TH_PROB` & `TH_NMS`. |
| `run_detection(mat, AiStats&)` | Jalankan pre → inferensi → post, mengembalikan `std::vector<detection>` & statistik waktu. |
| `pre_process(img)` | Letterbox 640x640, normalisasi 0–1, susun ke buffer NCHW. |
| `get_result_tensors()` | Ambil output multi-layer dari runtime, konversi float16→float32, petakan ke tensor box/kelas. |
| `post_process(img_w, img_h)` | Dekode DFL, sigmoid kelas, skala ke ukuran asli, terapkan NMS. |
| `dfl_decode(tensor)` | Softmax 16 bin untuk tiap sisi bbox dan kembalikan offset sub-pixel. |
| `float16_to_float32(a)` | Konversi manual bit-level untuk output FP16. |

### Bounding Box Utility (`src/box.cpp`)
| Fungsi | Kegunaan |
| --- | --- |
| `overlap` | Hitung panjang tumpang tindih 1D. |
| `box_intersection` | Luas irisan dua kotak. |
| `box_union` | Luas gabungan dua kotak. |
| `box_iou` | Intersection over Union (IoU). |
| `filter_boxes_nms` | Non-Maximum Suppression kelas-sadar. |

### Moildev Wrapper & Math
| Fungsi | Berkas | Keterangan |
| --- | --- | --- |
| `Moil_Create / Destroy` | `src/moildev_wrapper.cpp` | Alokasi/cleanup objek `Moildev`. |
| `Moil_Config` | `src/moildev_wrapper.cpp` | Set parameter sensor & polinomial lensa. |
| `Moil_MapsPanorama` | `src/moildev_wrapper.cpp` | Buat peta panorama untuk remap. |
| `Moil_AnyPoint` | `src/moildev_wrapper.cpp` | Peta PTZ virtual (AnyPoint). |
| `MoilNative::fillRegion` | `src/main_drp_moildev.cpp` | Hitung koordinat remap (AnyPoint/Panorama) memakai polinomial `getRho` dan rotasi Euler sederhana. |
| `MoilNative::getRho` | `src/main_drp_moildev.cpp` | Evaluasi polinomial orde-5: `((((p0*α+p1)α+p2)α+p3)α+p4)α+p5)*α`. |

### DMA & Tampilan (ringkas)
- `allocateDRPBuffer` / `freeDRPBuffer` — alokasi DMA buffer untuk pool input/proses/output.
- Thread utama: **capture** (kamera → YUYV), **AI** (BGR → YOLO), **mapUpdate** (remap matrix), **display** (GTK).

## Catatan Pembaruan Terakhir
- Dokumentasi rumus IoU/NMS dan DFL diperjelas.
- Tabel fungsi untuk modul AI, bounding box, dan Moildev disediakan lengkap.
- README diperluas dengan langkah build, run, serta parameter `define.h`.

## Lisensi
Hak cipta mengikuti sumber asli Renesas (lihat header tiap berkas). README ini hanya dokumentasi tambahan.
