#ifndef MOIL_WRAPPER_H
#define MOIL_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    // Handle pointer generik untuk menyembunyikan class Moildev asli
    typedef void* MoilHandle;

    // Fungsi untuk membuat dan menghapus objek
    MoilHandle Moil_Create();
    void Moil_Destroy(MoilHandle h);

    // Wrapper untuk Config
    void Moil_Config(MoilHandle h, const char* name, 
        double sensor_w, double sensor_h, 
        double iCx, double iCy, double ratio, 
        double img_w, double img_h, double calib_ratio, 
        double p0, double p1, double p2, double p3, double p4, double p5);

    // Wrapper untuk Maps Panorama
    void Moil_MapsPanorama(MoilHandle h, float* mapX, float* mapY, double max_alpha, double min_alpha, double tilt);
    
    // Wrapper untuk AnyPoint (PTZ)
    void Moil_AnyPoint(MoilHandle h, float* mapX, float* mapY, double alpha, double beta, double zoom);

#ifdef __cplusplus
}
#endif

#endif