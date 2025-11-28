#include "moildev_wrapper.h"
#include "moildev.hpp"
#include <string>

// Implementation to encapsulate Moildev C++ class

MoilHandle Moil_Create() { 
    return new Moildev(); 
}

void Moil_Destroy(MoilHandle h) { 
    delete (Moildev*)h; 
}

void Moil_Config(MoilHandle h, const char* name, 
    double sensor_w, double sensor_h, double iCx, double iCy, double ratio, 
    double img_w, double img_h, double calib_ratio, 
    double p0, double p1, double p2, double p3, double p4, double p5) {
    
    // Gunakan std::string di sini, karena file ini di-compile untuk ABI yang cocok dengan libmoildevren.a
    ((Moildev*)h)->Config(std::string(name), sensor_w, sensor_h, iCx, iCy, ratio, img_w, img_h, calib_ratio, p0, p1, p2, p3, p4, p5);
}

void Moil_MapsPanorama(MoilHandle h, float* mapX, float* mapY, double max_alpha, double min_alpha, double tilt) {
    ((Moildev*)h)->MapsPanoramaM_Rt(mapX, mapY, max_alpha, min_alpha, tilt);
}

void Moil_AnyPoint(MoilHandle h, float* mapX, float* mapY, double alpha, double beta, double zoom) {
    ((Moildev*)h)->AnyPointM(mapX, mapY, alpha, beta, zoom);
}