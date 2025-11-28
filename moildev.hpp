#ifndef MOILDEV_HPP
#define MOILDEV_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class Moildev {
public:
    // Constructor & Destructor
    Moildev();
    ~Moildev();

    /**
     * Konfigurasi Parameter Kamera
     * Urutan: camera_name, sensor_width, sensor_height, iCx, iCy, ratio, 
     * image_width, image_height, calibration_ratio, para0..para5
     */
    void Config(std::string name, double sensor_width, double sensor_height, 
                double iCx, double iCy, double ratio, 
                double image_width, double image_height, 
                double calibration_ratio, 
                double para0, double para1, double para2, 
                double para3, double para4, double para5);

    /**
     * Generate Peta Panorama (MapsPanoramaM_Rt)
     * Sesuai library Moildevren.o: (float* mapX, float* mapY, double max_alpha, double min_alpha, double tilt)
     */
    void MapsPanoramaM_Rt(float* mapX, float* mapY, double max_alpha, double min_alpha, double tilt);

    /**
     * Generate Peta Anypoint (AnyPointM)
     * (float* mapX, float* mapY, double alpha, double beta, double zoom)
     */
    void AnyPointM(float* mapX, float* mapY, double alpha, double beta, double zoom);

    // Fungsi utilitas lain yang mungkin ada di library (opsional untuk deklarasi)
    double getRhoFromAlpha(double alpha);
    double getAlphaFromRho(int rho);
};

#endif