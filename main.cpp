#include <iostream>
#include <cmath>

using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

#include "utils.hpp"

inline float CIELAB_f(float t) {
    const float SIGMA = 6.0/29.0;

    return (t > (SIGMA * SIGMA * SIGMA)) ?
           cbrtf(t) :
           (t / (3 * (SIGMA * SIGMA))) + (16.0 / 116.0);

    // return (t > (powf(SIGMA, 3))) ?
    //        cbrtf(t) :
    //        ((1.0/3.0) * (powf(1.0/SIGMA, 2) * t)) + (4.0 / 29.0);
}

Mat RGB_2_CIEXYZ(const Mat &);
Mat XYZ_2_CIELAB(const Mat &);
Mat CIELAB_2_XYZ(const Mat &);

int main(void) {
    const string IMG_PATH = "./res/",
                 IMG_EXT = ".jpg",
                 IMG_SRC_NAME = "test1",
                 IMG_TRG_NAME = "test7",
                 IMG_SRC_FILENAME = IMG_PATH + IMG_SRC_NAME + IMG_EXT,
                 IMG_TRG_FILENAME = IMG_PATH + IMG_TRG_NAME + IMG_EXT;

    Mat src = imread(IMG_SRC_FILENAME),
        trg = imread(IMG_TRG_FILENAME),
        output = colorTransfer(src, trg);

    Mat rgb_ciexyz = RGB_2_CIEXYZ(src);
    cout << "RGB_2_CIEXYZ:" << '\n'
         << "\t-> Original RGB pixel: " <<        src.at<Vec3b>(0, 0) << '\n'
         << "\t-> Pixel in XYZ space: " << rgb_ciexyz.at<Vec3f>(0, 0) << endl;

    Mat ciexyz_cielab = XYZ_2_CIELAB(rgb_ciexyz);
    cout << "XYZ_2_CIELAB:" << '\n'
         << "\t-> Original XYZ pixel: " <<    rgb_ciexyz.at<Vec3f>(0, 0) << '\n'
         << "\t-> Pixel in LAB space: " << ciexyz_cielab.at<Vec3f>(0, 0) << endl;

    Mat cielab_ciexyz = CIELAB_2_XYZ(ciexyz_cielab);
    cout << "CIELAB_2_XYZ:" << '\n'
         << "\t-> Original LAB pixel: "      << ciexyz_cielab.at<Vec3f>(0, 0) << '\n'
         << "\t-> Pixel back in XYZ space: " << cielab_ciexyz.at<Vec3f>(0, 0) << endl;

    // Mat rgb_cielab_cv;
    // src.convertTo(src, CV_32FC3, 1.0 / 255.0);
    // cvtColor(rgb_ciexyz, rgb_cielab_cv, COLOR_RGB2Lab);
    // cout << "RGB_2_CIELAB:" << '\n'
    //      << "\t-> Original RGB pixel: " <<           src.at<Vec3f>(0, 0) << '\n'
    //      << "\t-> Pixel in LAB space: " << rgb_cielab_cv.at<Vec3f>(0, 0) << endl;

    waitKey();
}

Mat RGB_2_CIEXYZ(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_CIEXYZ: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        0.4124f, 0.3576f, 0.1805f,
        0.2126f, 0.7152f, 0.0722f,
        0.0193f, 0.1192f, 0.9505f
    );

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32FC3, 1.0 / 255.0, 0);
    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < srcFloat.rows; i++) {
        Vec3f *row = (Vec3f *) srcFloat.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < srcFloat.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat XYZ_2_CIELAB(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_CIELAB: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    // Tristimulus values from Illuminant D65 2°
    const float X_n = 0.95047,
                Y_n = 1.00000,
                Z_n = 1.08883;

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++) {
            out[j][0] = 116 *  CIELAB_f((row[j][1]) / Y_n) - 16;
            out[j][1] = 500 * (CIELAB_f((row[j][0]) / X_n) - CIELAB_f(row[j][1] / Y_n));
            out[j][2] = 200 * (CIELAB_f((row[j][1]) / Y_n) - CIELAB_f(row[j][2] / Z_n));
        }
    }

    return output;
}

Mat CIELAB_2_XYZ(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_CIELAB: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    float f_y = 0,
          f_x = 0,
          f_z = 0;
    
    // Tristimulus values from Illuminant D65 2°
    const float SIGMA = 6.0/29.0,
                FRAC  = 16.0/116.0,
                X_n   = 0.95047,
                Y_n   = 1.00000,
                Z_n   = 1.08883;

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *)    src.ptr<Vec3f>(i),
              *out = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++) {
            f_y = (row[j][0] + 16.0) / 166.0;
            f_x = f_y + (row[j][1] / 500.0);
            f_z = f_y + (row[j][2] / 200.0);

            out[j][0] = (f_x > SIGMA) ? (X_n * powf(f_x, 3)) : (f_x - FRAC);
            out[j][1] = (f_y > SIGMA) ? (Y_n * powf(f_y, 3)) : (f_y - FRAC);
            out[j][2] = (f_z > SIGMA) ? (Z_n * powf(f_z, 3)) : (f_z - FRAC);

            // out[j][1] = Y_n * (1.0 / CIELAB_f((1.0 / 116.0) * (row[j][0] + 16)));
            // out[j][0] = X_n * (1.0 / CIELAB_f((1.0 / 116.0) * (row[j][0] + 16) + ((1.0 / 500.0) * row[j][1])));
            // out[j][2] = Z_n * (1.0 / CIELAB_f((1.0 / 116.0) * (row[j][0] + 16) - ((1.0 / 200.0) * row[j][2])));
        }
    }

    return output;
}