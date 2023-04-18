#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

#include "utils.hpp"

// Color space transformation
inline float CIELAB_f(float t) {
    const float SIGMA = 6.0/29.0,
                CTE = 4.0/29.0;

    return (t > (powf(SIGMA, 3))) ?
           cbrtf(t) :
           (t / (3 * powf(SIGMA, 2))) + CTE;
}

inline float CIELAB_f_1(float t) {
    const float SIGMA = 6.0/29.0,
                CTE = 4.0/29.0;
    
    return (t > SIGMA) ?
           powf(t, 3) :
           ((3 * powf(SIGMA, 2)) * (t - CTE));
}

Mat RGB_2_CIEXYZ(const Mat &);
Mat XYZ_2_CIELAB(const Mat &);
Mat CIELAB_2_XYZ(const Mat &);
Mat CIEXYZ_2_RGB(const Mat &);
Mat RGB_2_CIELAB(const Mat &);
Mat CIELAB_2_RGB(const Mat &);

int main(void) {
    srand(time(0));

    const string IMG_PATH = "./res/",
                 IMG_EXT = ".jpg",
                 IMG_SRC_NAME = "test3",
                 IMG_TRG_NAME = "test7",
                 IMG_SRC_FILENAME = IMG_PATH + IMG_SRC_NAME + IMG_EXT,
                 IMG_TRG_FILENAME = IMG_PATH + IMG_TRG_NAME + IMG_EXT;

    Mat src = imread(IMG_SRC_FILENAME),
        trg = imread(IMG_TRG_FILENAME),
        output = colorTransfer(src, trg);

    Mat src_lab = RGB_2_CIELAB(src),
        trg_lab = RGB_2_CIELAB(trg);
    
    Mat src_rgb = CIELAB_2_RGB(src_lab);

    float min_l = src_lab.at<Vec3f>(0, 0)[0],
          min_a = src_lab.at<Vec3f>(0, 0)[1],
          min_b = src_lab.at<Vec3f>(0, 0)[2],
          max_l = src_lab.at<Vec3f>(0, 0)[0],
          max_a = src_lab.at<Vec3f>(0, 0)[1],
          max_b = src_lab.at<Vec3f>(0, 0)[2];
    
    for(int i = 0; i < src_lab.rows; i++) {
        Vec3f *row = src_lab.ptr<Vec3f>(i);
        for(int j = 0; j < src_lab.cols; j++) {
            if(row[j][0] <= min_l) min_l = row[j][0];
            if(row[j][1] <= min_a) min_a = row[j][1];
            if(row[j][2] <= min_b) min_b = row[j][2];

            if(row[j][0] >= max_l) max_l = row[j][0];
            if(row[j][1] >= max_a) max_a = row[j][1];
            if(row[j][2] >= max_b) max_b = row[j][2];
        }
    }

    imshow(IMG_SRC_NAME, src);
    imshow(IMG_SRC_NAME + " converted", src_rgb);

    cout << "From source, the values:\n"
         << "\tL channel:\n"
         << "\t\t- Min (L channel): " << min_l << '\n'
         << "\t\t- Max (L channel): " << max_l << '\n'
         << "\ta channel:\n"
         << "\t\t- Min (a channel): " << min_a << '\n'
         << "\t\t- Max (a channel): " << max_a << '\n'
         << "\tb channel:\n"
         << "\t\t- Min (b channel): " << min_b << '\n'
         << "\t\t- Max (b channel): " << max_b << endl;

    waitKey();
}

Mat RGB_2_CIEXYZ(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_CIEXYZ: Image is empty orh monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f
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

Mat CIEXYZ_2_CIELAB(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! CIEXYZ_2_CIELAB: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    // Tristimulus values from Illuminant D65 2°
    const float X_n = 0.950489,
                Y_n = 1.000000,
                Z_n = 1.088840;

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

Mat CIELAB_2_CIEXYZ(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! CIELAB_2_CIEXYZ: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    float f_y = 0,
          f_x = 0,
          f_z = 0;

    // Tristimulus values from Illuminant D65 2°
    const float X_n = 0.950489,
                Y_n = 1.000000,
                Z_n = 1.088840;

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *)    src.ptr<Vec3f>(i),
              *out = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++) {
            f_x = CIELAB_f_1((row[j][0] + 16.0)/116.0 + (row[j][1]/500.0));
            f_y = CIELAB_f_1((row[j][0] + 16.0)/116.0);
            f_z = CIELAB_f_1((row[j][0] + 16.0)/116.0 - (row[j][2]/200.0));

            out[j][0] = X_n * f_x;
            out[j][1] = Y_n * f_y;
            out[j][2] = Z_n * f_z;
        }
    }

    return output;
}

Mat CIEXYZ_2_RGB(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! CIEXYZ_2_RGB: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    // Convert back again to a range between 0-255 and uchar
    normalize(output, output, 0, 255, NORM_MINMAX);
    output.convertTo(output, CV_8UC3);

    return output;
}

Mat RGB_2_CIELAB(const Mat &src) {
    return CIEXYZ_2_CIELAB(RGB_2_CIEXYZ(src));
}

Mat CIELAB_2_RGB(const Mat &src) {
    return CIEXYZ_2_RGB(CIELAB_2_CIEXYZ(src));
}
