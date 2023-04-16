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

    return (t > (powf(SIGMA, 3))) ?
           cbrtf(t) :
           (t / (3 * powf(SIGMA, 2))) + (16.0 / 116.0);

    // return (t > (powf(SIGMA, 3))) ?
    //        cbrtf(t) :
    //        ((1.0/3.0) * (powf(1.0/SIGMA, 2) * t)) + (4.0 / 29.0);
}
inline double map(double x, double start1, double stop1, double start2, double stop2) { return (((stop2 - start2)/(stop1 - start1)) * (x - start1)) + start2; }

Mat RGB_2_CIEXYZ(const Mat &);
Mat XYZ_2_CIELAB(const Mat &);
Mat CIELAB_2_XYZ(const Mat &);
Mat CIEXYZ_2_RGB(const Mat &);

// Histogram stuff
int *getImageHistogram(const Mat &);
Mat createHistogram(const Mat &, string, int = 572, int = 572);
void imgMinMax(const Mat &, int &, int &);

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

    Mat ciexyz_rgb = CIEXYZ_2_RGB(cielab_ciexyz);
    cout << "CIEXYZ_2_RGB:" << '\n'
         << "\t-> Original XYZ pixel: "      << cielab_ciexyz.at<Vec3f>(0, 0) << '\n'
         << "\t-> Pixel back in RGB space: " <<    ciexyz_rgb.at<Vec3b>(0, 0) << endl;

    // imshow("Original", src);
    // imshow("Conversion", ciexyz_rgb);

    Mat hist = createHistogram(src, "Titulo");

    waitKey();
    return 0;
}

Mat RGB_2_CIEXYZ(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_CIEXYZ: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    // const Mat M_CONV = (Mat_<float>(3, 3) <<
    //     0.4124564f, 0.3575761f, 0.1804375f,
    //     0.2126729f, 0.7151522f, 0.0721750f,
    //     0.0193339f, 0.1191920f, 0.9503041f
    // );

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
        cout << "\n\t! XYZ_2_CIELAB: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
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
        cout << "\n\t! CIELAB_2_XYZ: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
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
            f_z = f_y - (row[j][2] / 200.0);

            // f_x *= X_n;
            // f_y *= Y_n;
            // f_z *= Z_n;

            // out[j][0] = (f_x > SIGMA) ? powf(f_x, 3) : (f_x - FRAC) * 3 * powf(SIGMA, 2);
            // out[j][1] = (f_y > SIGMA) ? powf(f_y, 3) : (f_y - FRAC) * 3 * powf(SIGMA, 2);
            // out[j][2] = (f_z > SIGMA) ? powf(f_z, 3) : (f_z - FRAC) * 3 * powf(SIGMA, 2);

            out[j][0] = (f_x > SIGMA) ? (X_n * powf(f_x, 3)) : (f_x - FRAC);
            out[j][1] = (f_y > SIGMA) ? (Y_n * powf(f_y, 3)) : (f_y - FRAC);
            out[j][2] = (f_z > SIGMA) ? (Z_n * powf(f_z, 3)) : (f_z - FRAC);
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

    // const Mat M_CONV = (Mat_<float>(3, 3) <<
    //      3.2404542f, -1.5371385f, -0.4985314f,
    //     -0.9692660f,  1.8760108f,  0.0415560f,
    //      0.0556434f, -0.2040259f,  1.0572252f
    // );

    const Mat M_CONV = (Mat_<float>(3, 3) <<
         3.2410f, -1.5374f, -0.4986f,
        -0.9692f,  1.8760f,  0.0416f,
         0.0556f, -0.2040f,  1.0570f
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

int *getImageHistogram(const Mat &src) {
    if(!src.data) {
        cout << "\n\t! getImageHistogram: Image is empty! Please input an image with actual content." << endl;
    }
}

// Histogram stuff
Mat createHistogram(const Mat &src, string title, int width, int height) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data) {
        cout << "\n\t! createHistogram: Image is empty! Please input an image with actual content." << endl;
        return output;
    }

    const int MARGIN = 20;

    int fontFace = FONT_ITALIC,
        thicknessTitle = 1,
        thicknessText  = 1;

    double fontScaleTitle = 0.7,
           fontScaleText  = 0.3;

    width  = ( width < 572) ? 572 : width;
    height = (height < 572) ? 572 : height;

    Size titleSize = getTextSize(title, fontFace, fontScaleTitle, thicknessTitle, 0);

    Mat globalContainer(height, width, CV_8UC3, Vec3b(230, 230, 230));

    Mat histContainer(height - (MARGIN * 3 + titleSize.height), width - (MARGIN * 2), CV_8UC3, Vec3b(0, 0, 255)),
        gradContainer(MARGIN + 10, width - (MARGIN * 2), CV_8UC3, Vec3b(255, 0, 0));
    
    putText(
        globalContainer,
        title,
        Point((histContainer.cols / 2) - (titleSize.width / 2), MARGIN + (titleSize.height)),
        fontFace,
        fontScaleTitle,
        0,
        thicknessTitle,
        LINE_8,
        false
    );

    imshow("globalContainer", globalContainer);
    imshow("histContainer", histContainer);

    histContainer.copyTo(
        globalContainer(
            Rect(
                MARGIN,
                2 * MARGIN + titleSize.height,
                histContainer.cols,
                histContainer.rows
            )
        )
    );

    gradContainer.copyTo(
        globalContainer(
            Rect(
                MARGIN,
                MARGIN + histContainer.rows,
                gradContainer.cols,
                gradContainer.rows
            )
        )
    );

    imshow("globalContainer", globalContainer);

    switch(src.channels()) {
        case 1:
            cout << "The image has three channels" << endl;
            // histContainer = Mat(height - (MARGIN * 2), width - (MARGIN * 2), CV_8UC3, (0, 0, 255));
            // histContainer.copyTo(globalContainer(Rect(MARGIN, MARGIN, width - MARGIN, height - MARGIN)));
            // imshow("histContainer", histContainer);
        break;

        case 3:
            cout << "The image has three channels" << endl;
        break;
    }

    return output;
}