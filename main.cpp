#include <iostream>
#include <string>
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

// Utils
inline int randInt(int min, int max) { return (rand() % (max - min)) + min; }
template<class T>
void channelMinMax(const Mat &src, T &min, T &max) {
    if(!src.data) {
        cout << "channelMinMax: ! Image is empty. Please enter a valid image." << endl;
        return;
    }

    // Assuming src is of type CV_32FC1
    min = src.at<float>(0, 0);
    max = min;

    for(int i = 0; i < src.rows; i++) {
        float *row = (float *) src.ptr<float>(i);
        for(int j = 0; j < src.cols; j++) {
            if(row[j] <= min) min = row[j];
            if(row[j] >= max) max = row[j];
        }
    }
}

Mat showHist(const vector<Mat> &);

int main(void) {
    srand(time(0));

    const string IMG_PATH = "./res/",
                 IMG_EXT = ".jpg",
                 IMG_SRC_NAME = "test2",
                 IMG_TRG_NAME = "test7",
                 IMG_SRC_FILENAME = IMG_PATH + IMG_SRC_NAME + IMG_EXT,
                 IMG_TRG_FILENAME = IMG_PATH + IMG_TRG_NAME + IMG_EXT;

    Mat src = imread(IMG_SRC_FILENAME),
        trg = imread(IMG_TRG_FILENAME),
        output = colorTransfer(src, trg);

    Mat srcLab = RGB_2_CIELAB(src),
        trg_lab = RGB_2_CIELAB(trg);

    vector<Mat> srcLabChannels;
    split(srcLab, srcLabChannels);

    cout << srcLabChannels[0].type() << endl;

    float min_l, min_a, min_b,
          max_l, max_a, max_b;

    channelMinMax(srcLabChannels[0], min_l, max_l);
    channelMinMax(srcLabChannels[1], min_a, max_a);
    channelMinMax(srcLabChannels[2], min_b, max_b);

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

    int histSize = 256;
    float range[] = {-127, 128};
    const float *histRange[] = {range};

    vector<Mat> hists(srcLabChannels.size());

    calcHist(&srcLabChannels[0], 1, 0, Mat(), hists[0], 1, &histSize, histRange, true, false);
    calcHist(&srcLabChannels[1], 1, 0, Mat(), hists[1], 1, &histSize, histRange, true, false);
    calcHist(&srcLabChannels[2], 1, 0, Mat(), hists[2], 1, &histSize, histRange, true, false);

    Mat histContainer = showHist(hists);
    imshow("Hist", histContainer);

    // Calculating cumulative hist
    vector<Mat> cum_hists(hists.size());
    for(int i = 0; i < hists.size(); i++)
        cum_hists[i] = hists[i].clone();

    for(int i = 0; i < cum_hists.size(); i++)
        for(int j = 1; j < cum_hists[i].rows; j++)
            cum_hists[i].at<float>(j) += cum_hists[i].at<float>(j - 1);
    
    Mat histCumContainer = showHist(cum_hists);
    imshow("Cum Hist", histCumContainer);

    // normalize(l_hist_cum, l_hist_cum, 0, 1, NORM_MINMAX);
    // normalize(a_hist_cum, a_hist_cum, 0, 1, NORM_MINMAX);
    // normalize(b_hist_cum, b_hist_cum, 0, 1, NORM_MINMAX);

    waitKey();
}

Mat showHist(const vector<Mat> &hists) {
    const int CONTAINER_PADDING = 20;
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w/hists[0].rows);

    Mat histContainer(hist_h + CONTAINER_PADDING * 2, hist_w + CONTAINER_PADDING * 2, CV_8UC3, Scalar(230, 230, 230)),
        histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

    vector<Mat> histsCopies(hists.size());
    for(int i = 0; i < hists.size(); i++)
        histsCopies[i] = hists[i].clone();

    for(int i = 0; i < histsCopies.size(); i++)
        normalize(histsCopies[i], histsCopies[i], 0, histImage.rows, NORM_MINMAX, -1, Mat());

    int fontFace = FONT_ITALIC,
        thicknessText = 1;

    double fontScaleText = 0.3;
    Size textSize = getTextSize("Channel 1", fontFace, fontScaleText, thicknessText, 0);

    for(int i = 0; i < histsCopies.size(); i++) {
        Scalar histColor(
            randInt(0, 255),
            randInt(0, 255),
            randInt(0, 255)
        );

        line(
            histImage,
            Point(10, (10 * (i + 1)) + textSize.height / 2),
            Point(25, (10 * (i + 1)) + textSize.height / 2),
            histColor,
            2,
            8,
            0
        );

        putText(
            histImage,
            "Channel " + to_string(i),
            Point(35, ((10 + textSize.height / 2) * (i + 1))),
            fontFace,
            fontScaleText,
            0,
            thicknessText,
            LINE_8,
            false
        );

        for(int j = 1; j < histsCopies[i].rows; j++) {
            line(
                histImage,
                Point(bin_w * (j - 1), hist_h - cvRound(histsCopies[i].at<float>(j - 1))),
                Point(bin_w * (j),     hist_h - cvRound(histsCopies[i].at<float>(j))),
                histColor,
                2,
                8,
                0
            );
        }
    }

    histImage.copyTo(histContainer(Rect(CONTAINER_PADDING, CONTAINER_PADDING, histImage.cols, histImage.rows)));
    rectangle(
        histContainer,
        Point(CONTAINER_PADDING, CONTAINER_PADDING),
        Point(CONTAINER_PADDING + histImage.cols, CONTAINER_PADDING + histImage.rows),
        0,
        1,
        LINE_8
    );

    return histContainer;
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
