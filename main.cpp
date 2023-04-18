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
float distance(const Scalar &, const Scalar &);

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

vector<Mat> getHists(const Mat &, int, int, int);
vector<Mat> getCumHists(const vector<Mat> &);
vector<Mat> getCumHists(const Mat &, int, int, int);

Mat showHist(const vector<Mat> &);

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

    Mat srcLab = RGB_2_CIELAB(src),
        trg_lab = RGB_2_CIELAB(trg);

    imshow("Src image", src);

    vector<Mat> rgbHists = getHists(src, 256, 0, 255);
    Mat rgbHistContainer = showHist(rgbHists);
    imshow("RGB Hist", rgbHistContainer);

    vector<Mat> rgbCumHists = getCumHists(rgbHists);
    Mat rgbCumHistContainer = showHist(rgbCumHists);
    imshow("RGB Cum Hist", rgbCumHistContainer);

    vector<Mat> labHists = getHists(srcLab, 256, -127, 128);
    Mat histContainer = showHist(labHists);
    imshow("CIELAB Hist", histContainer);

    vector<Mat> cumHists = getCumHists(labHists);
    Mat histCumContainer = showHist(cumHists);
    imshow("CIELAB Cum Hist", histCumContainer);

    // for(int i = 0; i < cumHists.size(); i++) {
    //     normalize(cumHists[i], cumHists[i], 0, 1, NORM_MINMAX);
    //     for(int j = 1; j < cumHists[i].rows; j++) {

    //     }
    // }

    waitKey();
}

float distance(const Scalar &p1, const Scalar &p2) {
    float sum = 0;
    for(int i = 0; i < p1.channels; i++) {
        sum += (p2(i) - p1(i)) * (p2(i) - p1(i));
    }

    return sqrt(sum);
}

vector<Mat> getHists(const Mat &src, int hSize, int minR, int maxR) {
    vector<Mat> srcChannels;
    split(src, srcChannels);

    int histSize = hSize;
    float range[] = {minR, maxR};
    const float *histRange[] = {range};

    vector<Mat> hists(src.channels());
    for(int i = 0; i < src.channels(); i++)
        calcHist(&srcChannels[i], 1, 0, Mat(), hists[i], 1, &histSize, histRange, true, false);

    return hists;
}

vector<Mat> getCumHists(const vector<Mat> &hists) {
    // Calculating cumulative hist
    vector<Mat> cumHists(hists.size());
    for(int i = 0; i < hists.size(); i++)
        cumHists[i] = hists[i].clone();

    for(int i = 0; i < cumHists.size(); i++)
        for(int j = 1; j < cumHists[i].rows; j++)
            cumHists[i].at<float>(j) += cumHists[i].at<float>(j - 1);

    return cumHists;
}

vector<Mat> getCumHists(const Mat &src, int hSize, int minR, int maxR) {
    vector<Mat> cumHists = getHists(src, hSize, minR, maxR);

    for(int i = 0; i < cumHists.size(); i++)
        for(int j = 1; j < cumHists[i].rows; j++)
            cumHists[i].at<float>(j) += cumHists[i].at<float>(j - 1);
    
    return cumHists;
}

Mat showHist(const vector<Mat> &hists) {
    const int CONTAINER_PADDING = 20,
              HIST_PADDING = 10;
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
        int b = randInt(0, 255),
            g = randInt(0, 255),
            r = randInt(0, 255);

        Scalar histColor(b, g, r);

        line(
            histImage,
            Point(HIST_PADDING * 1.2, ((HIST_PADDING * 1.2) * (i + 1)) + textSize.height/2 + 5),
            Point(HIST_PADDING + 15,  ((HIST_PADDING * 1.2) * (i + 1)) + textSize.height/2 + 5),
            histColor,
            2,
            8,
            0
        );

        putText(
            histImage,
            "Channel " + to_string(i),
            Point(HIST_PADDING + 25, (((HIST_PADDING * 1.2) + textSize.height / 2) * (i + 1)) + 5),
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

    rectangle(
        histImage,
        Point(HIST_PADDING - 5, HIST_PADDING - textSize.height + 5),
        Point(HIST_PADDING + 25 + textSize.width + 5, HIST_PADDING + ((textSize.height * 2) * histsCopies.size()) + 5),
        0,
        1,
        LINE_8
    );

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
