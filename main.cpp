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

    const int SEGMENTS = 4,
              THRESHOLDS = SEGMENTS - 1;
    const float STEP = 1.0 / SEGMENTS;
    vector<vector<float>> thresholds(SEGMENTS - 1);

    cout << "Thresholds: " << SEGMENTS << '\n'
         << "Step: " << STEP << endl;

    for(int i = 0; i < cumHists.size(); i++) {
        normalize(cumHists[i], cumHists[i], 0, 1, NORM_MINMAX);
        for(int j = 0; j < THRESHOLDS; j++) {
            int thresh = 0;
            for(int k = 0; k < cumHists[i].rows; k++)
                if(cumHists[i].at<float>(k) <= (STEP * (j + 1)))
                    thresh = k;
                cout << cumHists[i].at<float>(thresh) << endl;
            thresholds[i].push_back(thresh);
        }
    }

    cout << "[";
    for(int i = 0; i < thresholds.size(); i++) {
        cout << "[";
        for(int j = 0; j < thresholds[i].size(); j++) {
            cout << thresholds[i][j] << ((j + 1 == thresholds[i].size()) ? "" : ", ");
        }
        cout << "]" << ((i + 1) == thresholds.size() ? "" : ", ");
    }
    cout << "]" << endl;

    waitKey();
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
