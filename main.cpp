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
                 IMG_SRC_NAME = "test1",
                 IMG_TRG_NAME = "test7",
                 IMG_SRC_FILENAME = IMG_PATH + IMG_SRC_NAME + IMG_EXT,
                 IMG_TRG_FILENAME = IMG_PATH + IMG_TRG_NAME + IMG_EXT;

    Mat src = imread(IMG_SRC_FILENAME),
        trg = imread(IMG_TRG_FILENAME),
        output = colorTransfer(src, trg);

    Mat srcLab  = RGB_2_CIELAB(src),
        trg_lab = RGB_2_CIELAB(trg);

    imshow("Src image", src);

    // RGB histograms
    vector<Mat> rgbHists = getHists(src, 256, 0, 255);
    Mat rgbHistContainer = showHist(rgbHists);
    imshow("RGB Hist", rgbHistContainer);

    // RGB Cumulative histogram
    vector<Mat> rgbCumHists = getCumHists(rgbHists);
    Mat rgbCumHistContainer = showHist(rgbCumHists);
    imshow("RGB Cum Hist", rgbCumHistContainer);

    // LAB histograms
    vector<Mat> labHists = getHists(srcLab, 256, -127, 128);
    labHists.erase(labHists.begin());

    Mat histContainer = showHist(labHists);
    imshow("CIELAB Hist", histContainer);

    // LAB Cumulative histogram
    vector<Mat> cumHists = getCumHists(labHists);
    Mat histCumContainer = showHist(cumHists);
    imshow("CIELAB Cum Hist", histCumContainer);

    const int SEGMENTS = 4,
              THRESHOLDS = SEGMENTS - 1;
    const float STEP = 1.0 / SEGMENTS;
    vector<vector<float>> thresholds(cumHists.size());

    cout << "Image size: " << src.size() << '\n'
         << "Segments: " << SEGMENTS << '\n'
         << "Thresholds: " << THRESHOLDS << '\n'
         << "Step: " << STEP << '\n'
         << "CumHists size: " << cumHists.size() << endl << endl;

    vector<Mat> labChannels;
    split(srcLab, labChannels);
    labChannels.erase(labChannels.begin());
    // Computing thresholds according to the Segments
    for(int i = 0; i < cumHists.size(); i++) {
        float min = 0, max = 0;
        channelMinMax(labChannels[i], min, max);

        normalize(cumHists[i], cumHists[i], 0, 1, NORM_MINMAX);
        cout << "Channel " <<   i << ":\n"
             << "\tMin: "  << min << '\n'
             << "\tMax: "  << max << '\n';
        for(int j = 0; j < THRESHOLDS; j++) {
            int thresh = 0;
            for(int k = 0; k < cumHists[i].rows; k++)
                if(cumHists[i].at<float>(k) <= (STEP * (j + 1)))
                    thresh = k;
            cout << "\tThreshold at " << thresh << "(" << mapNum(thresh, 0, 255, -127, 128) << "): " << cumHists[i].at<float>(thresh) << endl;
            // thresholds[i].push_back(mapNum(thresh, 0, 255, 0, 1));
            // thresholds[i].push_back(mapNum(thresh, 0, 255, min, max));
            thresholds[i].push_back(mapNum(thresh, 0, 255, -127, 128));
            // thresholds[i].push_back(thresh);
            // thresholds[i].push_back(cumHists[i].at<float>(thresh));
        }
        cout << endl;
    }

    // Printing thresholds
    cout << "Thresholds: [";
    for(int i = 0; i < thresholds.size(); i++) {
        cout << "[";
        for(int j = 0; j < thresholds[i].size(); j++) {
            cout << (float) thresholds[i][j] << ((j + 1 == thresholds[i].size()) ? "" : ", ");
        }
        cout << "]" << ((i + 1) == thresholds.size() ? "" : ", ");
    }
    cout << "]" << endl;

    vector<vector<Mat>> segmentedImages;
    vector<Mat> segments;

    // Masking image according to the segments
    for(int i = 0; i < cumHists.size(); i++) {
        segments.clear();

        cout << "From channel " << i << "\n\tThe segment 0 goes from -127 to " << thresholds[i][0] << endl;
        // Creating image for segment 0
        Mat temp = Mat::zeros(srcLab.rows, srcLab.cols, CV_32FC3);
        for(int j = 0; j < temp.rows; j++) {
            Vec3f *row = (Vec3f *) srcLab.ptr<Vec3f>(j),
                  *out = (Vec3f *)   temp.ptr<Vec3f>(j);

            for(int k = 0; k < temp.cols; k++)
                // if(row[k][i] >= -127 && row[k][i] < thresholds[i][0]) {
                if(row[k][i + 1] >= -127 && row[k][i + 1] < thresholds[i][0]) {
                    // cout << "\tPoint " << row[k][i + 1] << " is in range" << endl;
                    out[k] = row[k];
                }
                // if(row[k][i] >= 0 && row[k][i] < thresholds[i][0])
        }
        segments.push_back(temp.clone());

        // Creating image for segments 1-(SEGMENTS - 1)
        for(int j = 1; j < THRESHOLDS; j++) {
            temp = Mat::zeros(src.rows, src.cols, CV_32FC3);
            cout << "\tThe segment " << j << " goes from " << thresholds[i][j - 1] << " to " << thresholds[i][j] << endl;
            for(int k = 0; k < temp.rows; k++) {
                Vec3f *row = (Vec3f *) srcLab.ptr<Vec3f>(k),
                      *out = (Vec3f *)   temp.ptr<Vec3f>(k);

                for(int l = 0; l < temp.cols; l++) {
                    // if(row[l][i]     >= thresholds[i][j - 1] && row[l][i] < thresholds[i][j])
                    if(row[l][i + 1] >= thresholds[i][j - 1] && row[l][i + 1] < thresholds[i][j])
                        out[l] = row[l];
                }

            }
            segments.push_back(temp.clone());
        }

        // Creating image for the last segment
        temp = Mat::zeros(src.rows, src.cols, CV_32FC3);
        cout << "\tThe segment " << THRESHOLDS << " goes from " << thresholds[i][THRESHOLDS - 1] << " to " << 128 << endl;
        for(int j = 0; j < temp.rows; j++) {
            Vec3f *row = (Vec3f *) srcLab.ptr<Vec3f>(j),
                  *out = (Vec3f *)   temp.ptr<Vec3f>(j);

            for(int k = 0; k < temp.cols; k++)
                // if(row[k][i] >= thresholds[i][THRESHOLDS] && row[k][i] < 1.0)
                // if(row[k][i] >= thresholds[i][THRESHOLDS] && row[k][i] < 128)
                if(row[k][i + 1] >= thresholds[i][THRESHOLDS - 1] && row[k][i + 1] < 128)
                    out[k] = row[k];
        }

        segments.push_back(temp.clone());
        segmentedImages.push_back(segments);
    }

    // Showing image segments
    for(int i = 0; i < segmentedImages.size(); i++) {
        for(int j = 0; j < segmentedImages[i].size(); j++)
            imshow(
                "Channel " + to_string(i) + ", segment " + to_string(j),
                CIELAB_2_RGB(segmentedImages[i][j]));
        waitKey();
    }

    // Rejoining image segments into one image
    vector<Mat> rejoined;
    for(int i = 0; i < segmentedImages.size(); i++) {
        Mat temp = Mat::zeros(src.rows, src.cols, CV_32FC3);
        for(int j = 0; j < segmentedImages[i].size(); j++) {

            for(int k = 0; k < segmentedImages[i][j].rows; k++) {
                Vec3f *row = (Vec3f *) segmentedImages[i][j].ptr<Vec3f>(k),
                      *out = (Vec3f *)                  temp.ptr<Vec3f>(k);
                for(int l = 0; l < segmentedImages[i][j].cols; l++)
                    if(row[l] != Vec3f(0.0, 0.0, 0.0))
                        out[l] = row[l];

            }
            imshow("Something", CIELAB_2_RGB(temp));
            waitKey();
        }
        rejoined.push_back(CIELAB_2_RGB(temp));
    }

    for(int i = 0; i < rejoined.size(); i++)
        imshow("Image from channel " + to_string(i), rejoined[i]);

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

    Mat histContainer(hist_h + CONTAINER_PADDING * 2, hist_w + CONTAINER_PADDING * 2, CV_32FC3, Vec3f(230, 230, 230)),
        histImage(hist_h, hist_w, CV_32FC3, Vec3f(255, 255, 255));

    vector<Mat> histsCopies(hists.size());
    for(int i = 0; i < hists.size(); i++)
        histsCopies[i] = hists[i].clone();

    for(int i = 0; i < histsCopies.size(); i++)
        normalize(histsCopies[i], histsCopies[i], 0, histImage.rows, NORM_MINMAX, -1, Mat());

    int fontFace = FONT_ITALIC,
        thicknessText = 1;

    double fontScaleText = 0.3;
    Size textSize = getTextSize("Channel 1", fontFace, fontScaleText, thicknessText, 0);

    const float STEP = 360 / histsCopies.size();

    cvtColor(histImage, histImage, COLOR_BGR2HSV_FULL);
    for(int i = 0; i < histsCopies.size(); i++) {
        Scalar histColor(360 - ((int) ((i + 1) * STEP) % 360), 255, 255);

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
    }

    rectangle(
        histImage,
        Point(HIST_PADDING - 5, HIST_PADDING - textSize.height + 5),
        Point(HIST_PADDING + 25 + textSize.width + 5, HIST_PADDING + ((textSize.height * 2) * histsCopies.size()) + 5),
        0,
        1,
        LINE_8
    );
    cvtColor(histImage, histImage, COLOR_HSV2BGR_FULL);

    histImage.copyTo(histContainer(Rect(CONTAINER_PADDING, CONTAINER_PADDING, histImage.cols, histImage.rows)));
    rectangle(
        histContainer,
        Point(CONTAINER_PADDING, CONTAINER_PADDING),
        Point(CONTAINER_PADDING + histImage.cols, CONTAINER_PADDING + histImage.rows),
        0,
        1,
        LINE_8
    );

    histContainer.convertTo(histContainer, CV_8UC3);
    return histContainer;
}
