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

vector<vector<float>> getThresholds(const vector<Mat> &, int);
vector<vector<Mat>> getSegments(const Mat &, const Mat &, const vector<Mat> &, const vector<vector<float>> &, int);
vector<Mat> constructImageFromSegments(const Mat &, const vector<vector<Mat>> &);

// Debugging functions
void printThresholds(const vector<vector<float>> &);
void showSegments(const vector<vector<Mat>> &);

int main(void) {
    srand(time(0));

    const string IMG_PATH = "./res/",
                 IMG_EXT = ".jpg",
                 IMG_SRC_NAME = "test10",
                 IMG_TRG_NAME = "test8",
                 IMG_SRC_FILENAME = IMG_PATH + IMG_SRC_NAME + IMG_EXT,
                 IMG_TRG_FILENAME = IMG_PATH + IMG_TRG_NAME + IMG_EXT;

    Mat src = imread(IMG_SRC_FILENAME),
        trg = imread(IMG_TRG_FILENAME);

    Mat srcLab  = RGB_2_CIELAB(src),
        trgLab  = RGB_2_CIELAB(trg);

    imshow("Src image", src);
    imshow("Trg image", trg);

    // RGB histograms (normal and cumulative)
    vector<Mat> srcRgbHists    = getHists(src, 256, 0, 255),
                trgRgbHists    = getHists(trg, 256, 0, 256),
                srcRgbCumHists = getCumHists(srcRgbHists),
                trgRgbCumHists = getCumHists(trgRgbHists);

    vector<Mat> srcLabHists = getHists(srcLab, 256, -127, 128),
                trgLabHists = getHists(trgLab, 256, -127, 128),
                srcCumHists, trgCumHists;
    srcLabHists.erase(srcLabHists.begin()); // We ignore the L* component for this application
    trgLabHists.erase(trgLabHists.begin()); // We ignore the L* component for this application
    srcCumHists = getCumHists(srcLabHists);
    trgCumHists = getCumHists(trgLabHists);

    // Channel segmentations
    const int   SEGMENTS   = 4,
                THRESHOLDS = SEGMENTS - 1,
                CHANNELS   = srcCumHists.size();

    // Thresholds for each channel
    vector<vector<float>> srcThresholds = getThresholds(srcCumHists, SEGMENTS);
    vector<vector<float>> trgThresholds = getThresholds(trgCumHists, SEGMENTS);

    // Masking image according to the segments
    vector<vector<Mat>> srcSegmentedImages = getSegments(src, srcLab, srcCumHists, srcThresholds, SEGMENTS);
    vector<vector<Mat>> trgSegmentedImages = getSegments(trg, trgLab, trgCumHists, trgThresholds, SEGMENTS);
    vector<vector<Mat>> output(srcSegmentedImages.size());

    // Transfer colors between segments
    for(int i = 0; i < srcSegmentedImages.size(); i++) {
        for(int j = 0; j < srcSegmentedImages[i].size(); j++) {
            Mat temp = colorTransfer(
                    CIELAB_2_RGB(srcSegmentedImages[i][j]),
                    CIELAB_2_RGB(trgSegmentedImages[i][j])
                );
            output[i].push_back(temp.clone());
        }
    }

    // Showing segments
    // for(int i = 0; i < output.size(); i++)
    //     for(int j = 0; j < output[i].size(); j++)
    //         imshow("Channel " + to_string(i) + ", segment " + to_string(j), output[i][j]);

    // Merging all segments again
    vector<Mat> reMerged(output.size());
    for(int i = 0; i < output.size(); i++) {
        Mat temp = Mat::zeros(output[i][0].rows, output[i][0].cols, CV_8UC3);
        for(int j = 0; j < output[i].size(); j++)
            for(int r = 0; r < output[i][j].rows; r++) {
                Vec3b *row = (Vec3b *) output[i][j].ptr<Vec3b>(r),
                      *out = (Vec3b *)         temp.ptr<Vec3b>(r);
                for(int c = 0; c < output[i][j].cols; c++)
                    if(row[c] != Vec3b(1, 1, 1))
                        out[c] = row[c];
            }
        reMerged[i] = temp.clone();
    }

    for(int i = 0; i < reMerged.size(); i++)
        imshow("Output from channel " + to_string(i), reMerged[i]);

    // Showing image segments
    // showSegments(srcSegmentedImages);
    // showSegments(trgSegmentedImages);
    // showSegments(output);

    // Rejoining image segments into one image
    // vector<Mat> srcRejoined = constructImageFromSegments(src, srcSegmentedImages);
    // vector<Mat> trgRejoined = constructImageFromSegments(trg, trgSegmentedImages);
    // vector<Mat> outRejoined = constructImageFromSegments(src, output);

    // for(int i = 0; i < srcRejoined.size(); i++)
    //     imshow("Source image from channel " + to_string(i), srcRejoined[i]);

    // for(int i = 0; i < trgRejoined.size(); i++)
    //     imshow("Target image from channel " + to_string(i), trgRejoined[i]);

    // for(int i = 0; i < outRejoined.size(); i++)
    //     imshow("Output image from channel " + to_string(i) + ".jpg", outRejoined[i]);

    waitKey();
    return 0;
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

vector<vector<float>> getThresholds(const vector<Mat> &cumHists, int SEGMENTS) {
    const int   THRESHOLDS = SEGMENTS - 1,
                CHANNELS   = cumHists.size();
    const float STEP       = 1.0 / SEGMENTS; // Gap between each threshold

    vector<vector<float>> thresholds(CHANNELS);

    // split(srcLab, labChannels);
    // labChannels.erase(labChannels.begin());

    // Computing thresholds according to the Segments
    // First we iterate over each channel
    Mat normHist;
    for(int i = 0; i < CHANNELS; i++) {
        // float min = 0, max = 0;
        // channelMinMax(labChannels[i], min, max);
        // cout << "Channel " <<   i << ":\n"
        //      << "\tMin: "  << min << '\n'
        //      << "\tMax: "  << max << '\n';

        // We make the values from the cumulative histogram from its range into a probability
        normalize(cumHists[i], normHist, 0, 1, NORM_MINMAX);

        // Find the thresholds
        for(int j = 0; j < THRESHOLDS; j++) {
            int thresh = 0;
            for(int k = 0; k < normHist.rows; k++)
                if(normHist.at<float>(k) <= (STEP * (j + 1)))
                    thresh = k;
            cout << "\tThreshold at " << thresh << "(" << mapNum(thresh, 0, 255, -127, 128) << "): " << normHist.at<float>(thresh) << endl;
            thresholds[i].push_back(mapNum(thresh, 0, 255, -127, 128)); // We push the threshold into the thresholds of channel i.
                                                                        // Since the histogram goes from [0, 255],
                                                                        // we map the threshold from that range to [-127, +128]
                                                                        // because a* and b* values are between that range
        }
        cout << endl;
    }

    return thresholds;
}

vector<vector<Mat>> getSegments(const Mat &src, const Mat &srcLab, const vector<Mat> &cumHists, const vector<vector<float>> &thresholds, int SEGMENTS) {
    vector<vector<Mat>> segmentedImages;
    vector<Mat> segments;

    const int   THRESHOLDS = SEGMENTS - 1,
                CHANNELS   = cumHists.size();
    
    for(int i = 0; i < CHANNELS; i++) {
        segments.clear();

        cout << "From channel " << i << "\n\tThe segment 0 goes from -127 to " << thresholds[i][0] << endl;
        // Creating image for segment 0 (values between [rangeMin, 0], in this case, rangeMin is -127)
        Mat temp = Mat::zeros(srcLab.rows, srcLab.cols, CV_32FC3);
        for(int j = 0; j < temp.rows; j++) {
            Vec3f *row = (Vec3f *) srcLab.ptr<Vec3f>(j),
                  *out = (Vec3f *)   temp.ptr<Vec3f>(j);

            for(int k = 0; k < temp.cols; k++)
                if(row[k][i + 1] >= -127 && row[k][i + 1] < thresholds[i][0])
                    out[k] = row[k];
        }
        segments.push_back(temp.clone());

        // Creating image for segments [1, SEGMENTS - 1) (values between the previous threshold and the current one)
        for(int j = 1; j < THRESHOLDS; j++) {
            temp = Mat::zeros(src.rows, src.cols, CV_32FC3);
            cout << "\tThe segment " << j << " goes from " << thresholds[i][j - 1] << " to " << thresholds[i][j] << endl;
            for(int k = 0; k < temp.rows; k++) {
                Vec3f *row = (Vec3f *) srcLab.ptr<Vec3f>(k),
                      *out = (Vec3f *)   temp.ptr<Vec3f>(k);

                for(int l = 0; l < temp.cols; l++) {
                    if(row[l][i + 1] >= thresholds[i][j - 1] && row[l][i + 1] < thresholds[i][j])
                        out[l] = row[l];
                }

            }
            segments.push_back(temp.clone());
        }

        // Creating image for the last segment (values between [lastThreshold, rangeMax], in this case, rangeMax is 128)
        temp = Mat::zeros(src.rows, src.cols, CV_32FC3);
        cout << "\tThe segment " << THRESHOLDS << " goes from " << thresholds[i][THRESHOLDS - 1] << " to " << 128 << endl;
        for(int j = 0; j < temp.rows; j++) {
            Vec3f *row = (Vec3f *) srcLab.ptr<Vec3f>(j),
                  *out = (Vec3f *)   temp.ptr<Vec3f>(j);

            for(int k = 0; k < temp.cols; k++)
                if(row[k][i + 1] >= thresholds[i][THRESHOLDS - 1] && row[k][i + 1] < 128)
                    out[k] = row[k];
        }

        segments.push_back(temp.clone());
        segmentedImages.push_back(segments);
    }

    return segmentedImages;
}

vector<Mat> constructImageFromSegments(const Mat &src, const vector<vector<Mat>> &segmentedImages) {
    vector<Mat> rejoined;
    for(int i = 0; i < segmentedImages.size(); i++) {
        Mat temp = Mat::zeros(src.rows, src.cols, CV_32FC3);
        cout << "From channel " << i << ": " << endl;
        for(int j = 0; j < segmentedImages[i].size(); j++) {
            cout << "\t- Segment " << j << ": " << endl;
            for(int k = 0; k < segmentedImages[i][j].rows; k++) {
                Vec3f *row = (Vec3f *) segmentedImages[i][j].ptr<Vec3f>(k),
                      *out = (Vec3f *)                  temp.ptr<Vec3f>(k);
                cout << "\t\t- First pixel of row " << k << ": " << row[0] << endl;
                for(int l = 0; l < segmentedImages[i][j].cols; l++) {
                    if(row[l] != Vec3f(0.0, 0.0, 0.0)) {
                        out[l] = row[l];
                    }
                }
                // imshow("Reconstructing image from channel and its segments", CIELAB_2_RGB(temp));
                // waitKey(1);
            }
            // imshow("Reconstructing image from channel and its segments", CIELAB_2_RGB(temp));
            // waitKey();
        }
        rejoined.push_back(CIELAB_2_RGB(temp));
    }
    return rejoined;
}

// Debugging functions
void printThresholds(const vector<vector<float>> &thresholds) {
    cout << "[";
    for(int i = 0; i < thresholds.size(); i++) {
        cout << "[";
        for(int j = 0; j < thresholds[i].size(); j++) {
            cout << (float) thresholds[i][j] << ((j + 1 == thresholds[i].size()) ? "" : ", ");
        }
        cout << "]" << ((i + 1) == thresholds.size() ? "" : ", ");
    }
    cout << "]" << endl;

}

void showSegments(const vector<vector<Mat>> &segmentedImages) {
    for(int i = 0; i < segmentedImages.size(); i++) {
        for(int j = 0; j < segmentedImages[i].size(); j++)
            imshow(
                "Channel " + to_string(i) + ", segment " + to_string(j),
                CIELAB_2_RGB(segmentedImages[i][j]));
    }
}
