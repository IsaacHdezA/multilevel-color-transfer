#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;

using cv::Mat;
using cv::Mat_;
using cv::Vec3f;
using cv::Vec3b;

inline float fast_inv_sqrt(float n) {
    const float threehalfs = 1.5F;
    float y = n;

    long i = * ( long * ) &y;

    i = 0x5f3759df - ( i >> 1 );
    y = * ( float * ) &i;

    y = y * ( threehalfs - ( (n * 0.5F) * y * y ) );

    return y;
}

// Function prototypes
Mat   RGB_2_XYZ(const Mat &);
Mat   XYZ_2_LMS(const Mat &);
Mat   RGB_2_LMS(const Mat &);
Mat LMS_2_LAlBe(const Mat &);
Mat LAlBe_2_LMS(const Mat &);
Mat   LMS_2_RGB(const Mat &);

Mat RGB_2_LAlBe(const Mat &);
Mat LAlBe_2_RGB(const Mat &);

Mat colorTransfer(const Mat &, const Mat &);

vector<float> channelMeans(const Mat &);
vector<float>   channelStd(const Mat &, const vector<float> &);

// Function definitions
Mat RGB_2_XYZ(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        0.5141f, 0.3239f, 0.1604f,
        0.2651f, 0.6702f, 0.0641f,
        0.0241f, 0.1228f, 0.8444f
    );

    Mat srcCopy;
    src.convertTo(srcCopy, CV_32FC3);
    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < srcCopy.rows; i++) {
        Vec3f *row = (Vec3f *) srcCopy.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < srcCopy.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat XYZ_2_LMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
         0.3897f, 0.6890f, -0.0787f,
        -0.2298f, 1.1834f,  0.0432f,
         0.0000f, 0.0000f,  1.0000f
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *)    src.ptr<Vec3f>(i),
              *out = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++) {
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
            for(int k = 0; k < output.channels(); k++)
                // This is for handling logarithms. After matrix-vector product,
                // if the result is 0, sum 1.0 to avoid log(0) errors
                out[j][k] = log10((out[j][k] == 0.0000 ? 1.0000 : 0.0) + out[j][k]);
        }
    }

    return output;
}

Mat RGB_2_LMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        0.3811f, 0.5783f, 0.0402f,
        0.1967f, 0.7244f, 0.0782f,
        0.0241f, 0.1288f, 0.8444f
    );

    Mat srcCopy;
    src.convertTo(srcCopy, CV_32FC3);
    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < srcCopy.rows; i++) {
        Vec3f *row = (Vec3f *) srcCopy.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < srcCopy.cols; j++) {
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
            for(int k = 0; k < 3; k++)
                // This is for handling logarithms. After matrix-vector product,
                // if the result is 0, sum 1.0 to avoid log(0) errors
                out[j][k] = log10((out[j][k] == 0.0000 ? 1.0000 : 0.0) + out[j][k]);
        }
    }

    return output;
}

Mat LMS_2_LAlBe(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const float SQRT_3   = fast_inv_sqrt(3),
                SQRT_6   = fast_inv_sqrt(6),
                SQRT_2   = fast_inv_sqrt(2),
                SQRT_2_3 = -1 * sqrt(2.0/3.0);

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        SQRT_3,  SQRT_3,   SQRT_3,
        SQRT_6,  SQRT_6, SQRT_2_3,
        SQRT_2, -SQRT_2,   0.0000
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);
    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *)    src.ptr<Vec3f>(i),
              *out = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat LAlBe_2_LMS(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const float SQRT_3   = fast_inv_sqrt(3),
                SQRT_6   = fast_inv_sqrt(6),
                SQRT_2   = fast_inv_sqrt(2),
                SQRT_2_3 = -1 * sqrt(2.0/3.0);

    const Mat M_CONV = (Mat_<float>(3, 3) <<
        SQRT_3,   SQRT_6,  SQRT_2,
        SQRT_3,   SQRT_6, -SQRT_2,
        SQRT_3, SQRT_2_3,  0.0000
    );

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i),
              *out = (Vec3f *)  output.ptr<Vec3f>(i);
        for(int j = 0; j < src.cols; j++)
            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
    }

    return output;
}

Mat LMS_2_RGB(const Mat &src) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    const Mat M_CONV = (Mat_<float>(3, 3) <<
         4.4679f, -3.5873f,  0.1193f,
        -1.2186f,  2.3809f, -0.1624f,
         0.0497f, -0.2439f,  1.2045f
    );

    // Raising values to power ten to go back to linear space
    Mat log_pow10 = src.clone();

    output = Mat::zeros(src.rows, src.cols, CV_32FC3);
    for(int i = 0; i < log_pow10.rows; i++) {
        Vec3f *row = (Vec3f *) log_pow10.ptr<Vec3f>(i),
              *out = (Vec3f *)    output.ptr<Vec3f>(i);
        for(int j = 0; j < log_pow10.cols; j++) {
            // Once again, the values that were 0, now are 1, so we turn them to 0 back again.
            for(int k = 0; k < log_pow10.channels(); k++) {
                row[j][k] = pow(10, row[j][k]);
                row[j][k] = ((row[j][k] == 1.0000) ? 0 : row[j][k]);
            }

            gemm(M_CONV, row[j], 1.0, Mat(), 0.0, out[j]);
        }
    }

    // Convert image to 3-channel uchar
    output.convertTo(output, CV_8UC3);

    return output;
}

Mat RGB_2_LAlBe(const Mat &src) {
    return (LMS_2_LAlBe(RGB_2_LMS(src)));
}

Mat LAlBe_2_RGB(const Mat &src) {
    return (LMS_2_RGB(LAlBe_2_LMS(src)));
}

Mat colorTransfer(const Mat &src, const Mat &trg) {
    Mat output = Mat::zeros(1, 1, CV_32FC3);

    if(!src.data || src.channels() == 1) {
        cout << "\n\t! RGB_2_LMS: Image is empty or monochromatic. Should be three channels (BGR)." << endl;
        return output;
    }

    Mat src_lalbe = RGB_2_LAlBe(src),
        trg_lalbe = RGB_2_LAlBe(trg);

    output = src_lalbe.clone();

    const int M = src.rows,
              N = src.cols;

    vector<float> src_means = channelMeans(src_lalbe),
                  trg_means = channelMeans(trg_lalbe),
                  src_std   =   channelStd(src_lalbe, src_means),
                  trg_std   =   channelStd(trg_lalbe, trg_means);

    // Substracting mean to original pixels
    for(int i = 0; i < M; i++) {
        Vec3f *row = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < N; j++) {
            row[j][0] -= src_means[0];
            row[j][1] -= src_means[1];
            row[j][2] -= src_means[2];
        }
    }

    // Scale data points
    for(int i = 0; i < M; i++) {
        Vec3f *row = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < N; j++) {
            row[j][0] = (trg_std[0] / src_std[0]) * row[j][0];
            row[j][1] = (trg_std[1] / src_std[1]) * row[j][1];
            row[j][2] = (trg_std[2] / src_std[2]) * row[j][2];
        }
    }

    // Adding mean to modified pixels
    for(int i = 0; i < M; i++) {
        Vec3f *row = (Vec3f *) output.ptr<Vec3f>(i);
        for(int j = 0; j < N; j++) {
            row[j][0] += trg_means[0];
            row[j][1] += trg_means[1];
            row[j][2] += trg_means[2];
        }
    }

    output = LAlBe_2_RGB(output);

    return output;
}

vector<float> channelMeans(const Mat &src) {
    vector<float> means;

    const int M = src.rows,
              N = src.cols,
              SIZE = M * N;

    float mean_l = 0,
          mean_a = 0,
          mean_b = 0;
    for(int i = 0; i < M; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i);
        for(int j = 0; j < N; j++) {
            mean_l += row[j][0];
            mean_a += row[j][1];
            mean_b += row[j][2];
        }
    }

    means.push_back(mean_l / SIZE);
    means.push_back(mean_a / SIZE);
    means.push_back(mean_b / SIZE);

    return means;
}

vector<float> channelStd(const Mat &src, const vector<float> &means) {
    vector<float> stds;

    const int M = src.rows,
              N = src.cols,
              SIZE = M * N;

    // Computing standard deviation
    float std_l = 0,
          std_a = 0,
          std_b = 0;

    for(int i = 0; i < M; i++) {
        Vec3f *row = (Vec3f *) src.ptr<Vec3f>(i);
        for(int j = 0; j < N; j++) {
            std_l += (row[j][0] - means[0]) * (row[j][0] - means[0]);
            std_a += (row[j][1] - means[1]) * (row[j][1] - means[1]);
            std_b += (row[j][2] - means[2]) * (row[j][2] - means[2]);
        }
    }

    stds.push_back(sqrt(std_l / SIZE));
    stds.push_back(sqrt(std_a / SIZE));
    stds.push_back(sqrt(std_b / SIZE));

    return stds;
}

#endif // UTILS_HPP
