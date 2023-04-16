#include <iostream>

using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main(void) {
    const string IMG_PATH = "./res/",
                 IMG_EXT = ".jpg",
                 IMG_SRC_NAME = "test1",
                 IMG_TRG_NAME = "test7",
                 IMG_SRC_FILENAME = IMG_PATH + IMG_SRC_NAME + IMG_EXT,
                 IMG_TRG_FILENAME = IMG_PATH + IMG_TRG_NAME + IMG_EXT;
    
    Mat src = imread(IMG_SRC_FILENAME),
        trg = imread(IMG_TRG_FILENAME);
    
    imshow("Source image (" + IMG_SRC_NAME + ")", src);
    imshow("Target image (" + IMG_TRG_NAME + ")", trg);

    waitKey();
}
