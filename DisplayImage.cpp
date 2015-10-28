#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv/highgui.h>

using namespace cv;

int main(int argc, char** argv )
{
    cvNamedWindow("Camera_Output", 1);    //Create window
    VideoCapture capture("IMG_1265.mov");  //Capture from video
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
    Mat hand;
    namedWindow("hand",1);
    for(;;)
    {
        Mat frame;
        capture >> frame; // get a new frame from video
        resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC);
        Rect myROI(50, 0, 420, 300);
        frame = frame(myROI);
        // cvtColor(frame, hand, CV_BGR2GRAY);
        GaussianBlur(frame, hand, Size(5,5), 1.5, 1.5);
        double thresh = 120;
        double maxValue = 255;
        threshold(hand,hand, thresh, maxValue, THRESH_BINARY);

        unsigned char *input = (unsigned char*)(hand.data);
        for(int j = 0;j < hand.rows;j++){
            for(int i = 0;i < hand.cols;i++){
                // unsigned char b = input[hand.step * j + i ] ;
                // unsigned char g = input[hand.step * j + i + 1];
                unsigned char r = input[hand.step * j + i + 2];

            }
        }
        
        FileStorage fs("img.xml", FileStorage::WRITE);
        fs << "hand" << hand;
        fs.release();

        // display image in the "hand" window
        imshow("hand", hand);

        if(waitKey(30) >= 0) break;
    }
    return 0;
}