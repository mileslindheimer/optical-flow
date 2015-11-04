#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv/highgui.h>
#include <Eigen/Dense>

#define KERNX 3 //this is the x-size of the kernel. It will always be odd. (for convolution)
#define KERNY 3 //this is the y-size of the kernel. It will always be odd. (for convolution)

using namespace cv;

void convolve(float* in, float* out, int data_size_X, int data_size_Y, float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    // main convolution loop
    for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
        for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
            for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
                for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
                    // only do the operation if not out of bounds
                    if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
                        //Note that the kernel is flipped
                        out[x+y*data_size_X] += 
                                kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv )
{
    // frame and hand images as opencv matrices
    Mat frame, hand;
    // Capture from video
    VideoCapture capture("IMG_1265.mov"); 

    // Create window
    cvNamedWindow("Camera_Output", 1);    
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
    namedWindow("hand",1);

    // Intinite loop until manually broken by end of video or keypress
    for(;;)
    {
        /*
            IMAGE PROCESSING 
        */

        // Get a new frame from video
        capture >> frame;
        // Adjust the resolution of the video frame, in this case using cubic interpolation
        resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC);

        // This is a hardcoded cropping for the current test video: IMG_1265.mov
        // Rect myROI(50, 0, 420, 300);
        // frame = frame(myROI);

        // Convert to grayscale
        cvtColor(frame, frame, CV_BGR2GRAY);
        // Gaussian blur convolution to reduce noise of the image
        GaussianBlur(frame, frame, Size(5,5), 1.5, 1.5);

        // Binary threshold convolution to separate background from hand
        // Note: not great, should find a better way of getting the hand's pixels
        double thresh = 75;
        double maxValue = 255;
        threshold(frame, hand, thresh, maxValue, THRESH_BINARY);

        /*
            OPTICAL FLOW
        */
        // See OpticalFLowDemo.cpp for starting baseline code

        // General iteration thru opencv image matrix,
        // q is the current pixel
        unsigned char *input = (unsigned char*)(hand.data);
        for(int j = 0;j < hand.rows;j++){
            for(int i = 0;i < hand.cols;i++){
                unsigned char q = input[hand.step * j + i];

            }
        }

        /*
            OUTPUT
        */

        // Write our output to xml file.
        // Ideally for now our output should be our vector f of flows that we get from
        // our optical flow
        FileStorage fs("img.xml", FileStorage::WRITE);
        fs << "hand" << hand;
        fs.release();

        // display image in the "hand" window
        imshow("hand", hand);

        if(waitKey(30) >= 0) break;
    }
    return 0;
}