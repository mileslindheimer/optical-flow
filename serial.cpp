#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv/highgui.h>
#include <vector>

#define KERNX 5 //this is the x-size of the kernel. It will always be odd. (for convolution)
#define KERNY 5 //this is the y-size of the kernel. It will always be odd. (for convolution)

#define WIN_X 500
#define WIN_Y 500
using namespace cv;
using namespace std;

struct Flow {
  float vX;
  float vY;
};

static float sqr(float x) {
    return x*x;
}

int linreg(int n, const float x[], const float y[], float* m, float* b, float* r)
{
    float   sumx = 0.0;                        /* sum of x                      */
    float   sumx2 = 0.0;                       /* sum of x**2                   */
    float   sumxy = 0.0;                       /* sum of x * y                  */
    float   sumy = 0.0;                        /* sum of y                      */
    float   sumy2 = 0.0;                       /* sum of y**2                   */

   for (int i=0;i<n;i++)   
      { 
      sumx  += x[i];       
      sumx2 += sqr(x[i]);  
      sumxy += x[i] * y[i];
      sumy  += y[i];      
      sumy2 += sqr(y[i]); 
      } 

   float denom = (n * sumx2 - sqr(sumx));
   if (denom == 0) {
       // singular matrix. can't solve the problem.
       *m = 0;
       *b = 0;
       *r = 0;
       return 0;
   }

   *m = (n * sumxy  -  sumx * sumy) / denom;
   *b = (sumy * sumx2  -  sumx * sumxy) / denom;
   if (r!=NULL) {
      *r = (sumxy - sumx * sumy / n) /          /* compute correlation coeff     */
            sqrt((sumx2 - sqr(sumx)/n) *
            (sumy2 - sqr(sumy)/n));
   }

   return 1; 
}

void computeDerivatives(Mat img1, Mat img2, Mat Ix, Mat Iy, Mat It){
  Mat sobelX1, sobelY1, sobelX2, sobelY2;
  // Compute Sobel_x and Sobel_y on each image

  // first image
  Sobel(img1, sobelX1, CV_32F, 1, 0);
  Sobel(img1, sobelY1, CV_32F, 0, 1);
  // second image
  Sobel(img2, sobelX2, CV_32F, 1, 0);
  Sobel(img2, sobelY2, CV_32F, 0, 1);

  // TODO: need to combine approx derivatives computed by sobel
  // and need to compute constant It
  Ix = sobelX1 + sobelX2;
  Iy = sobelY1 + sobelY2;
  
}

vector<Flow> lucasKanade(Mat Ix, Mat Iy, Mat It){
  vector<Flow> flows;
  return flows;
}

void pyramidSetup(Mat img1, Mat img2, vector<Mat> pyr1, vector<Mat> pyr2);

vector<Flow> interpolatePLK(vector< vector<Flow> > rawFlows){
  return rawFlows.at(0);
}

int main(int argc, char** argv )
{
    // frame1 and hand images as opencv matrices
    Mat frame1, frame2;
    // Capture from video
    VideoCapture capture("IMG_1265.mov"); 

    // Create window
    cvNamedWindow("Camera_Output", 1);    
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 360);
    namedWindow("hand",1);
    
    // Intinite loop until manually broken by end of video or keypress
    for(;;)
    {
        /*
            IMAGE PROCESSING 
        */

        // first image

        // Get a new frame from video
        capture >> frame1;
        // Adjust the resolution of the video frame, in this case using cubic interpolation
        resize(frame1, frame1, Size(320, 180), 0, 0, INTER_CUBIC);

        // Convert to grayscale
        cvtColor(frame1, frame1, CV_BGR2GRAY);

        // second image

        // Get a new frame from video
        capture >> frame2;
        // Adjust the resolution of the video frame, in this case using cubic interpolation
        resize(frame2, frame2, Size(320, 180), 0, 0, INTER_CUBIC);

        // Convert to grayscale
        cvtColor(frame2, frame2, CV_BGR2GRAY);

        /*
            IMAGE DERIVATIVES
        */

        /*
            BUILD GAUSSIAN PYRAMIDS
        */

        /*
            LUCAS-KANADE (for each level in the pyramid)
        */

        /*
            INTERPOLATE PLK
        */

        /*
            OUTPUT
        */

        // display image in the "hand" window
        imshow("hand", frame1+frame2);

        if(waitKey(30) >= 0) break;
    }
    return 0;
}