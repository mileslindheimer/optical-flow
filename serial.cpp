#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv/highgui.h>
#include <vector>

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

vector<Mat> pyramidSetup(Mat img, int numLevels){
  vector<Mat> pyramid;
  return pyramid;
}

void computeDerivatives(Mat img1, Mat img2, Mat* Ix, Mat* Iy, Mat* It){
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
  *Ix = sobelX1 + sobelX2;
  *Iy = sobelY1 + sobelY2;
  
}

vector<Flow> lucasKanade(Mat* Ix, Mat* Iy, Mat* It, int windowSize){
  vector<Flow> flows;
  return flows;
}

vector<Flow> interpolatePLK(vector< vector<Flow> > rawFlows, int numLevels){
  Flow dummyFlow;
  dummyFlow.vX = 1; dummyFlow.vY = 2;
  vector<Flow> dummyFlows;
  dummyFlows.push_back(dummyFlow);
  return dummyFlows;
}

int main(int argc, char** argv){
    // frame1 and hand images as opencv matrices
    Mat frame1, frame2;

    int numLevels = 4;
    vector<Mat> pyr1(numLevels);
    vector<Mat> pyr2(numLevels);
    
    Mat* Ix = NULL;
    Mat* Iy = NULL;
    Mat* It = NULL;
    int windowSize = 16;

    vector< vector<Flow> > rawFlows(numLevels);
    vector<Flow> finalFlows; 

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
        capture >> frame1;
        resize(frame1, frame1, Size(320, 180), 0, 0, INTER_CUBIC);
        cvtColor(frame1, frame1, CV_BGR2GRAY);

        // second image
        capture >> frame2;
        resize(frame2, frame2, Size(320, 180), 0, 0, INTER_CUBIC);
        cvtColor(frame2, frame2, CV_BGR2GRAY);
        /*
            BUILD GAUSSIAN PYRAMIDS
        */
        pyr1 = pyramidSetup(frame1, numLevels);
        pyr2 = pyramidSetup(frame2, numLevels);

        
        Mat pyrImg1, pyrImg2;
        vector<Flow> flows;
        for(int i=0; i<numLevels; i++){
          /*
            IMAGE DERIVATIVES
          */

          pyrImg1 = pyr1.at(i);
          pyrImg2 = pyr2.at(i);
          computeDerivatives(pyrImg1, pyrImg2, Ix, Iy, It);
          /*
            LUCAS-KANADE (for each level in the pyramid)
          */
          flows = lucasKanade(Ix, Iy, It, windowSize);
          rawFlows.push_back(flows);
        }

        /*
            INTERPOLATE PLK
        */
        finalFlows = interpolatePLK(rawFlows, numLevels);
        /*
            OUTPUT
        */
        // TODO: draw flow field onto output frame

        imshow("hand", frame1);

        // press the spacekey to stop
        if(waitKey(30) >= 0) break;
    }
    return 0;
}