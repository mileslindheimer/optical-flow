#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv/highgui.h>

#define KERNX 5 //this is the x-size of the kernel. It will always be odd. (for convolution)
#define KERNY 5 //this is the y-size of the kernel. It will always be odd. (for convolution)

#define WIN_X 500
#define WIN_Y 500
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

void normalize_kernel( float * kernel, int size_x, int size_y) {
  int sum = 0;
  for (int i = 0; i < size_x*size_y; i++ ) {
    sum += kernel[i];
  }
  for (int i = 0; i < size_x*size_y && sum != 0; i++ ) {
    kernel[i] /= sum;
  }
}

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

int main(int argc, char** argv )
{
    // frame and hand images as opencv matrices
    Mat frame;
    // Capture from video
    VideoCapture capture("IMG_1265.mov"); 

    // Create window
    cvNamedWindow("Camera_Output", 1);    
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
    namedWindow("hand",1);

    float *in_image, *out_image;
    
    // Intinite loop until manually broken by end of video or keypress
    for(;;)
    {
        /*
            IMAGE PROCESSING 
        */

        // Get a new frame from video
        capture >> frame;
        // Adjust the resolution of the video frame, in this case using cubic interpolation
        resize(frame, frame, Size(320, 180), 0, 0, INTER_CUBIC);

        // Convert to grayscale
        cvtColor(frame, frame, CV_BGR2GRAY);

        // Convert from opencv to float arrays
        int rows = frame.rows;
        int cols = frame.cols;
        int step = frame.step;
        in_image = (float *) malloc(sizeof(float) * rows*cols);
        out_image = (float *) malloc(sizeof(float) * rows*cols);

        for(int j = 0;j < rows;j++){
            for(int i = 0;i < cols;i++){
                in_image[step * j + i] = frame.data[step * j + i];
            }
        }

        /*
            OPTICAL FLOW
        */

        // sample of linear regression on hardcoded float arrays
        int n = 6;
        float x[6]= {1, 2, 4,  5,  10, 20};
        float y[6]= {4, 6, 12, 15, 34, 68};

        // for(int j = 0;j < 6;j++){
        //     for(int i = 0;i < 6;i++){
        //         x[i] = frame.data[i];
        //     }
        //     y[j] = frame.data[j];
        // }

        float m = 0.0;
        float b = 0.0;
        float r = 0.0;

        linreg(n,x,y,&m,&b,&r);
        printf("m=%.04f b=%.04f r=%.04f\n",m,b,r);

        /*
            OUTPUT
        */

        // Write our output to xml file.
        // Ideally for now our output should be our vector f of flows that we get from
        // our optical flow
        // FileStorage fs("img.xml", FileStorage::WRITE);
        // for (int i = frame.rows*frame.cols - 1; i >= 0; i--) {
        //     fs << out_image[i];
        // }
        // fs.release();

        // display image in the "hand" window
        imshow("hand", frame);

        if(waitKey(30) >= 0) break;
    }
    return 0;
}