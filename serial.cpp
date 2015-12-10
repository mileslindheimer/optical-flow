// Lucas & Kanade Optical Flow
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
// To run this code, you should have OpenCV in your computer.
// Have fun with it.
// 

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <pthread.h>

#include <sys/time.h>
#include <time.h>

using namespace cv;
using namespace std;

#define ATD at<double>
#define ATF at<float>
#define elif else if

#ifndef bool
    #define bool int
    #define false ((bool)0)
    #define true  ((bool)1)
#endif

double timestamp()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

Mat get_fx(Mat &src1, Mat &src2){
    Mat fx;
    Mat kernel = Mat::ones(2, 2, CV_64FC1);
    kernel.ATD(0, 0) = -1.0;
    kernel.ATD(1, 0) = -1.0;

    Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    filter2D(src2, dst2, -1, kernel);

    fx = dst1 + dst2;
    return fx;
}

Mat get_fy(Mat &src1, Mat &src2){
    Mat fy;
    Mat kernel = Mat::ones(2, 2, CV_64FC1);
    kernel.ATD(0, 0) = -1.0;
    kernel.ATD(0, 1) = -1.0;

    Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    filter2D(src2, dst2, -1, kernel);

    fy = dst1 + dst2;
    return fy;
}

Mat get_ft(Mat &src1, Mat &src2){
    Mat ft;
    Mat kernel = Mat::ones(2, 2, CV_64FC1);
    kernel = kernel.mul(-1);

    Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    kernel = kernel.mul(-1);
    filter2D(src2, dst2, -1, kernel);

    ft = dst1 + dst2;
    return ft;
}

bool isInsideImage(int y, int x, Mat &m){
    int width = m.cols;
    int height = m.rows;
    if(x >= 0 && x < width && y >= 0 && y < height) return true;
    else return false;
}

double get_Sum9(Mat &m, int y, int x){
    if(x < 0 || x >= m.cols) return 0;
    if(y < 0 || y >= m.rows) return 0;

    double val = 0.0;
    int tmp = 0;
    if(isInsideImage(y - 1, x - 1, m)){
        ++ tmp;
        val += m.ATD(y - 1, x - 1);
    }
    if(isInsideImage(y - 1, x, m)){
        ++ tmp;
        val += m.ATD(y - 1, x);
    }
    if(isInsideImage(y - 1, x + 1, m)){
        ++ tmp;
        val += m.ATD(y - 1, x + 1);
    }
    if(isInsideImage(y, x - 1, m)){
        ++ tmp;
        val += m.ATD(y, x - 1);
    }
    if(isInsideImage(y, x, m)){
        ++ tmp;
        val += m.ATD(y, x);
    }
    if(isInsideImage(y, x + 1, m)){
        ++ tmp;
        val += m.ATD(y, x + 1);
    }
    if(isInsideImage(y + 1, x - 1, m)){
        ++ tmp;
        val += m.ATD(y + 1, x - 1);
    }
    if(isInsideImage(y + 1, x, m)){
        ++ tmp;
        val += m.ATD(y + 1, x);
    }
    if(isInsideImage(y + 1, x + 1, m)){
        ++ tmp;
        val += m.ATD(y + 1, x + 1);
    }
    if(tmp == 9) return val;
    else return m.ATD(y, x) * 9;
}

Mat get_Sum9_Mat(Mat &m){
    Mat res = Mat::zeros(m.rows, m.cols, CV_64FC1);
    for(int i = 1; i < m.rows - 1; i++){
        for(int j = 1; j < m.cols - 1; j++){
            res.ATD(i, j) = get_Sum9(m, i, j);
        }
    }
    return res;
}

void saveMat(Mat &M, string s){
    s += ".txt";
    FILE *pOut = fopen(s.c_str(), "w+");
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            fprintf(pOut, "%lf", M.ATD(i, j));
            if(j == M.cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}


Mat matmuld(Mat a, Mat b) {
    int n = a.rows;
    Mat output = Mat::zeros(n, n, CV_64FC1);

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            output.ATD(i,j) = a.ATD(i,j)*b.ATD(i,j);
        }
    }
    return output;
}

Mat divideMats(Mat a, Mat b){
    int n = a.rows;
    Mat output = Mat::zeros(n, n, CV_64FC1);

    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            output.ATD(i,j) = a.ATD(i,j)/b.ATD(i,j);
        }
    }
    return output;
}

void getLucasKanadeOpticalFlow(Mat &img1, Mat &img2, Mat &u, Mat &v){

    double time;
    double new_time;
    double elapsed;
    
    
    /* Start timer */
    time = timestamp();
    
    Mat fx = get_fx(img1, img2);
    Mat fy = get_fy(img1, img2);
    Mat ft = get_ft(img1, img2);
    
    
    new_time = timestamp();
    elapsed = new_time - time;
    cout<<"Derivatives: "<< elapsed <<" seconds\n";

    /* Start timer */
    time = timestamp();

    /* Start algorithm */

    Mat fx2 = fx.mul(fx);
    Mat fy2 = fy.mul(fy);
    Mat fxfy = fx.mul(fy);
    Mat fxft = fx.mul(ft);
    Mat fyft = fy.mul(ft);
    

    // Mat fx2 = matmuld(fx,fx);
    // Mat fy2 = matmuld(fy,fy);
    // Mat fxfy = matmuld(fx,fy);
    // Mat fxft = matmuld(fx,ft);
    // Mat fyft = matmuld(fy, ft);

    /* End algorithm */
    //duration1 = ( clock() - start1 ) / (double) CLOCKS_PER_SEC;

    //cout<<"Matmul: "<< duration1*1000 <<" milliseconds\n";
    /* End timer */

   

    /* Start algorithm */

    Mat sumfx2 = get_Sum9_Mat(fx2);
    Mat sumfy2 = get_Sum9_Mat(fy2);
    Mat sumfxft = get_Sum9_Mat(fxft);
    Mat sumfxfy = get_Sum9_Mat(fxfy);
    Mat sumfyft = get_Sum9_Mat(fyft);




    

    // Mat tmp = matmuld(sumfx2, sumfy2) - matmuld(sumfxfy, sumfxfy);
    // u = matmuld(sumfxfy, sumfyft) - matmuld(sumfy2, sumfxft);
    // v = matmuld(sumfxft, sumfxfy) - matmuld(sumfx2, sumfyft);
    // u = divideMats(u, tmp);
    // v = divideMats(v, tmp);



    Mat tmp = sumfx2.mul(sumfy2) - sumfxfy.mul(sumfxfy);
    u = sumfxfy.mul(sumfyft) - sumfy2.mul(sumfxft);
    v = sumfxft.mul(sumfxfy) - sumfx2.mul(sumfyft);
    divide(u, tmp, u);
    divide(v, tmp, v);

    new_time = timestamp();
    elapsed = new_time - time;
    cout<<"Least-Squares: "<< elapsed <<" seconds\n";



    /* End timer */

//    saveMat(u, "U");
//    saveMat(v, "V");   
}

vector<Mat> getGaussianPyramid(Mat &img, int nLevels){
    vector<Mat> pyr;
    pyr.push_back(img);
    for(int i = 0; i < nLevels - 1; i++){
        Mat tmp;
        pyrDown(pyr[pyr.size() - 1], tmp);
        pyr.push_back(tmp);
    }
    return pyr;
}

void coarseToFineEstimation(Mat &img1, Mat &img2, Mat &u, Mat &v, int nLevels){

    vector<Mat> pyr1 = getGaussianPyramid(img1, nLevels);
    vector<Mat> pyr2 = getGaussianPyramid(img2, nLevels);
    Mat upu, upv;
    
    
    
    double totalElapsed = 0;
    for(int i = nLevels - 1; i >= 0; i--){

        Mat tmpu = Mat::zeros(pyr1[i].rows, pyr1[i].cols, CV_64FC1);
        Mat tmpv = Mat::zeros(pyr2[i].rows, pyr2[i].cols, CV_64FC1);
        
        
        
        
        double newStart = timestamp();
        getLucasKanadeOpticalFlow(pyr1[i], pyr2[i], tmpu, tmpv);
        double newEnd = timestamp();
        double newElapsed = newEnd - newStart;
        printf("one LK = %f seconds.\n", newElapsed);
        totalElapsed += newElapsed;
        
        
        
        
        
        if(i != nLevels - 1){
            tmpu += upu;
            tmpv += upv;
        }
        if(i == 0){
            u = tmpu;
            v = tmpv;
            
            
            
            printf("total LK = %f seconds.\n", totalElapsed);
            
            return;
        }
        pyrUp(tmpu, upu);
        pyrUp(tmpv, upv);

        Mat map1(upu.size(), CV_32FC2);
        Mat map2(upu.size(), CV_32FC2);
        for (int y = 0; y < map1.rows; ++y){
            for (int x = 0; x < map1.cols; ++x){
                Point2f f = Point2f((float)(upu.ATD(y, x)), (float)(upv.ATD(y, x)));
                map1.at<Point2f>(y, x) = Point2f(x + f.x / 2, y + f.y / 2);
                map2.at<Point2f>(y, x) = Point2f(x - f.x / 2, y - f.y / 2);
            }
        }
        Mat warped1, warped2;
        remap(pyr1[i - 1], warped1, map1, cv::Mat(), INTER_LINEAR);
        remap(pyr2[i - 1], warped2, map2, cv::Mat(), INTER_LINEAR);
        warped1.copyTo(pyr1[i - 1]);
        warped2.copyTo(pyr2[i - 1]);
    }
    
    
    
    //printf("total LK\n");
    
}

int getMaxLayer(Mat &img){
    int width = img.cols;
    int height = img.rows;
    int res = 1;
    int p = 1;
    while(1){
        int tmp = pow(2.0, p);
        if(width % tmp == 0) ++ p;
        else break;
    }
    res = p;
    p = 1;
    while(1){
        int tmp = pow(2.0, p);
        if(height % tmp == 0) ++ p;
        else break;
    }
    res = res < p ? res : p;
    return res;
}

#define DIFF_THRESH 10
#define LEARNING_RATE 0.3

int main(){
    double newStart;
    double newEnd;
    double newElapsed;
    
    vector<Point2f> points1;
    vector<Point2f> points2;

    // Capture from video
    VideoCapture capture(0);

    // Create window
    //capture.set(CV_CAP_PROP_FRAME_WIDTH, 800);
    //capture.set(CV_CAP_PROP_FRAME_WIDTH, 800);
    
    //capture.set(CV_CAP_PROP_FRAME_WIDTH, 800);
    //capture.set(CV_CAP_PROP_FRAME_WIDTH, 800);
    
    
    //capture.set(CV_CAP_PROP_FRAME_WIDTH, 400);
    //capture.set(CV_CAP_PROP_FRAME_WIDTH, 400);
    
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 200);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 200);
    
    
    namedWindow("hand",1);
    
    // Intinite loop until manually broken by end of video or keypress
    Mat frame, current_frame, img1, img2;
    Mat prevFrame, prevDiff;
    bool firstPassFrame = true;
    bool firstPassDiff = true;
    for(;;){

        
        
        double totalStart = timestamp();
        
        
        
        newStart = timestamp();
        
        
        // first image
        capture >> frame;
        
        
        
        
        //resize(frame, current_frame, Size(1080, 1080), 0, 0, INTER_CUBIC);
        
        
        //resize(frame, current_frame, Size(800, 800), 0, 0, INTER_CUBIC);
        
        //resize(frame, current_frame, Size(400, 400), 0, 0, INTER_CUBIC);
        
        resize(frame, current_frame, Size(200, 200), 0, 0, INTER_CUBIC);
        
        
        
        GaussianBlur(current_frame, current_frame, Size(9,9), 1.5, 1.5); 
        cvtColor(current_frame, current_frame, CV_BGR2GRAY);

        if (firstPassFrame) {
            firstPassFrame = false;
            prevFrame = current_frame;
            continue;
        }


        Mat diff = current_frame - LEARNING_RATE * prevFrame;
        prevFrame = current_frame; 

        threshold(diff, diff, DIFF_THRESH, 255, THRESH_TOZERO);

        Mat sobelX1, sobelY1;

        // first  
        Sobel(diff, sobelX1, CV_64FC1, 1, 0);
        Sobel(diff, sobelY1, CV_64FC1, 0, 1);

        diff = sobelX1 + sobelY1;
        dilate(diff, diff, Mat(), Point(-1,-1), 2);
        erode(diff, diff, Mat(), Point(-1,-1), 2);


        if (firstPassDiff) {
            firstPassDiff = false;
            prevDiff = diff;
            continue;
        }

        Mat u = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
        Mat v = Mat::zeros(img1.rows, img1.cols, CV_64FC1);

        
        newEnd = timestamp();
        newElapsed = newEnd - newStart;;
        printf("preprocessing elapsed time = %f seconds.\n", newElapsed);
        
        /* Start timer */
        /*
        clock_t start;
        double duration;

        start = clock();
         */
        /* Start algorithm */
        // getLucasKanadeOpticalFlow(prevDiff, diff, u, v);

        newStart = timestamp();
        int maxLayer = getMaxLayer(prevDiff);


        //int maxLayer = getMaxLayer(prevDiff);
        //int maxLayer = 4;

        coarseToFineEstimation(prevDiff, diff, u, v, maxLayer);
        newEnd = timestamp();
        /* End algorithm */


        newElapsed = newEnd - newStart;
        printf("total elapsed time (pyramid) = %f seconds.\n", newElapsed);


        /* End timer */

        
        
        
        
        newStart = timestamp();
        
        
        
        
        
        prevDiff = diff;

        Mat mag = u;

        double avgX = 0;
        double avgY = 0;
        int counts = 0;
        for (int i = 0; i < u.rows; i++) {
            for (int j = 0; j < u.cols; j++) {


                double xFlow = u.ATD(i,j);
                double yFlow = v.ATD(i,j);

                double val = sqrt(xFlow * xFlow + yFlow * yFlow);

                if (val < 20) {
                    val = 0;
                } else {
                    avgX += j;
                    avgY += i;
                    counts++;
                    // circle(frame, Point2f(avgX, avgY), 1, Scalar(255, 0, 0), 2, 8, 0);


                }

                mag.ATD(i,j) = val;

            }
        }

        normalize(mag, mag, 255);

        int radius = 35;
        avgX /= counts;
        avgY /= counts;

        // rescale
        float scale = (frame.cols/current_frame.cols);
        avgX *= scale;
        avgY *= scale;

        if (counts > 500) {
            circle(frame, Point2f(avgX, avgY), radius, Scalar(0, 0, 255), 2, 8, 0);
        }
        
        newEnd = timestamp();
        newElapsed = newEnd - newStart;;
        printf("postprocessing elapsed time = %f seconds.\n", newElapsed);
        
        
        
        
        
        newStart = timestamp();
        imshow("hand", frame);
        newEnd = timestamp();
        newElapsed = newEnd - newStart;
        printf("imshow elapsed time = %f seconds.\n", newElapsed);
        
        
        
        if(waitKey(30) >= 0) break;
        
        
        
        double totalEnd = timestamp();
        double totalElapsed = totalEnd - totalStart;
        printf("total elapsed time of for loop = %f seconds.\n", totalElapsed);

    }
    return 0;
}
