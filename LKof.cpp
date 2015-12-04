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

void getLucasKanadeOpticalFlow(Mat &img1, Mat &img2, Mat &u, Mat &v){

    Mat fx = get_fx(img1, img2);
    Mat fy = get_fy(img1, img2);
    Mat ft = get_ft(img1, img2);

    Mat fx2 = fx.mul(fx);
    Mat fy2 = fy.mul(fy);
    Mat fxfy = fx.mul(fy);
    Mat fxft = fx.mul(ft);
    Mat fyft = fy.mul(ft);

    Mat sumfx2 = get_Sum9_Mat(fx2);
    Mat sumfy2 = get_Sum9_Mat(fy2);
    Mat sumfxft = get_Sum9_Mat(fxft);
    Mat sumfxfy = get_Sum9_Mat(fxfy);
    Mat sumfyft = get_Sum9_Mat(fyft);

    Mat tmp = sumfx2.mul(sumfy2) - sumfxfy.mul(sumfxfy);
    u = sumfxfy.mul(sumfyft) - sumfy2.mul(sumfxft);
    v = sumfxft.mul(sumfxfy) - sumfx2.mul(sumfyft);
    divide(u, tmp, u);
    divide(v, tmp, v);

//    saveMat(u, "U");
//    saveMat(v, "V");   
}



int main(){
    // FileStorage file("u.xml", FileStorage::WRITE);
    vector<Point2f> points1;
    vector<Point2f> points2;

    // Capture from video
    VideoCapture capture(0);//("pacman.mp4"); 
    // if( !capture.isOpened()){
    //      cout << "Cannot open the video file" << endl;
    //      return -1;
    // }

    // Create window
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 360);
    namedWindow("hand",1);
    
    // Intinite loop until manually broken by end of video or keypress
    Mat frame1,frame2, img1, img2;
    Mat prevFrame;
    bool firstPass = true;
    for(;;){


        // first image
        capture >> frame1;
        resize(frame1, frame1, Size(320, 180), 0, 0, INTER_CUBIC);

        // second image
        capture >> frame2;
        resize(frame2, frame2, Size(320, 180), 0, 0, INTER_CUBIC);

        GaussianBlur(frame1, frame1, Size(9,9), 1.5, 1.5);
        GaussianBlur(frame2, frame2, Size(9,9), 1.5, 1.5);


        //frame1.convertTo(img1, CV_64FC1, 1.0/255, 0);
        //frame2.convertTo(img2, CV_64FC1, 1.0/255, 0);
        cvtColor(frame1, frame1, CV_BGR2GRAY);
        cvtColor(frame2, frame2, CV_BGR2GRAY);


        Mat diff = frame2 - frame1;

        threshold(diff, diff, 10, 255, THRESH_BINARY);

        Mat sobelX1, sobelY1, sobelX2, sobelY2;
        // Compute Sobel_x and Sobel_y on each image

        // first  
        Sobel(diff, sobelX1, CV_64FC1, 1, 1);
       // Sobel(diff, sobelY1, CV_64FC1, 0, 1); 

        //cout << diff << "\n";
        diff = sobelX1 + sobelX2;
        dilate(diff, diff, Mat(), Point(-1,-1), 2);
        erode(diff, diff, Mat(), Point(-1,-1), 2);

        Mat u = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
        Mat v = Mat::zeros(img1.rows, img1.cols, CV_64FC1);

        if (firstPass) {
            firstPass = false;
            prevFrame = diff;
            continue;
        }


        getLucasKanadeOpticalFlow(prevFrame, diff, u, v);

        prevFrame = diff;

        double avgX = 0;
        double avgY = 0;
        int counts = 0;
        for (int i = 0; i < u.rows; i++) {
            for (int j = 0; j < u.cols; j++) {

                 double x = u.data[u.step*i+j];
                 double y = v.data[v.step*i+j];

                 double dist = x * x + y * y;

                 if (x != 0 && y != 0) {
                    avgX += j;
                    avgY += i;
                    counts++;
                 }

            }
        }

        avgX /= counts;
        avgY /= counts;


        circle(frame2, Point2f(avgX - u.cols/2, avgY), 10, Scalar(0, 255, 0), 2, 8, 0);
        imshow("hand", frame2);
        if(waitKey(30) >= 0) break;

        // Mat opticalFlow = v;
        // // GaussianBlur(opticalFlow, opticalFlow, Size(5,5), 1.5, 1.5);

        // // // // // threshold flows
        // double xAvg = 0;
        // double yAvg = 0;

        // int counts = 0;
        // float FLOW_THRESHOLD = 225;
        // float FLOW_THRESHOLD1 = 250;
        // for (int i = 0; i < u.rows; i++) {
        //     for (int j = 0; j< u.cols; j++) {
        //         //cout << "hi\n";
        //         double x = u.data[u.step*i+j];
        //         double y = v.data[v.step*i+j];

        //         double flowX = x - j;
        //         double flowY = y - i;


        //         double dist = sqrt(flowX*flowX + flowY*flowY);
        //         //double disty = sqrt(y * y);
        //         if (dist > FLOW_THRESHOLD) {

        //             // cout << dist << "\n";

        //             xAvg += (int)x;
        //             yAvg += (int)y;
        //             counts++;

        //             circle(frame2, Point2f(j,i), 1, Scalar(255, 0, 0), 2, 8, 0);
        //             if (dist > FLOW_THRESHOLD1){
        //                 circle(frame2, Point2f(j,i), 1, Scalar(0, 255, 0), 2, 8, 0);
        //             }
        //         }
        //     }    
        // }


        // if (counts>0){
        //     xAvg = xAvg/counts;
        //     yAvg = yAvg/counts;
        //     circle(frame2, Point2f(yAvg,xAvg), 5, Scalar(0, 0, 255), 2, 8, 0);

        // }


        // int THRESHOLD = 1;
        // double avgx = 0;
        // double avgy = 0;
        // for (int i = 0; i < u.rows; i++) {
        //     for (int j = 0; j< u.cols; j++) {
        //         //cout << "hi\n";
        //         double x = u.data[u.step*i+j];
        //         double y = v.data[v.step*i+j];
        //         avgx += x;
        //         avgy += y;

        //         if (x < THRESHOLD) {
        //             u.data[u.step*i+j] = 0;
        //         }


        //         if (y < THRESHOLD) {
        //             v.data[v.step*i+j] = 0;
        //         }
        //     }    
        // }

        // avgx /= (u.rows * u.cols);
        // avgy /= (v.rows * v.cols);

        // // cout << avgx << "x\n";
        // // cout << avgy << "y\n";

        // if (counts > 0) {
        //     avgx = ((int)avgx/counts);
        //     avgy = ((int)avgy/counts);

        //     // cout << xAvg << ", " << yAvg << "\n";

        //     circle(frame2, Point2f(avgx,avgy), 50, Scalar(255, 0, 0), 3, 8, 0);
        // }

        
        // Write to file!
        // cout << u << "\n";

        // u.copyTo(points1);
        // v.copyTo(points2);
        // int a,b;
        // for (a = 0)
 

        // int i, k;
        // for (i = k = 0; i < points2.size(); i++) {

        //     if ((points1[i].x - points2[i].x) > 0) {
        //       line(opticalFlow, points1[i], points2[i], Scalar(0, 0, 255), 1, 1, 0);

        //       // circle(opticalFlow, points1[i], 2, Scalar(255, 0, 0), 1, 1, 0);

        //       line(opticalFlow, points1[i], points2[i], Scalar(0, 0, 255), 1, 1, 0);
        //       // circle(opticalFlow, points1[i], 1, Scalar(255, 0, 0), 1, 1, 0);
        //     } else {
        //       line(opticalFlow, points1[i], points2[i], Scalar(0, 255, 0), 1, 1, 0);

        //       // circle(opticalFlow, points1[i], 2, Scalar(255, 0, 0), 1, 1, 0);

        //       line(opticalFlow, points1[i], points2[i], Scalar(0, 255, 0), 1, 1, 0);
        //       // circle(opticalFlow, points1[i], 1, Scalar(255, 0, 0), 1, 1, 0);
        //     }
        //     points1[k++] = points1[i];

        // }


    // Mat fx = get_fx(frame1, frame2);
    // Mat fy = get_fy(frame1, frame2);
    //    imshow("hand", sobelX1 + sobelX2);
    //    if(waitKey(30) >= 0) break;

    }
    return 0;
}
