// Coarse to fine Lucas & Kanade Optical Flow
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
#include <omp.h>

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

/*
double get_Sum9(Mat &m, int y, int x){
    if(x < 0 || x >= m.cols) return 0;
    if(y < 0 || y >= m.rows) return 0;
    
    double val = 0.0;
    int tmp = 0;
    for(int i =  -1; i <= 1; i++){
        for(int j = -1; j <= 1; j++){
            if(isInsideImage(y + i, x + j, m)){
                ++ tmp;
                val += m.ATD(y + i, x + j);
            }
        }
    }
    if(tmp == 9) return val;
    else return m.ATD(y, x) * 9;
}
*/


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
    int nthr = 4;
    omp_set_num_threads(nthr);
    
    
    #pragma omp parallel for
    for(int i = 1; i < m.rows - 1; i++){
        for(int j = 1; j < m.cols - 1; j++){
            res.ATD(i, j) = get_Sum9(m, i, j);
        }
        //cout << "Thread count: " << omp_get_num_threads() << "\n";
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

// void matmuld(double **a, double **b, double **c, int n) {
//   for(int i=0;i<n;i++)
//     for(int j=0;j<n;j++)
//       for(int k=0;k<n;k++)
//         c[i][j] += a[i][k]*b[k][j];
// }

Mat matMul(Mat a, Mat b){
    Mat res = Mat::zeros(a.rows, a.cols, CV_64FC1);
    int nthr = 4;
    omp_set_num_threads(nthr);
    
    #pragma omp parallel for
    for (int r=0; r<a.rows; r++){
        for (int c =0; c<a.cols; c++){
            res.ATD(r,c) = a.ATD(r,c) * b.ATD(r,c);
        }
    }
    return res;
}

void do_mv(Mat a, Mat b, Mat c, int n, int i){
    for (int k=0; k<n; k++){
        for (int j=0; j<n; j++){
            c.ATD(i,j) += a.ATD(i,k) * b.ATD(k,j);
        }
    }
}

Mat matmuld(Mat a, Mat b) {
    int nthr = 4;
    int n = a.rows;
    Mat output = Mat::zeros(n, n, CV_64FC1);
    
    
    omp_set_num_threads(nthr);
    #pragma omp parallel
    {
        #pragma omp single
        {
            for(int i = 0; i<n; i++){
                #pragma omp task firstprivate(i)
                {
                    do_mv(a,b,output,n,i);
                }
            }
        }
    }
    return output;
}


// cout << "here\n";



// int n = a.rows;
// Mat output = Mat::zeros(n, n, CV_64FC1);


// for(int i=0;i<n;i++){
//   for(int j=0;j<n;j++){
//     for(int k=0;k<n;k++){
//       // c[i][j] += a[i][k]*b[k][j];
//       // cout << "entered\n";
//       output.ATD(i,j) = a.ATD(i,k) * b.ATD(k,j);
//     }
//   }
// }
// // cout << "done\n";
// return output;

Mat divideMats(Mat a, Mat b){
    int n = a.rows;
    Mat output = Mat::zeros(n, n, CV_64FC1);
    
    int nthr = 4;
    omp_set_num_threads(nthr);
    
    #pragma omp parallel for
    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            output.ATD(i,j) = a.ATD(i,j)/b.ATD(i,j);
        }
        //cout << "Thread count: " << omp_get_num_threads() << "\n";
    }
    return output;
}

/*
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
}
 */

void getLucasKanadeOpticalFlow(Mat &img1, Mat &img2, Mat &u, Mat &v){
    double time;
    double new_time;
    double elapsed;
    
    /* Start timer */
    time = timestamp();

    /* Start algorithm */
    /*
    Mat fx = get_fx(img1, img2);
    Mat fy = get_fy(img1, img2);
    Mat ft = get_ft(img1, img2);
     */
    
    Mat fx;
    Mat fy;
    Mat ft;
    
    #pragma omp parallel num_threads(3)
    {
        if(omp_get_thread_num()==0)
        {
            fx = get_fx(img1, img2);
            //cout<<"Derivatives thread 0\n ";
        }
        else if (omp_get_thread_num()==1)
        {
            fy = get_fy(img1, img2);
            //cout<<"Derivatives thread 1\n ";
        }
        else if (omp_get_thread_num()==2)
        {
            ft = get_ft(img1, img2);
            //cout<<"Derivatives thread 2\n ";
        }
    }
     
    
    new_time = timestamp();
    elapsed = new_time - time;
    cout<<"Derivatives: "<< elapsed <<" seconds\n";
    
    
    
    /* Start timer */
    time = timestamp();

    /* Start algorithm */
    
    Mat fx2;
    Mat fy2;
    Mat fxfy;
    Mat fxft;
    Mat fyft;
    #pragma omp parallel num_threads(5)
    {
        if(omp_get_thread_num()==0)
        {
            fx2 = fx.mul(fx);
        }
        else if(omp_get_thread_num()==1)
        {
            fy2 = fy.mul(fy);
        }
        else if(omp_get_thread_num()==2)
        {
            fxfy = fx.mul(fy);
        }
        else if (omp_get_thread_num()==3)
        {
            fxft = fx.mul(ft);
        }
        else if (omp_get_thread_num()==4)
        {
            fyft = fy.mul(ft);
        }
    }
    
    /*
    Mat fx2 = fx.mul(fx);
    Mat fy2 = fy.mul(fy);
    Mat fxfy = fx.mul(fy);
    Mat fxft = fx.mul(ft);
    Mat fyft = fy.mul(ft);
    */
    
    /*
    Mat fx2 = matMul(fx,fx);
    Mat fy2 = matMul(fy, fy);
    Mat fxfy = matMul(fy, fx);
    Mat fxft = matMul(fx, ft);
    Mat fyft = matMul(fy,ft);
    */
    /*
    Mat fx2 = matmuld(fx,fx);
    Mat fy2 = matmuld(fy,fy);
    Mat fxfy = matmuld(fx,fy);
    Mat fxft = matmuld(fx,ft);
    Mat fyft = matmuld(fy, ft);
     */

    new_time = timestamp();
    elapsed = new_time - time;
    cout<<"matmul: "<< elapsed <<" secs\n";
    
    
    
    
    time = timestamp();
    /* Start algorithm */
    
    Mat sumfx2 = get_Sum9_Mat(fx2);
    Mat sumfy2 = get_Sum9_Mat(fy2);
    Mat sumfxft = get_Sum9_Mat(fxft);
    Mat sumfxfy = get_Sum9_Mat(fxfy);
    Mat sumfyft = get_Sum9_Mat(fyft);
    
    /* End algorithm */
    
    new_time = timestamp();
    elapsed = new_time - time;
    printf("get_Sum9_Mat() elapsed time = %f seconds.\n", elapsed);

    /* End timer */
    
        ///**** Actual Serial implementation without OpenCV
    ///Commented out because full serial becomes too slow to output
    ///Need to optimize matmul first
    
    // Mat tmp = matmuld(sumfx2, sumfy2) - matmuld(sumfxfy, sumfxfy);
    // u = matmuld(sumfxfy, sumfyft) - matmuld(sumfy2, sumfxft);
    // v = matmuld(sumfxft, sumfxfy) - matmuld(sumfx2, sumfyft);
    // u = divideMats(u, tmp);
    // v = divideMats(v, tmp);
    
    
    
    time = timestamp();
    
    Mat tmp = sumfx2.mul(sumfy2) - sumfxfy.mul(sumfxfy);
    u = sumfxfy.mul(sumfyft) - sumfy2.mul(sumfxft);
    v = sumfxft.mul(sumfxfy) - sumfx2.mul(sumfyft);
    
    /*
    Mat tmp;
    #pragma omp parallel num_threads(3)
    {
        if(omp_get_thread_num()==0)
        {
            tmp = sumfx2.mul(sumfy2) - sumfxfy.mul(sumfxfy);
        }
        else if(omp_get_thread_num()==1)
        {
            u = sumfxfy.mul(sumfyft) - sumfy2.mul(sumfxft);
        }
        else if(omp_get_thread_num()==2)
        {
            v = sumfxft.mul(sumfxfy) - sumfx2.mul(sumfyft);
        }
    }
    */
    /* End algorithm */
    new_time = timestamp();
    elapsed = new_time - time;
    
    cout<<"Least-Squares Part 1: "<< elapsed <<" seconds\n";
    
    
    
    
    
    time = timestamp();
    
    /*
    divide(u, tmp, u);
    divide(v, tmp, v);
    */
    
    #pragma omp parallel num_threads(2)
    {
        if(omp_get_thread_num()==0)
        {
            divide(u, tmp, u);
        }
        else if(omp_get_thread_num()==1)
        {
            divide(v, tmp, v);
        }
    }
    
    /* End algorithm */
    
    new_time = timestamp();
    elapsed = new_time - time;
    cout<<"Least-Squares Part 2: "<< elapsed <<" seconds\n";
    
    
    
    
    
    //Compute time for divideMats
    /*
    
    divideMats(u, tmp);
    divideMats(v, tmp);
        cout<<"divideMats: "<< duration4 <<" seconds\n";
    */
      
}



int main(){
    Mat ori1 = imread("table1.jpg", 0);
    Mat ori2 = imread("table2.jpg", 0);
    Mat img1 = ori1(Rect(0, 0, 640, 448));
    Mat img2 = ori2(Rect(0, 0, 640, 448));
    
    img1.convertTo(img1, CV_64FC1, 1.0/255, 0);
    img2.convertTo(img2, CV_64FC1, 1.0/255, 0);
    
    Mat u2 = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
    Mat v2 = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
    
    /* Start timer */
    /*
    clock_t start;
    double duration;
    
    start = clock();
    */
    /* Start algorithm */
    double newStart = timestamp();
    getLucasKanadeOpticalFlow(img1, img2, u2, v2);
    double newEnd = timestamp();
    /* End algorithm */
    /*
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    
    cout<<"Overall Duration: "<< duration <<" seconds\n";
    */
    double newElapsed = newEnd - newStart;
    printf("elapsed time = %f seconds.\n", newElapsed);
    /* End timer */
    
    //saveMat(u2, "U2");
    //saveMat(v2, "V2");
    
    //check results for u2
    std::ifstream infile("U2Test.txt");
    std::string line;
    int row = 0;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        double a;
        int col = 0;
        while(iss >> a)
        {
            double currElem = u2.ATD(row, col);
            //currElem = roundf(currElem * 10000) / 10000;
            //a = roundf(a * 10000) / 10000;
            double thresh = 0.000001; //arbitrary
            if (abs(currElem - a) > thresh)
            {
                cout << "NOTE: The test code may not be correct.";
                cout << "Found mismatch in u2\n";
                cout << "currElem index: " << row << "," << col << "\n";
                cout << "u2 value: " << currElem << "\n";
                cout << "Correct value: " << a << "\n";
                return 0;
            }
            col++;
        }
        
        row++;
    }
    
    
    //check results for v2
    std::ifstream v2file("V2Test.txt");
    row = 0;
    while (std::getline(v2file, line))
    {
        std::istringstream iss(line);
        double a;
        int col = 0;
        while(iss >> a)
        {
            double currElem = v2.ATD(row, col);
            //currElem = roundf(currElem * 10000) / 10000;
            //a = roundf(a * 10000) / 10000;
            double thresh = 0.000001; //arbitrary
            if (abs(currElem - a) > thresh)
            {
                cout << "NOTE: The test code may not be correct.";
                cout << "Found mismatch in v2\n";
                cout << "currElem index: " << row << "," << col << "\n";
                cout << "v2 value: " << currElem << "\n";
                cout << "Correct value: " << a << "\n";
                return 0;
            }
            col++;
        }
        
        row++;
    }
    
    
    
    cout << "Success!\n";
    //    waitKey(0);
    
    return 0;
}
