# optical-flow
Hand Gesture Recognition using Feature Detection and Optical Flow Analysis

There are 2 main algorithms people use for optical flow, Lucas-Kanade and Horn-Schunck. Most implementations appear to use an adaptation of the L-K method called Pyramidal L-K, which runs on different resolutions of the same image. Because L-K works best when the differences between frames are not more than a few pixels, which won't always be the case, especially with something like hand gesture recognition - hence the Pyramidal part of the algorithm, shown below. We will implement this algorithm for our optical flow analysis.

Pyramidal Lucas-Kanade Method:
<img src="http://cdn.iopscience.com/images/0957-0233/24/5/055602/Full/mst449341f3_online.jpg" />

Our biggest immediate challenge is that the Pyramidal L-K algorithm depends on the difference between frames, and we'll need to track this difference through feature detection. The algorithms for feature detection range greatly, so we'll try to go with the simplest one that still works for us. Hopefully, something as simple as color calibration and background subtraction, hopefully not something as complicated as SURF (https://en.wikipedia.org/wiki/Speeded_up_robust_features).

Research Links:

Blob Detection: http://www.cs.umd.edu/~djacobs/CMSC426/Blob.pdf

Illustration of stages for detecting hand: http://vipulsharma20.blogspot.com/2015/03/gesture-recognition-using-opencv-python.html

Optical flow in OpenCV:
http://robots.stanford.edu/cs223b05/notes/CS%20223-B%20T1%20stavens_opencv_optical_flow.pdf
