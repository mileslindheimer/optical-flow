# optical-flow
Hand Gesture Recognition using Feature Detection and Optical Flow Analysis

Basically there are 2 main algorithms people use for optical flow, Lucas-Kanade (https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method) and Horn-Schunck (https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method). So far the only benefit I've seen from Horn-Schunck is solving the aperture problem - when the system is unable to determine the flow of movement because it comes in and out of frame. 

Most implementations appear to use an adaptation of the L-K method called Pyramidal L-K, which runs on different resolutions of the same image. This is because L-K works best when the differences between frames are not more than a few pixels, which won't always be the case, especially with something like handgesture recognition - hence the Pyramidal part of the algorithm, shown below.

Pyramidal Lucas-Kanade Method:
<img src="http://cdn.iopscience.com/images/0957-0233/24/5/055602/Full/mst449341f3_online.jpg" />

The PK-L method shouldn't be too hard to implement, but the main problem is that it depends on the difference between frames, and we'll need to track this difference through feature detection… which is a whole other beast. The algorithms for feature detection range greatly, so we'll try to go with the simplest one that still works for us. Hopefully, something as simple as color calibration and background subtraction, hopefully not something as complicated as SURF (https://en.wikipedia.org/wiki/Speeded_up_robust_features). I didn't take AI or Machine Learning, so maybe you guys might have some better ideas for recognizing the hand or training or something.

As for the actual recognition of gestures, as opposed to just tracking a hand, we'll probably need something like HMMs or something of the like, but we should save this part for last since I think we'll be fine if we get the feature detection and optical flow parts working well.

As for frameworks, we'll use OpenCV (http://opencv.org/) to handle all the image/video retrieval and some basic preprocessing to save us some work, since this is just our minimal viable product. For the actual feature detection and optical flow analysis (and any other places we need linear algebra more complex than a dot product or vector addition), we'll use Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page). 

Later today I'll try to write an outline for the steps in optical flow, as well as feature detection and the overall project

That's all I got for now, feel free to post whatever helpful info you find up here! 

Blob Detection: http://www.cs.umd.edu/~djacobs/CMSC426/Blob.pdf
