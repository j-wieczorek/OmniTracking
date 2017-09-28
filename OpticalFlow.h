#pragma once
#include <opencv2\opencv.hpp>
using namespace cv;

class OpticalFlow
{
	Mat Ir;		///gradient image in r-axis (uchar)
	Mat Itheta;	///gradient image in angle axis (uchar)
	Mat It;		///gradient image with respect to time (uchar)
	
	Mat Ir_int;	///gradient image in r-axis (int)
	Mat Itheta_int;	///gradient image in angle axis (int)
	Mat It_int;	///gradient image with respect to time (int)

	Mat mask;	///binary mask enabling to select only circular image

	int R;		///radius of omni image in pixels
	int perimeter;	///perimeter of omni image in pixels
	int alpha;	///hyperparameter chosen in article as alpha=500	

	int img_center_x;	///center of image - x
	int img_center_y;	///center of image - y

	int magnitude_threshold;	///threshold above which change in optical flow is taken as movemnent
	Mat cart_2_polar_matrix;	///3-channel matrix in which every element M_{ij} contains coordinates of pixel with r = i and theta = j on omni image (3rd channel not used)
	Mat bsMask(int x, int y, int width, int height);	///method for computing the bsMask
	void preprocessing(Mat src_image);		///initial operations on image
	
public:
	Mat u;	///matrix containing values of optical flow in x axis
	Mat v;	///matrix containing values of optical flow in y axis
	
	Mat cart2PolarTransformation(Mat src_image);	///method for computation of cart_2_polar_matrix
	Mat transformOmniImage(Mat src_image);			///omni image transformed to coordinates r and theta
	void calculate(Mat current_frame_grey, Mat current_frame, Mat previous_frame);	///method for calculation of optical flow
	Mat calculateMagnitude();		///method for calculation of magnitude of optical flow

	OpticalFlow();
	OpticalFlow::OpticalFlow(int size_x, int size_y);
	~OpticalFlow();
};

