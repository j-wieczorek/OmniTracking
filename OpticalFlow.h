#pragma once
#include <opencv2\opencv.hpp>
using namespace cv;

class OpticalFlow
{
	Mat Ir;
	Mat Itheta;
	Mat It;
	
	Mat Ir_int;
	Mat Itheta_int;
	Mat It_int;

	Mat mask;

	int R;
	int perimeter;
	int alpha;

	int img_center_x;
	int img_center_y;

	int magnitude_threshold;
	Mat cart_2_polar_matrix;
	Mat bsMask(int x, int y, int width, int height);
	void preprocessing(Mat src_image);
	
public:
	Mat u; 
	Mat v;
	
	Mat cart2PolarTransformation(Mat src_image);
	Mat transformOmniImage(Mat src_image);
	void calculate(Mat current_frame_grey, Mat current_frame, Mat previous_frame);
	Mat calculateMagnitude();

	OpticalFlow();
	OpticalFlow::OpticalFlow(int size_x, int size_y);
	~OpticalFlow();
};

