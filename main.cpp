/*------------------------------------------------------------------------------
Example code that shows the use of the 'cam2world" and 'world2cam" functions
Shows also how to undistort images into perspective or panoramic images

NOTE, IF YOU WANT TO SPEED UP THE REMAP FUNCTION I STRONGLY RECOMMEND TO INSTALL
INTELL IPP LIBRARIES ( http://software.intel.com/en-us/intel-ipp/ )
YOU JUST NEED TO INSTALL IT AND INCLUDE ipp.h IN YOUR PROGRAM

Copyright (C) 2009 DAVIDE SCARAMUZZA, ETH Zurich
Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org
------------------------------------------------------------------------------*/

#include "ocam_functions.h"
//#include "omnivision_tracking.h"
#include "OpticalFlow.h"
#include "KPF.h"
#include <math.h>

vector<int> Rs;
vector<int> thetas;
int point_counter = 0;
Rect area;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (point_counter < 1)
		{
			int R = sqrt((520 - y)*(520 - y) + (x - 960)*(x - 960));
			int theta = fastAtan2(520 - y, x - 960);
			Rs.push_back(R);
			thetas.push_back(theta);
			point_counter++;
			cout << "Point_counter: " << point_counter << endl;
		}
		else
		{
			int R = sqrt((520 - y)*(520 - y) + (x - 960)*(x - 960));
			int theta = fastAtan2(520 - y, x - 960);
			Rs.push_back(R);
			thetas.push_back(theta);
			int min_R = 0, max_R = 0, min_theta = 0, max_theta = 0;
			if (Rs.at(0) < Rs.at(1))
			{
				min_R = Rs.at(0);
				max_R = Rs.at(1);
			}
			else
			{
				min_R = Rs.at(1);
				max_R = Rs.at(0);
			}
			if (thetas.at(0) < thetas.at(1))
			{
				min_theta = thetas.at(0);
				max_theta = thetas.at(1);
			}
			else
			{
				min_theta = thetas.at(1);
				max_theta = thetas.at(0);
			}
			area = Rect(Point(min_theta, min_R), Point(max_theta, max_R));
			point_counter++;
			destroyWindow("Click in two corners of object to track");
		}

	}
}
int main(int argc, char *argv[])
{
	struct ocam_model  o_cata; // our ocam_models for the fisheye and catadioptric cameras
	get_ocam_model(&o_cata, "./calib_results_catadioptric.txt"); //Read the parameters of the omnidirectional camera from the TXT file 

	CvSize size_pan_image = cvSize(1620, 452);        // size of the undistorted panoramic image
	Mat dst_pan = Mat(size_pan_image, 8, 3);
	Mat mapx_pan(size_pan_image.height, size_pan_image.width, CV_32FC1);
	Mat mapy_pan(size_pan_image.height, size_pan_image.width, CV_32FC1);
	char key;
	Mat src2;

	//VideoCapture cap("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
	
	VideoCapture cap("video2.avi");
	if (!cap.isOpened()) //check if we succeeded
	{
		cerr << "Fail to open camera " << endl;
		return -1;
	}
	cap >> src2;

	namedWindow("Click in two corners of object to track");
	setMouseCallback("Click in two corners of object to track", mouseCallback,NULL);
	imshow("Click in two corners of object to track", src2);
	waitKey(0);

	Mat current_frame = Mat::zeros(src2.rows / 2, 360, CV_64FC1), previous_frame = Mat::zeros(src2.rows / 2, 360, CV_64FC1);
	OpticalFlow omni_optical_flow(src2.cols, src2.rows);
	Mat image_transformed_KPF = omni_optical_flow.transformOmniImage(src2);
	KPF kernel_particle_filter(src2.rows / 2, image_transformed_KPF(area));
	Mat Ir_show, Itheta_show;
	for (;;)
	{

		cap >> src2;     // source image 2
		// undistorted panoramic image

		/* --------------------------------------------------------------------  */
		float Rmax = 470;  // the maximum radius of the region you would like to undistort into a panorama
		float Rmin = 20;   // the minimum radius of the region you would like to undistort into a panorama
		//create_panoramic_undistortion_LUT(mapx_pan, mapy_pan, Rmin, Rmax, o_cata.xc, o_cata.yc);

		//remap(src2, dst_pan, mapx_pan, mapy_pan, CV_INTER_LINEAR, 0, 0);


		Mat image_transformed = omni_optical_flow.transformOmniImage(src2);
		//Mat image_transformed_gray;

		//cvtColor(image_transformed, image_transformed_gray, CV_BGR2GRAY);
		//if (current_frame.cols == 0)
		//{
		//	image_transformed_gray.convertTo(previous_frame, CV_64F);
		//}
	//	else
		//{
		//	previous_frame = current_frame.clone();
		//}

		//image_transformed_gray.convertTo(current_frame, CV_64F);

		//omni_optical_flow.calculate(image_transformed_gray, current_frame, previous_frame);

		//Mat magnitude_thresholded = omni_optical_flow.calculateMagnitude();
		Vector4f x_k = kernel_particle_filter.compute(src2, image_transformed);
		int x = x_k(1)*cos(x_k(0)) + src2.cols / 2;
		int y = x_k(1)*sin(x_k(0)) + src2.rows / 2;
		ellipse(src2, Point(x, y), Size(x_k(2), x_k(3)), 0.0, 0.0, 360.0, Scalar(0, 0, 255));
		imshow("Image", src2);
		waitKey(0);
		//imshow("Image", src2);
		//imshow("Magnitude thresholded", magnitude_thresholded);

		//Mat samples = drawSamples(src2.rows / 2);
		//Mat At = samples.clone();

		//imshow( "Undistorted Panoramic Image", dst_pan );
		//key = (char)waitKey(10);

//		if (key == 27)
//			break;
//		if (key == 'p')
/*		{
			imwrite("/home/ubuntu/Desktop/zdjecie_panorama.jpg", dst_pan);
			imwrite("/home/ubuntu/Desktop/zdjecie_lustra.jpg", src2);
		}
*/
		//waitKey(1);

	}
	src2.release();

	dst_pan.release();

	mapx_pan.release();
	mapy_pan.release();

	return 0;
}
