#include "OpticalFlow.h"


OpticalFlow::OpticalFlow()
{
	R = 240;
	perimeter = 360;
	alpha = 500;
	magnitude_threshold = 1;

	img_center_x = 320;
	img_center_y = 240;
	
	u = Mat::zeros(R, perimeter, CV_64FC1);
	v = Mat::zeros(R, perimeter, CV_64FC1);

	Ir = Mat::zeros(R, perimeter, CV_64FC1);
	Itheta = Mat::zeros(R, perimeter, CV_64FC1);
	It = Mat::zeros(R, perimeter, CV_64FC1);

	mask = bsMask(320, 280, 640, 480);

	cart_2_polar_matrix = Mat(R, 360, CV_16UC2);
	for (int theta = 0; theta < cart_2_polar_matrix.cols; theta++)
	{
		for (int r = 0; r<cart_2_polar_matrix.rows; r++)
		{
			Vec2s point_pixel = Vec2s(r*cos(CV_PI*theta / 180) + img_center_x, r*sin(CV_PI*theta / 180) + img_center_y);
			cart_2_polar_matrix.at<Vec2s>(r, theta) = point_pixel;
		}
	}
}

OpticalFlow::OpticalFlow(int size_x, int size_y)
{
	R = size_y/2;
	perimeter = 360;
	alpha = 500;
	magnitude_threshold = 1;

	img_center_x = size_x / 2;
	img_center_y = size_y / 2;

	u = Mat::zeros(R, perimeter, CV_64FC1);
	v = Mat::zeros(R, perimeter, CV_64FC1);

	Ir = Mat::zeros(R, perimeter, CV_64FC1);
	Itheta = Mat::zeros(R, perimeter, CV_64FC1);
	It = Mat::zeros(R, perimeter, CV_64FC1);

	mask = bsMask(img_center_x, img_center_y, size_x, size_y);

	cart_2_polar_matrix = Mat(R, 360, CV_16UC2);

	for (int theta = 0; theta < cart_2_polar_matrix.cols; theta++)
	{
		for (int r = 0; r<cart_2_polar_matrix.rows; r++)
		{
			Vec2s point_pixel = Vec2s(r*cos(CV_PI*theta / 180) + img_center_x, r*sin(CV_PI*theta / 180) + img_center_y);
			cart_2_polar_matrix.at<Vec2s>(r, theta) = point_pixel;
		}
	}
}
OpticalFlow::~OpticalFlow()
{
}


//public functions

Mat OpticalFlow::transformOmniImage(Mat src_image)
{
	Mat image_cropped = mask.mul(src_image);
	Mat image_transformed = cart2PolarTransformation(image_cropped);
	//cvtColor(image_transformed, image_transformed, CV_BGR2GRAY);
	return image_transformed;
}
void OpticalFlow::calculate(Mat current_frame_grey, Mat current_frame, Mat previous_frame)
{
	preprocessing(current_frame_grey);

	It_int = current_frame - previous_frame;

	Itheta_int.convertTo(Itheta, CV_64FC1);
	Ir_int.convertTo(Ir, CV_64FC1);
	It_int.convertTo(It, CV_64FC1);

	Mat kernel_u = (Mat_<double>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);
	Mat kernel_v = (Mat_<double>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0);
	Mat u_filtered, v_filtered;
	
	filter2D(u, u, CV_64F, kernel_u);
	filter2D(v, v, CV_64F, kernel_v);
	
	u = 1 / 8 * u;
	v = 1 / 4 * v;
	
	for (int r = 0; r < u.rows; r++)
	{
		for (int theta = 0; theta < u.cols; theta++)
		{
			double u_val = u.at<double>(r, theta);
			double v_val = v.at<double>(r, theta);

			double Ir_val = Ir.at<double>(r, theta);
			double It_val = It.at<double>(r, theta);
			double Itheta_val = Itheta.at<double>(r, theta);

			u_val = u_val - Ir_val * (Ir_val * u_val + Itheta_val*v_val + It_val) / (alpha*alpha + Ir_val*Ir_val + It_val*It_val);
			v_val = v_val - Itheta_val * (Ir_val * u_val + Itheta_val*v_val + It_val) / (alpha*alpha + Ir_val*Ir_val + It_val*It_val);

			u.at<double>(r, theta) = u_val;
			v.at<double>(r, theta) = v_val;
		}

	}

}

Mat OpticalFlow::calculateMagnitude()
{
	Mat magnitude = u.mul(u) + v.mul(v);
	Mat magnitude_thresholded;
	sqrt(magnitude, magnitude);

	magnitude *= 255;
	magnitude.convertTo(magnitude, CV_8UC1);

	threshold(magnitude, magnitude_thresholded, magnitude_threshold, 255, THRESH_BINARY);

	return magnitude_thresholded;
}

//private functions

Mat OpticalFlow::bsMask(int x, int y, int width, int height)
{
	Mat mask = Mat(height, width, CV_8UC3);
	for (int i = 0; i<mask.rows; i++)
	{
		for (int j = 0; j<mask.cols; j++)
		{
			if (((i - y)*(i - y) + (j - x)*(j - x))>R*R)
			{
				mask.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else
			{
				mask.at<Vec3b>(i, j) = Vec3b(1, 1, 1);
			}
		}
	}
	return mask;
}

Mat OpticalFlow::cart2PolarTransformation(Mat src_image)
{
	Mat image_transformed = Mat::zeros(cart_2_polar_matrix.rows, cart_2_polar_matrix.cols, CV_8UC3);

	for (int y = 0; y<image_transformed.rows; y++)
	{
		for (int x = 0; x<image_transformed.cols; x++)
		{
			Vec2s pixel_coordinates; //vector containing coordinates (x,y) on the source_image of pixel to insert into image_transformed
			pixel_coordinates = cart_2_polar_matrix.at<Vec2s>(y, x);
			image_transformed.at<Vec3b>(y, x) = src_image.at<Vec3b>(pixel_coordinates[1], pixel_coordinates[0]);
		}
	}
	return image_transformed;

}
void OpticalFlow::preprocessing(Mat src_image)
{
	GaussianBlur(src_image, src_image, Size(3, 3), 1.0);
	Sobel(src_image, Itheta_int, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(src_image, Ir_int, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
}