#pragma once
#include <opencv2\opencv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace std;
using namespace cv;
using namespace Eigen;

class KPF
{
	int nx;
	int N;
	int I;

	double lambda_0;
	double lambda_opt;
	double eta;

	double sigma;

	int max_a;
	int max_b;

	int R;
	int perimeter;
	Mat image;	
	Mat omni_image;

	Matrix4f At;
	Matrix4f Ct;

	Vector4f e;
	VectorXf weights;
	MatrixXf samples;
	VectorXf previous_weights;
	MatrixXf previous_samples;

	Vector4f x;

	Mat pattern_histogram;
	vector<Mat> new_histograms;


	Matrix4f covarianceMatrix(MatrixXf X, int N);

	MatrixXf drawSamples();
	VectorXf initialWeights();
	Vector4f drawErrorSamples();

	Vector4f meanShift_sample(int j, double lambda);
	void meanShift(double lambda);

	Vector4f estimate();

	void perturbation();

	double Klambda(Vector4f x, double lambda);
	double reweight_sample(int i);
	void reweight();

	double kernelRadius();
	//meanShift again
	//perturbation again
	Mat calcHistogram(Mat image);
	vector<Mat> calcHistograms();
	double computeBhatcharyyaDistance(Mat query, Mat pattern);
	//reweight again

public:
	KPF();
	KPF(int radius, Mat pattern);
	~KPF();

	Vector4f compute(Mat omni_image, Mat transformed_image);
};

