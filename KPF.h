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
	/*!
		\Brief Class implementing Kernel Particle Filter algorithm. Numbers of equations in brackets
		refer to article 'Tracking unknown moving targets on omnidirectional vision' by 
		Yang Shu-Ying, Ge WeiMin and Zhang Cheng
	*/

	int nx; ///size of sample vector
	int N; ///number of particles
	int I; ///number of iterations for one position

	double lambda_opt; ///kernel width (10)
	double lambda_0;	///0.5 * lambda_opt
	double eta;			///hyperparameter chosen (in article) emprically to be 0.8 

	double sigma;	///parameter of Gaussian function for (21)

	int max_a;		///maximum value of semi-axis
	int max_b;		///maximum value of 2nd semi-axis

	int R;			///radius of circle on omnidirectional image
	int perimeter;	///perimeter of circle on omnidrectional image

	Mat image;		///image from camera
	Mat omni_image;	///unused, to be deleted

	Matrix4f Ct;	///covariance matrix
	Matrix4f At;	///Cholesky decomposition of covariance matrix (Table 1, step 3)
	

	Vector4f e;			///error vector drawn to add perturbation
	VectorXf weights;	///vector of weights
	MatrixXf samples;	///matrix of particles (MxN)
	VectorXf previous_weights;	///vector of previous weights (w_{t-1})
	MatrixXf previous_samples;	///matrix of previous particles (s_{t-1})

	Vector4f x;			///estimated position of object being tracked

	Mat pattern_histogram;	///histogram of object to track
	vector<Mat> new_histograms;	///vector of histograms of all particles


	Matrix4f covarianceMatrix(MatrixXf X, int N);	///method for computing covariance matrix Ct (Table 1, step 2)

	MatrixXf drawSamples();		///method for drawing initial samples
	VectorXf initialWeights();	///method for intializing weights (for all i w(i) = 1/N, (4.2.2.))
	Vector4f drawErrorSamples();///method for drawing error sample

	Vector4f meanShift_sample(int j, double lambda);	///meanShift for one sample
	void meanShift(double lambda);						///performing meanShift on all samples

	Vector4f estimate();								///method for final estimation of object position based on final weights and particles
	void perturbation();								///method for performing perturbation (Table 1, step 9)

	double Klambda(Vector4f x, double lambda);			//method for computing Gaussian function of vector
	double reweight_sample(int i);						//reweighting procedure for one sample ((11), (12), (13), (14))
	void reweight();									//reweighting procedure for all samples

	double kernelRadius();

	Mat calcHistogram(Mat image);	//method for computing histogram for a given image
	vector<Mat> calcHistograms();	//method for computing histograms for all new particles
	double computeBhatcharyyaDistance(Mat query, Mat pattern);	//method for computing BhatcharyyaDistance between two histograms
	//reweight again

public:
	KPF();
	KPF(int radius, Mat pattern);
	~KPF();

	Vector4f compute(Mat omni_image, Mat transformed_image);	//public method enabling user to track the object
};

