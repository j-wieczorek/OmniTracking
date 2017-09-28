#include "KPF.h"
#include <math.h>

KPF::KPF()
{
	nx = 4; //one sample vector has 4 elements: r, theta, max_a and max_b
	N = 100; //initial number of particles
	I = 3; //number of iterations

	lambda_opt = kernelRadius();
	lambda_0 = 0.5 * lambda_opt;
	
	eta = 0.8; //number empirically chosen by authors of paper
	sigma = 1.0;

	max_a = 50;
	max_b = 50;
	R = 540;
	perimeter = 360;
}
KPF::KPF(int radius, Mat pattern)
{
	nx = 4; //one sample vector has 4 elements: r, theta, max_a and max_b
	N = 100; //initial number of particles
	I = 3; //number of iterations

	lambda_opt = kernelRadius();
	lambda_0 = 0.5 * lambda_opt;
	eta = 0.8; //number empirically chosen by authors of paper
	sigma = 1.0;

	max_a = 50;
	max_b = 50;
	perimeter = 360;
	R = radius;

	pattern_histogram = calcHistogram(pattern);
}

MatrixXf KPF::drawSamples()
{
	MatrixXf temp = 0.5*(MatrixXf::Random(N,4) + MatrixXf::Constant(N,4,1.0));
	cout << temp << endl;
	RowVectorXf v(4);
	v << perimeter, R, max_a, max_b;
	MatrixXf R(v.colwise().replicate(N));
	MatrixXf result = temp.cwiseProduct(R);
	for (int i = 0; i < result.rows(); ++i)
	{
		Vector4f sample = result.row(i);
		if ((sample(0) + sample(2)) > perimeter)	result(i, 2) = perimeter - sample(0) - 1;
		if ((sample(1) + sample(3)) > this->R)			result(i, 3) = this->R - sample(1) - 1;
	}
	return result;
}

VectorXf KPF::initialWeights()
{
	VectorXf result = VectorXf::Constant(N, 1.0 / N);
	cout << result;
	return result;
}

Matrix4f KPF::covarianceMatrix(MatrixXf X, int N)
{
	MatrixXf ones = MatrixXf::Constant(N, 1, 1.0);
	MatrixXf mean = samples.colwise().mean();
	MatrixXf result = (samples - ones*mean).transpose()*((samples - ones*mean));
	return result;
}
Vector4f KPF::drawErrorSamples()
{
	Vector4f result;
	result = 0.5*(Vector4f::Random() + 2.0*Vector4f::Ones());
	return result;
}

void KPF::perturbation()
{
	for (int y = 0; y < samples.rows(); y++)
	{
		Vector4f row = samples.row(y);
		Vector4f part = lambda_0 * At * e;
		row = row + part;
	}
	new_histograms = calcHistograms();
}

double KPF::Klambda(Vector4f x, double lambda)
{
	double result = exp(-1.0*((x.transpose() *lambda* Matrix4f::Identity() * x)(0)));
	if (result > 0.0) cout << "Lambda = " << lambda << endl;
	return result;
}
double KPF::reweight_sample(int i)
{
	Vector4f s_i = samples.row(i);

	double denominator = 0.0;
	double nominator = 0.0;

	double distance = computeBhatcharyyaDistance(new_histograms.at(i), pattern_histogram);
	double distance_weight = 1 / (2 * CV_PI*sigma)*exp(-1 / (2 * sigma*sigma*distance*distance));
	double motion_part = 0;
	for (int l = 1; l < N; l++)
	{
		Vector4f s_l = samples.row(l);
		Vector4f s_l_prev = previous_samples.row(l);
		double w_l_prev = previous_weights(l);

		denominator += Klambda(s_i - s_l, lambda_opt);
		nominator += Klambda(s_i - s_l_prev, lambda_opt)*w_l_prev;
	}
	nominator *= distance_weight;
	return nominator / denominator;
	return 0.0;
}
void KPF::reweight()
{
	for (int i = 0; i < N; i++)
	{
		weights(i) = reweight_sample(i);
	}
}
double KPF::kernelRadius()
{
	double lambda = pow(4.0 / ((nx + 2)*N), 1.0 / (nx + 4));
	return lambda;
}
Vector4f KPF::meanShift_sample(int i, double lambda)
{
	
	Vector4f numerator = Vector4f::Zero();
	double denominator = 0.0;
	Vector4f result = Vector4f::Zero();

	Vector4f s_i = samples.row(i);
	for (int l = 0; l < N; l++)
	{
		Vector4f s_l = samples.row(l);
		double omega_l = weights(l);
		numerator += Klambda(s_i - s_l, lambda)*omega_l *s_l;
		denominator += Klambda(s_i - s_l, lambda)*omega_l;
	}
	result = numerator / denominator;
	return result;
}

void KPF::meanShift(double lambda)
{
	for (int i = 0; i < N; ++i)
	{
		samples.row(i) = meanShift_sample(i, lambda);
	}
}
Mat KPF::calcHistogram(Mat image)
{
	Mat histogram;

	int channels[] = { 0, 1, 2 };
	float b_ranges[] = { 0, 255 };
	float g_ranges[] = { 0, 255 };
	float r_ranges[] = { 0, 255 };
	const float* ranges[] = { b_ranges, g_ranges, r_ranges };
	int hist_size[] = { 16, 16, 16 };

	calcHist(&image, 1, channels, Mat(), histogram, 2, hist_size, ranges);

	return histogram;
}
vector<Mat> KPF::calcHistograms()
{
	vector<Mat> calculated_histograms;
	for (int i = 0; i < samples.rows(); ++i)
	{
		cout << samples.row(i) << endl;
		Rect rectangle = Rect(int(samples(i, 0)), int(samples(i, 1)), int(samples(i, 2)), int(samples(i, 3)));
		Mat sample_image = this->image(Rect(int(samples(i, 0)), int(samples(i, 1)), int(samples(i, 2)), int(samples(i, 3))));
		Mat histogram = calcHistogram(sample_image);

		/*
		int channels[] = { 0, 1, 2 };
		float b_ranges[] = { 0, 255 };
		float g_ranges[] = { 0, 255 };
		float r_ranges[] = { 0, 255 };
		const float* ranges[] = { b_ranges, g_ranges, r_ranges };
		int hist_size[] = { 16, 16, 16 };

		calcHist(&sample_image, 1, channels, Mat(), histogram, 2, hist_size, ranges);
		*/
		calculated_histograms.push_back(histogram);
	}
	return calculated_histograms;
}
double KPF::computeBhatcharyyaDistance(Mat query_hist, Mat pattern_hist)
{
	double distance = compareHist(query_hist, pattern_hist, CV_COMP_BHATTACHARYYA);
	return distance;
}

Vector4f KPF::compute(Mat omni_image, Mat transformed_image)
{
	this->image = transformed_image;
	samples = drawSamples();
	weights = initialWeights();
	previous_samples = samples;
	previous_weights = weights;

	
	new_histograms = calcHistograms();
	Ct = covarianceMatrix(samples, N);
	At = Ct.llt().matrixL();
	e = drawErrorSamples();
	perturbation();
	reweight();
	for (int j = 0; j < I; j++)
	{
		lambda_opt = kernelRadius();
		lambda_0 = 0.5*pow(eta, j)*lambda_opt;
		meanShift(0.5*lambda_opt);
		perturbation();
		reweight();
	}
	x = estimate();
	return x;
}

Vector4f KPF::estimate()
{
	Vector4f result = Vector4f::Zero();
	for (int i = 0; i < N; ++i)
	{
		result += weights(i)*samples.row(i);
	}
	previous_samples = samples;
	previous_weights = weights;
	return result;
}

KPF::~KPF()
{
}
