#include "omnivision_tracking.h"

//Cholesky decomposition of matrix A
vector<vector<double> > cholesky(vector<vector<double> > A)
{
	int n = A.size();
	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;
	vector<vector<double> > l(n, vector<double>(n));
	l[0][0] = sqrt(A[0][0]);
	for (int j = 1; j <= n - 1; j++)
		l[j][0] = A[j][0] / l[0][0];
	for (int i = 1; i <= (n - 2); i++)
	{
		for (int k = 0; k <= (i - 1); k++)
			sum1 += pow(l[i][k], 2);
		l[i][i] = sqrt(A[i][i] - sum1);
		for (int j = (i + 1); j <= (n - 1); j++)
		{
			for (int k = 0; k <= (i - 1); k++)
				sum2 += l[j][k] * l[i][k];
			l[j][i] = (A[j][i] - sum2) / l[i][i];
		}
	}
	for (int k = 0; k <= (n - 2); k++)
		sum3 += pow(l[n - 1][k], 2);
	l[n - 1][n - 1] = sqrt(A[n - 1][n - 1] - sum3);
		return l;
}

vector<vector<double>> convertToVector(Mat A)
{
	vector<vector<double>> result(A.rows,vector<double>(A.rows));
	for (int i = 0; i < A.rows; ++i)
	{
		for (int j = 0; j < A.cols; ++j)
		{
			result.at(i).at(j) = A.at <double>(i, j);
		}
	}
	return result;
}

void printVector(vector<vector<double>> A)
{
	cout << "[";
	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(i).size(); j++)
		{
			cout << A.at(i).at(j);
			if (!((j == A.at(i).size() - 1) && (i == A.size() - 1))) cout << ", ";
		}
		if (i != A.size() - 1) cout << endl;
	}
	cout << "]" << endl;
}

Mat convertToMat(vector<vector<double>> v)
{
	Mat result = Mat::zeros(Size(v.at(0).size(), v.at(0).size()), CV_64FC1);
	for (int i = 0; i < result.rows; ++i)
	{
		for (int j = 0; j < result.cols; ++j)
		{
			result.at <double>(i, j) = v[i][j];
		}
	}
	return result;
}
Mat choleskyOpenCV(Mat A)
{
	vector<vector<double>> v = convertToVector(A);
	cout << "Vector v: " << endl;
	printVector(v);
	vector<vector<double>> v_res = cholesky(v);
	cout << "Vector v_res: " << endl;
	printVector(v_res);
	Mat result = convertToMat(v_res);
	return result;
}
Mat bsMask(int r, int x, int y, int width, int height)
{
    Mat mask = Mat(height, width, CV_8UC3);
    for (int i=0; i<mask.rows; i++)
    {
        for (int j=0;j<mask.cols; j++)
        {
            if (((i-y)*(i-y)+(j-x)*(j-x))>r*r)
            {
                mask.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
            else
            {
                mask.at<Vec3b>(i,j) = Vec3b(1,1,1);
            }
        }
    }
    return mask;
}
Mat cartToPolarMatrix(int R, int x, int y)
{
    Mat transformation_matrix(R,360, CV_16UC2);

    for (int theta = 0; theta < transformation_matrix.cols; theta++)
    {
        for (int r = 0; r<transformation_matrix.rows;r++)
        {
            Vec2s point_pixel = Vec2s(r*cos(CV_PI*theta/180)+x, r*sin(CV_PI*theta/180)+y);
            transformation_matrix.at<Vec2s>(r,theta) = point_pixel;
        }
    }
    return transformation_matrix;
}

Mat cartToPolarTransformation(Mat transformation_matrix, Mat source_image)
{
    Mat image_transformed(transformation_matrix.rows,transformation_matrix.cols,CV_8UC3);

    for (int y=0; y<image_transformed.rows; y++)
    {
        for (int x=0; x<image_transformed.cols; x++)
        {
            Vec2s pixel_coordinates; //vector containing coordinates (x,y) on the source_image of pixel to insert into image_transformed
            pixel_coordinates = transformation_matrix.at<Vec2s>(y,x);
            image_transformed.at<Vec3b>(y,x) = source_image.at<Vec3b>(pixel_coordinates[1],pixel_coordinates[0]);
        }
    }
    return image_transformed;
	
}

void calculateOpticalFlow(Mat &u, Mat &v, Mat Ir, Mat Itheta, Mat It, double alpha)
{
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

Mat calculateOpticalFlowMagnitude(Mat u, Mat v)
{
	Mat magnitude = u.mul(u) + v.mul(v);
	sqrt(magnitude, magnitude);
	return magnitude;
}

Mat drawSamples(int r_max, int n, int max_a, int max_b)
{
	srand(time(NULL));
	Mat samples = Mat::zeros(n,4,CV_64FC1);
	for (int i = 0; i < n; i++)
	{
		samples.at<double>(i,0) = rand() % 360;
		samples.at<double>(i, 1) = rand() % r_max;
		samples.at<double>(i, 2) = rand() % max_a;
		samples.at<double>(i, 3) = rand() % max_b;
	}

	return samples;
}

double computeBhatcharyyaDistance(Mat query, Mat pattern)
{
	Mat hist_query, hist_pattern; 
	int channels[] = { 0, 1, 2 }; //numbers of channels used in histogram
	int hist_size[] = { 16, 16, 16 }; //sizes of bins
	float BGR_ranges[] = { 0, 255 }; //range of values for 1 channel
	const float* ranges[] = { BGR_ranges, BGR_ranges, BGR_ranges }; //ranges of values for all channels
	calcHist(&query, 1, channels, Mat(), hist_query, 3, hist_size, ranges);
	calcHist(&pattern, 1, channels, Mat(), hist_pattern, 3, hist_size, ranges);
	
	normalize(hist_query, hist_query, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(hist_pattern, hist_pattern, 0, 1, NORM_MINMAX, -1, Mat());

	double distance = compareHist(hist_query, hist_pattern, CV_COMP_BHATTACHARYYA);

	return distance;
}

void perturbation(Mat e, Mat &samples, double lambda, Mat At)
{
	for (int y = 0; y < samples.rows; y++)
	{
		Mat row = samples.row(y);
		Mat part = lambda * At * e;
		row = row + part.t();
	}
}
Mat initialWeights(int N)
{
	Mat weights = 1.0/N*Mat::ones(N, 4, CV_64FC1);
	return weights;
}
void KPFx(int r_max)
{
	const double nx = 4;
	const double N = 100;
	const double exponent = 1 / (nx + 4);
	const double base = 4 / ((nx + 2)*N);
	const double lambda_opt = pow(base, exponent);
	const double lambda_0 = 0.5 * lambda_opt;

	Mat samples = drawSamples(r_max);
	Mat weights = initialWeights(N);
	Mat C, mean;
	calcCovarMatrix(samples, C, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	Mat At = Mat::Mat(C.rows, C.cols, CV_64FC1);
	At = choleskyOpenCV(C);
	RNG random_generator;
	Mat e = Mat::zeros(nx,1,CV_64FC1);
	random_generator.fill(e, RNG::NORMAL, 0, 1);
	for (int y = 0; y < samples.rows; y++)
	{
		Mat row = samples.row(y);
		Mat part = lambda_0 * At * e;
		row = row + part.t();
	}
	cout << samples;
	cout << weights;
	int I = 3; //number of iterations
	for (int j = 0; j < I; j++)
	{
		double lambda = kernelRadius(N, j+1);
		for (int i = 0; i < N; i++)
		{
			Mat row = samples.row(i);
			row = meanShift(samples, weights, i, N, lambda);
		}
		perturbation(e, samples, lambda, At);
	}
	int a;
	cin >> a;
}

Mat Klambda(Mat x, double lambda)
{
	Mat result = Mat::zeros(4,1,CV_64FC1);
	for (int i = 0; i < 4; i++)
	{
		result.at<double>(i, 0) = exp(-x.at<double>(0,i)*x.at<double>(0,i)/(2*lambda*lambda));
	}
	return result.t();
}
Mat meanShift(Mat samples, Mat weights, int i, int N, double lambda)
{
	Mat numerator = Mat::zeros(4, 1, CV_64FC1);;
	Mat denominator = Mat::zeros(4, 1, CV_64FC1);;
	Mat result = Mat::zeros(4, 1, CV_64FC1);
	Mat s_i = samples.row(i);
	for (int l = 1; l < N; l++)
	{
		Mat s_l = samples.row(l);
		Mat omega_l = weights.row(l);
		numerator += Klambda(s_i - s_l, lambda)*omega_l.t() *s_l;
		denominator = Klambda(s_i - s_l, lambda)*omega_l.t();
	}
	for (int i = 0; i < 4; i++)
	{
		result.at<double>(i, 0) = numerator.at<double>(i) / denominator.at<double>(0);
	}
	return result;
}
Mat CholeskyDecomposition(Mat A, int n)
{
	Mat L = A.clone();

	for (int i = 0; i < n; i++)
	{
		L.at<double>(i, i) = sqrt(A.at<double>(i, i));
		for (int j = i + 1; j < n; j++)
		{
			L.at<double>(j, i) = A.at<double>(j, i) / L.at<double>(i, i);
			for (int k = i + 1; k < j + 1; k++)
			{
				A.at<double>(j, k) = A.at<double>(j, k) - L.at<double>(j, i)*L.at<double>(k, i);
			}
		}

	}
	Mat mask = lowerDiagonalMask(n);
	L = L.mul(mask);
	return L;
}

Mat lowerDiagonalMask(int n)
{
	Mat mask = Mat::zeros(n, n, CV_64FC1);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (j <= i) mask.at<double>(i, j) = 1;
		}
	}
	return mask;
}

double kernelRadius(int N, int i)
{
	int nx = 4;
	double eta = 0.8;
	double lambda = 0.5*pow(eta,i)*pow(4 / ((nx + 2)*N), 1 / (nx + 4));
	return lambda;
}