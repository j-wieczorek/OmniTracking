#ifndef OMNIVISION_TRACKING
#define OMNIVISION_TRACKING

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include <time.h>
#include <math.h>

using namespace cv;
using namespace std;

Mat bsMask(int r, int x, int y, int width, int height);
Mat cartToPolarMatrix(int R, int x, int y);
Mat cartToPolarTransformation(Mat transformation_matrix, Mat source_image);
void calculateOpticalFlow(Mat &u, Mat &v, Mat Ir, Mat Itheta, Mat It, double alpha = 500.0);
Mat calculateOpticalFlowMagnitude(Mat u, Mat v);
Mat drawSamples(int r_max, int n = 100, int max_a = 50, int max_b = 50);
vector<vector<double> > cholesky(vector<vector<double> > A);
Mat CholeskyDecomposition(Mat A, int n);
Mat choleskyOpenCV(Mat A);
void printVector(vector<vector<double>> A);
Mat lowerDiagonalMask(int n);
double computeBhatcharyyaDistance(Mat query, Mat pattern);
Mat initialWeights(int N);
Mat meanShift(Mat samples, Mat weights, int i, int N, double lambda);
void perturbation(Mat e, Mat &samples, double lambda, Mat At);
double kernelRadius(int N, int i);
void KPFx(int r_max);
#endif
