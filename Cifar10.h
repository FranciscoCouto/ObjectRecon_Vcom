#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
using namespace cv;
void read_CIFAR10(Mat &trainX, Mat &testX, Mat &trainY, Mat &testY);
void read_batch(string filename, vector<Mat> &vec, Mat &label);
Mat concatenateMat(vector<Mat> &vec);
Mat concatenateMatC(vector<Mat> &vec);