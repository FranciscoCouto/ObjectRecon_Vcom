#pragma once
#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <fstream>
#include <string>

using namespace cv;
using namespace std;

class Utils
{
public:

	int dictionary_size;
	int n_train_images, n_test_images;
	TermCriteria tc;

	map< std::string, int> names;
	vector <int> fails;
	string alg = "SIFT";

	Utils(int n_words, int n_train_images,int n_test_images, TermCriteria tc);

	//Feature Extraction - SIFT
	Mat extractLocalFeaturesSURF();

	//Clustering the Feature Vectors with KMeans
	Mat CreateBOW(Mat train_descriptors);

	//Create Training Data by creating histograms for each image to be used in SVM
	Mat CreateTrainingData(Mat dictionary);

	void applySVM(Mat training_data, Mat labels, Mat dictionary, Mat testY);

	//OpenCv shenanigans
	bool openImage(const std::string &filename, Mat &image);

	void imadjust(const Mat1b& src, Mat1b& dst);

	//Parsing Files
	Mat parseCSV();

	String findInMap(int value);

	inline void loadbar(unsigned int x, unsigned int n, unsigned int w);
};