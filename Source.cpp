#include <string>
#include <iostream>
#include "Cifar10.h"
#include "Utils.h"
#include <ctime>

#define NUM_FILES_TRAIN 50000 //Number of files to be used in images/train
#define NUM_FILES_TEST 10000 //Number of files to be used in images/test
#define DICTIONARY_SIZE 1000//typical values range from 10^3 to 10^5, however this depends much on the application.

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);

Utils u(DICTIONARY_SIZE, NUM_FILES_TRAIN, NUM_FILES_TEST, tc);


int main(int argc, char** argv)
{
	initModule_nonfree();

	Mat trainX, testX;
	Mat trainY, testY;
	trainX = Mat::zeros(1024, 50000, CV_64FC1);
	testX = Mat::zeros(1024, 10000, CV_64FC1);
	trainY = Mat::zeros(1, 50000, CV_64FC1);
	testX = Mat::zeros(1, 10000, CV_64FC1);

	read_CIFAR10(trainX, testX, trainY, testY);

	//for (int i = 0; i < testX.cols; i++)
	//{
	//	string filename = "./img_test/" + to_string(i);
	//	filename += ".jpg";
	//	cout << filename << endl;
	//	Mat temp = Mat::zeros(32, 32, CV_64F);
	//	int acc = 0;
	//	for (int j = 0; j < 32; j++) {
	//		for (int k = 0; k < 32; k++) {
	//			temp.at<double>(j, k) = testX.at<double>(acc, i);
	//			acc++;
	//		}
	//	}
	//	temp.convertTo(temp, CV_8UC3, 255.0);
	//	imwrite(filename, temp);
	//	//imshow("t", temp);
	//	//waitKey();
	//}
	//return 0;

	int start_s = clock();

	Mat train_descriptors, dictionary;

	int num_files = NUM_FILES_TRAIN - u.fails.size();
	Mat trainingData;// (num_files, DICTIONARY_SIZE, CV_32FC1);

	Mat labels;// (num_files, 1, CV_32FC1);
	string filename = "dictionary.yml";
	if (FILE *file = fopen(filename.c_str(), "r")) {
		fclose(file);
		FileStorage fs("dictionary.yml", FileStorage::READ);
		fs["vocabulary"] >> dictionary;
		fs.release();
		trainingData = u.LoadTrainingData(dictionary);
	}
	else {
		//1. We pick out features from all the images in our training dataset
		train_descriptors = u.extractLocalFeaturesSURF();

		//We cluster these features using any clustering algorithm
		dictionary = u.CreateBOW(train_descriptors);
		//3. We use the cluster as a vocabulary to construct histograms. We simply count the no. of features from each image belonging to each cluster. 
		//Then we normalize the histograms by dividing it with the no. of features. 
		//Therefore, each image in the dataset is represented by one histogram.
		trainingData = u.CreateTrainingData(dictionary);
	}

	//4. Parse the csv with the correspondent labels that give meaning to each class previously parsed
	labels = u.parseCSV();

	//5. Then these histograms are passed to an SVM for training. 
	u.applySVM(trainingData, labels, dictionary, testY);

	int stop_s = clock();
	double deltaT = (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
	cout << "execution time: " <<  deltaT << endl;
	ofstream exec_time;
	exec_time.open("exec_time_"+to_string(deltaT));
	exec_time.close();
	std::getchar();

	return 0;

}