#include <string>
#include <iostream>

#include "Utils.h"

#define NUM_FILES_TRAIN 4000 //Number of files to be used in images/train
#define NUM_FILES_TEST 25 //Number of files to be used in images/test
#define DICTIONARY_SIZE 128 

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);

Utils u(DICTIONARY_SIZE, NUM_FILES_TRAIN, NUM_FILES_TEST, tc);


int main(int argc, char** argv)
{
	initModule_nonfree();
	Mat train_descriptors, dictionary;
		
	Mat	training_data;
	Mat labels(200, 1, CV_32FC1);

	train_descriptors = u.extractLocalFeaturesSIFT();

	dictionary = u.CreateBOW(train_descriptors);

	training_data = u.CreateTrainingData(dictionary);

	 // we have to convert to float:
	training_data.convertTo(training_data, CV_32F);

	labels = u.parseCSV();

	u.applySVM(training_data, labels, dictionary);

	std::getchar();

	return 0;

}