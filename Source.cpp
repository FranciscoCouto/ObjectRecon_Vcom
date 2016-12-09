#include <string>
#include <iostream>

#include "Utils.h"

#define NUM_FILES_TRAIN 200 //Number of files to be used in images/train
#define NUM_FILES_TEST 20 //Number of files to be used in images/test
#define DICTIONARY_SIZE 40 

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);

Utils u(DICTIONARY_SIZE, NUM_FILES_TRAIN, NUM_FILES_TEST, tc);


int main(int argc, char** argv)
{
	initModule_nonfree();
	Mat train_descriptors, dictionary, trainning_data;

	train_descriptors = u.extractLocalFeaturesSIFT();

	dictionary = u.CreateBOW(train_descriptors);

	trainning_data = u.CreateTrainingData(dictionary);

	u.parseCSV();

	u.applySVM(trainning_data, dictionary);

	waitKey(0);

	return 0;

}