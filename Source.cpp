#include <string>
#include <iostream>

#include "Utils.h"

#define NUM_FILES_TRAIN 1500 //Number of files to be used in images/train
#define NUM_FILES_TEST 100 //Number of files to be used in images/test
#define DICTIONARY_SIZE 250 //typical values range from 10^3 to 10^5, however this depends much on the application.

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);

Utils u(DICTIONARY_SIZE, NUM_FILES_TRAIN, NUM_FILES_TEST, tc);


int main(int argc, char** argv)
{
	initModule_nonfree();
	Mat train_descriptors, dictionary;
		
	Mat	training_data;
	Mat labels;

	// Extract features from training set that contains all classes. - USING SURF: Fast version of SIFT
	train_descriptors = u.extractLocalFeaturesSURF();

	//Create a vocabulary of features by clustering the features (kNN, etc). Let's say 1000 features long.
	dictionary = u.CreateBOW(train_descriptors);

	//This time check the features in each image for their closest clusters in the vocabulary.
	//Create a histogram of responses for each image to words in the vocabulary, it will be a 1000 - entries long vector.
	//Create a sample - label dataset for the training.
	training_data = u.CreateTrainingData(dictionary);

	 // we have to convert to float:
	training_data.convertTo(training_data, CV_32F);

	//Parse the csv with the correspondent labels that give meaning to each class previously parsed
	labels = u.parseCSV();

	//Run the trainer and the classifier and it should, god willing, give you the right class
	u.applySVM(training_data, labels, dictionary);

	std::getchar();

	return 0;

}