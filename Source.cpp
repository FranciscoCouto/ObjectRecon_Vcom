#include <string>
#include <iostream>

#include "Utils.h"

#define NUM_FILES_TRAIN 50000 //Number of files to be used in images/train
#define NUM_FILES_TEST 30 //Number of files to be used in images/test
#define DICTIONARY_SIZE 1000//typical values range from 10^3 to 10^5, however this depends much on the application.

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);

Utils u(DICTIONARY_SIZE, NUM_FILES_TRAIN, NUM_FILES_TEST, tc);


int main(int argc, char** argv)
{
	initModule_nonfree();
	Mat train_descriptors, dictionary;
	
	int num_files = NUM_FILES_TRAIN - u.fails.size();
	Mat trainingData(num_files, DICTIONARY_SIZE, CV_32FC1);
	
	Mat labels(num_files, 1, CV_32FC1);

	//1. We pick out features from all the images in our training dataset
	train_descriptors = u.extractLocalFeaturesSURF();

	//We cluster these features using any clustering algorithm
	dictionary = u.CreateBOW(train_descriptors);

	//3. We use the cluster as a vocabulary to construct histograms. We simply count the no. of features from each image belonging to each cluster. 
	//Then we normalize the histograms by dividing it with the no. of features. 
	//Therefore, each image in the dataset is represented by one histogram.
	trainingData = u.CreateTrainingData(dictionary);

	//4. Parse the csv with the correspondent labels that give meaning to each class previously parsed
	labels = u.parseCSV();

	//5. Then these histograms are passed to an SVM for training. 
	u.applySVM(trainingData, labels, dictionary);

	std::getchar();

	return 0;

}