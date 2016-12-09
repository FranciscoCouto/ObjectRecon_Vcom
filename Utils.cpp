#include "Utils.h"

Utils::Utils(int n_words, int n_train_images, int n_test_images, TermCriteria tc)
{

	this->n_words = n_words;
	this->n_train_images = n_train_images;
	this->n_test_images = n_test_images;
	this->tc = tc;
}

Mat Utils::extractLocalFeaturesSIFT()
{

	cout << "Extracting the Descriptors (Feature Vectors) using SIFT" << endl;

	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

	Mat image;
	string filename;
	std::vector<KeyPoint> keypoints;
	Mat extracted_descriptor, train_descriptors;

	for (int i = 1; i <= n_train_images; i++)
	{
		filename = "images/train/" + to_string(i) + ".png";
		if (!openImage(filename, image)) {
			cout << "Could not open image " + to_string(i);
			continue;
		}

		detector->detect(image, keypoints);

		if (keypoints.empty()) {
			//cout << "Could not get keypoints for " + to_string(i);
			fails.push_back(i);
			continue;
		}

		extractor->compute(image, keypoints, extracted_descriptor);
		train_descriptors.push_back(extracted_descriptor);

		loadbar(i, n_train_images, 50);
	}

	cout << endl;
	return train_descriptors;

}

Mat Utils::CreateBOW(Mat train_descriptors)
{

	cout << "Creating Bag of Words" << endl;

	BOWKMeansTrainer bowTrainer(n_words, tc, 1, KMEANS_PP_CENTERS);

	Mat dictionary = bowTrainer.cluster(train_descriptors); //Created vocabulary using KMeans

	cout << "Storing Bag of Words" << endl;

	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	return dictionary;

}

Mat Utils::CreateTrainingData(Mat dictionary)
{

	cout << "Creating Training Data for SVM" << endl;

	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	Mat image, BOW_Descriptor, training_data;
	string filename;
	vector<KeyPoint> keypoints;


	cout << "Extracting histograms in the form of BOW for each trainning image" << endl;

	for (int i = 1; i <= n_train_images; i++)
	{
		filename = "images/train/" + to_string(i) + ".png";
		if (!openImage(filename, image)) {
			cout << "Could not open image " + to_string(i);
			continue;
		}


		detector->detect(image, keypoints);

		if (keypoints.empty()) {
			//cout << "Could not get keypoints for " + to_string(i);
			continue;
		}


		extractor->compute(image, keypoints, BOW_Descriptor);
		bowDE.compute(image, keypoints, BOW_Descriptor);

		training_data.push_back(BOW_Descriptor);

		loadbar(i, n_train_images, 50);

	}


	cout << "Training Data Created" << endl;

	return training_data;

}

void Utils::applySVM(Mat training_data, Mat dictionary)
{

	cout << "Applying SVM" << endl;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = tc;

	cout << "Training the SVM" << endl;

	CvSVM SVM;
	cout << training_data.size() << "     " << labels.size();

	CV_Assert(!training_data.empty() && training_data.type() == CV_32FC1);
	CV_Assert(!labels.empty() && labels.type() == CV_32SC1);

	SVM.train(training_data, labels, Mat(), Mat(), params); //SVM is trainning with the images (descriptors) - experimentar trainauto

	cout << "SVM Classifier Trained" << endl;

	Mat image, BOW_Descriptor;
	string filename;
	vector<KeyPoint> keypoints;

	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	for (int i = 1; i <= n_test_images; i++)
	{
		filename = "images/test/" + to_string(i) + ".png";
		if (!openImage(filename, image))
			continue;

		detector->detect(image, keypoints);
		extractor->compute(image, keypoints, BOW_Descriptor);
		bowDE.compute(image, keypoints, BOW_Descriptor);

		//cout << BOW_Descriptor;


		float res = SVM.predict(BOW_Descriptor);

		cout << "Image " << i << "predicted to be: " << res << names[res] << endl;

		FileStorage fs("results.txt", FileStorage::APPEND);

		fs << "Image" << names[res];
		fs.release();

		loadbar(i, n_test_images, 50);
	}


	waitKey(-1);
}

bool Utils::openImage(const std::string & filename, Mat & image)
{
	//cout << filename;
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

void Utils::parseCSV()
{

	cout << "Parsing Labels from CSV" << endl;

	ifstream file("trainLabels.csv");

	// Get and drop a line
	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	string line, id;

	int what_is_it = 0, index = 1;

	while (getline(file, line))
	{
		if ((find(fails.begin(), fails.end(), index) != fails.end())) { index++; cout << "pissou";  continue; }

		std::stringstream  lineStream(line);
		std::string        cell;
		while (getline(lineStream, cell, ','))
		{

			if (what_is_it == 0) {
				labels.push_back(stoi(cell));
				what_is_it++;
			}
			else {
				names.push_back(cell);
				what_is_it = 0;
				index++;
			}
		}

		if (index -1 == n_train_images) { break; }
	}

	cout << "Ended Parsing Labels from CSV" << endl;
}

inline void Utils::loadbar(unsigned int x, unsigned int n, unsigned int w)
{

	if ((x != n) && (x % (n / 100 + 1) != 0)) return;

	float ratio = x / (float)n;
	int   c = ratio * w;

	cout << setw(3) << (int)(ratio * 100) << "% [";
	for (int x = 0; x < c; x++) cout << "=";
	for (int x = c; x < w; x++) cout << " ";
	cout << "]\r" << flush;
}