#include "Utils.h"

Utils::Utils(int dictionary_size, int n_train_images, int n_test_images, TermCriteria tc)
{

	this->dictionary_size = dictionary_size;
	this->n_train_images = n_train_images;
	this->n_test_images = n_test_images;
	this->tc = tc;

	names = { {"airplane", 1},{ "automobile", 2 },{ "bird", 3 },{"cat", 4 },{ "deer", 5 },{ "dog", 6 },{ "frog", 7 },{ "horse", 8 },{ "ship", 9 },{ "truck", 10 } };
}

Mat Utils::extractLocalFeaturesSURF()
{

	cout << "Extracting the Descriptors (Feature Vectors) using SURF" << endl;

	Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	//SurfFeatureDetector detector(400); //test to check if less images failed by changing threshold of hessian matrix
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");

	Mat image;
	string filename;
	std::vector<KeyPoint> keypoints;
	Mat extracted_descriptor, train_descriptors;

	for (int i = 1; i <= n_train_images; i++)
	{
		filename = "images/train/" + to_string(i) + ".png";
		if (!openImage(filename, image)) {
			fails.push_back(i);
			cout << "Could not open image " + to_string(i);
			continue;
		}

		detector->detect(image, keypoints);

		if (keypoints.empty()) {
			fails.push_back(i);
			continue;
		}

		extractor->compute(image, keypoints, extracted_descriptor);

		if (extracted_descriptor.empty()) { fails.push_back(i); continue; }

		train_descriptors.push_back(extracted_descriptor);

		loadbar(i, n_train_images, 50);
	}

	//cout << train_descriptors.size() << endl;
	cout << endl << "Got the unclustered features!" << endl;
	return train_descriptors;

}

Mat Utils::CreateBOW(Mat train_descriptors)
{

	cout << "Clustering Features..." << endl;

	int retries = 1;
	int flags = KMEANS_PP_CENTERS;

	BOWKMeansTrainer bowTrainer(dictionary_size, tc, retries, flags);

	Mat dictionary = bowTrainer.cluster(train_descriptors); //Created vocabulary using KMeans

	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	cout << "Vocabulary stored in Dictionary.yml!" << endl;

	return dictionary;

}

Mat Utils::CreateTrainingData(Mat dictionary)
{

	cout << "Creating Training Data for SVM" << endl;

	//create a fast nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

	Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	//SurfFeatureDetector detector(400); //test to check if less images failed by changing threshold of hessian matrix
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");

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
		if ((find(fails.begin(), fails.end(), i) != fails.end())) { continue; }

		filename = "images/train/" + to_string(i) + ".png";
		if (!openImage(filename, image)) {

			cout << "Could not open image " + to_string(i);
			continue;
		}


		detector->detect(image, keypoints);

		if (keypoints.empty()) { continue;	}

		bowDE.compute(image, keypoints, BOW_Descriptor);

		if (BOW_Descriptor.empty()) { continue; }

		training_data.push_back(BOW_Descriptor);

		loadbar(i, n_train_images, 50);

	}

	cout << endl;
	cout << "Training Data Created" << endl;

	return training_data;

}

void Utils::applySVM(Mat training_data, Mat labels, Mat dictionary)
{

	cout << "Applying SVM" << endl;

	CvSVMParams params;
	//SVM type is defined as n-class classification n>=2, allows imperfect separation of classes
	params.svm_type = CvSVM::C_SVC;
	// No mapping is done, linear discrimination (or regression) is done in the original feature space.
	params.kernel_type = CvSVM::RBF;

	//params.gamma = 0.1;
	//params.C = 0;
	//Define the termination criterion for SVM algorithm.
	//Here stop the algorithm after the achieved algorithm-dependent accuracy becomes lower than epsilon
	//or run for maximum 100 iterations
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	cout << "Training the SVM" << endl;

	CvSVM SVM;

	cout << training_data.size() << "     " << labels.size() << endl;

	CV_Assert(!training_data.empty() && training_data.type() == CV_32FC1);
	CV_Assert(!labels.empty() && labels.type() == CV_32SC1);

	//#pragma omp parallel for schedule(dynamic,3)
	SVM.train_auto(training_data, labels, Mat(), Mat(), params); //SVM is trainning with the images (descriptors) - experimentar trainauto
	//SVM.train(training_data, labels, Mat(), Mat(), params);
	cout << params.gamma << params.degree;

	cout << "SVM Classifier Trained" << endl;

	Mat image, BOW_Descriptor;
	string filename;
	vector<KeyPoint> keypoints;

	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

	Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	ofstream results;
	results.open("results.csv");

	results << "Id, name\n";

	for (int i = 1; i <= n_test_images; i++)
	{
		filename = "images/test/" + to_string(i) + ".png";
		if (!openImage(filename, image))
			continue;

		detector->detect(image, keypoints);

		if (keypoints.empty()) {
			//cout << "Could not get keypoints for " + to_string(i);
			continue;
		}

		extractor->compute(image, keypoints, BOW_Descriptor);

		if (BOW_Descriptor.empty()) {
			//cout << "Could not get keypoints for " + to_string(i);
			continue;
		}

		bowDE.compute(image, keypoints, BOW_Descriptor);


		float res = SVM.predict(BOW_Descriptor); //retorna a label correspondente, de seguida procuramos no map o nome associado

		cout << "Image " << i << "predicted to be: " << findInMap(res) << res << endl;

		results << i << "," << findInMap(res) << "," << "\n";
		
		loadbar(i, n_test_images, 50);
	}

	results.close();
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

Mat Utils::parseCSV()
{

	cout << "Parsing Labels from CSV" << endl;

	ifstream file("trainLabels.csv");

	Mat labels;

	// Get and drop a line
	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	string line, id;

	int what_is_it = 0, index = 1;

	while (getline(file, line))
	{
		if ((find(fails.begin(), fails.end(), index) != fails.end())) { index++; continue; }

		std::stringstream  lineStream(line);
		std::string        cell;
		while (getline(lineStream, cell, ','))
		{

			if (what_is_it == 0) {
				what_is_it++;
			}
			else {
				//cout << cell << names.at(cell);
				labels.push_back(names.at(cell));
				what_is_it = 0;
				index++;
			}
		}

		if ((index -1) == n_train_images) { break; }
	}

	cout << "Finished Parsing Labels from CSV" << endl;

	return labels;
}

String Utils::findInMap(int value) {

	std::map<std::string, int>::const_iterator it;
	string name = "";

	for (it = names.begin(); it != names.end(); ++it)
	{
		if (it->second == value)
		{
			name = it->first;
			break;
		}
	}

	return name;

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