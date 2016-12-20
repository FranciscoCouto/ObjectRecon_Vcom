#include "Utils.h"

Utils::Utils(int dictionary_size, int n_train_images, int n_test_images, TermCriteria tc)
{

	this->dictionary_size = dictionary_size;
	this->n_train_images = n_train_images;
	this->n_test_images = n_test_images;
	this->tc = tc;

	names = { {"airplane", 0},{ "automobile", 1 },{ "bird", 2 },{"cat", 3 },{ "deer", 4 },{ "dog", 5 },{ "frog", 6 },{ "horse", 7 },{ "ship", 8 },{ "truck", 9 } };
}

Mat Utils::extractLocalFeaturesSURF()
{

	cout << "Extracting the Descriptors (Feature Vectors) using SURF" << endl;

	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	//SurfFeatureDetector detector(400); //test to check if less images failed by changing threshold of hessian matrix
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

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

		//Mat1b adjusted;
		//imadjust(image, adjusted);

		detector->detect(image, keypoints);

		if (keypoints.empty()) {
			fails.push_back(i);
			continue;
		}
		else KeyPointsFilter::retainBest(keypoints, 1000);

		extractor->compute(image, keypoints, extracted_descriptor);

		if (extracted_descriptor.empty()) { fails.push_back(i); continue; }
		//else { cout << extracted_descriptor.size(); }

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

	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	//SurfFeatureDetector detector(400); //test to check if less images failed by changing threshold of hessian matrix
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	Mat image, BOW_Descriptor;
	int num_files = n_train_images - fails.size();
	//Mat training_data(num_files, dictionary_size, CV_32FC1);
	Mat training_data;
	string filename;
	vector<KeyPoint> keypoints;


	cout << "Extracting histograms in the form of BOW for each trainning image" << endl;

	for (int i = 1; i <= n_train_images; i++)
	{
		if ((find(fails.begin(), fails.end(), i) != fails.end())) { continue; }

		filename = "images/train/" + to_string(i) + ".png";
		if (!openImage(filename, image)) {
			fails.push_back(i);
			cout << "Could not open image " + to_string(i);
			continue;
		}

		//Mat1b adjusted;
		//imadjust(image, adjusted);

		detector->detect(image, keypoints);

		if (keypoints.empty()) { 
			cout << "Descriptor not found for train img: "<< i << endl; 
			fails.push_back(i);
			continue; 
		}

		bowDE.compute(image, keypoints, BOW_Descriptor);

		if (BOW_Descriptor.empty()) { 
			cout << "BOW not found for train img: " << i << endl;
			fails.push_back(i);
			continue; 
		}


		training_data.push_back(BOW_Descriptor);

		loadbar(i, n_train_images, 50);

	}

	cout << endl;
	cout << "Training Data Created" << endl;

	return training_data;

}

void Utils::applySVM(Mat training_data, Mat labels, Mat dictionary, Mat testY)
{

	cout << "Applying SVM" << endl;

	/*CvParamGrid CvParamGrid_C(pow(2.0, -5), pow(2.0, 15), pow(2.0, 2));
	CvParamGrid CvParamGrid_gamma(pow(2.0, -15), pow(2.0, 3), pow(2.0, 2));
	if (!CvParamGrid_C.check() || !CvParamGrid_gamma.check())
		cout << "The grid is NOT VALID." << endl;
	CvSVMParams paramz;
	paramz.kernel_type = CvSVM::RBF;
	paramz.svm_type = CvSVM::C_SVC;
	*/

	CvSVMParams params;

	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.C = 0.1; //C is the parameter for the soft margin cost function, which controls the influence of each individual support vector; this process involves trading error penalty for stability.
	params.gamma = 8.0;

	//params.p = 0.0; // for CV_SVM_EPS_SVR

	//params.class_weights = NULL; // for CV_SVM_C_SVC
	//params = 0.5;

	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);
	cout << "Training the SVM" << endl;

	CvSVM SVM;

	cout << training_data.size() << "     " << labels.size() << endl;

	CV_Assert(!training_data.empty() && training_data.type() == CV_32FC1);
	CV_Assert(!labels.empty() && labels.type() == CV_32FC1);

	//#pragma omp parallel for schedule(dynamic,3)
	//SVM.train_auto(training_data, labels, Mat(), Mat(), params, 10); //SVM is trainning with the images (descriptors) - experimentar trainauto

	//cout << "C : " << SVM.get_params().C << "GAMMA : " << SVM.get_params().gamma;
	SVM.train(training_data, labels, Mat(), Mat(), params);
	//SVM.train_auto(training_data, labels, Mat(), Mat(), paramz, 10, CvParamGrid_C, CvParamGrid_gamma, CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
	cout << "SVM Classifier Trained" << endl;

	//SVM.save("training_storage"); // saving
	//SVM.load("svm_filename"); // loading

	Mat image, BOW_Descriptor;
	string filename;
	vector<KeyPoint> keypoints;

	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");


	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	ofstream results;
	results.open("results.csv");

	results << "Id, name\n";

	//#pragma omp parallel for schedule(dynamic,3)
	int acc = 0, numImgs = 0;
	for (int i = 0; i < n_test_images; i++)
	{
		filename = "img_test/" + to_string(i) + ".jpg";
		if (!openImage(filename, image))
			continue;

		//Mat1b adjusted;
		//imadjust(image, adjusted);

		detector->detect(image, keypoints);

		if (keypoints.empty()) {
			results << i << "," << "cat" << "\n";
			cout << "Fail keypoints" << endl;
			continue;
		}
		else KeyPointsFilter::retainBest(keypoints, 1000);

		extractor->compute(image, keypoints, BOW_Descriptor);

		if (BOW_Descriptor.empty()) {
			results << i << "," << "cat" << "\n";
			cout << "Fail descriptor" << endl;
			continue;
		}

		bowDE.compute(image, keypoints, BOW_Descriptor);


		float res = SVM.predict(BOW_Descriptor); //retorna a label correspondente, de seguida procuramos no map o nome associado
		if (res == testY.at<double>(i)) {
			acc++;
		}
		results << i << "," << findInMap(res) << "\n";
		numImgs++;
		loadbar(i, n_test_images, 50);
	}

	printf("test acc: %f | predicted imgs: %d \n", (acc / (double)numImgs), numImgs);
	//cout << "test acc: " << (acc / numImgs) << " predicted imgs: " << (numImgs / 10000) << endl;
	results.close();
}

bool Utils::openImage(const std::string & filename, Mat & image)
{
	//cout << filename;
	Mat temp = imread(filename, 1);
	cvtColor(temp, temp, CV_BGR2GRAY);

	equalizeHist(temp, image);


	if (!image.data) {
		std::cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

void Utils::imadjust(const Mat1b& src, Mat1b& dst)
{
	// src : input CV_8UC1 image
	// dst : output CV_8UC1 imge
	// tol : tolerance, from 0 to 100.
	// in  : src image bounds
	// out : dst image buonds

	int tol = 1;
	Vec2i in = Vec2i(0, 255), out = Vec2i(0, 255);

	dst = src.clone();

	tol = max(0, min(100, tol));

	if (tol > 0)
	{
		// Compute in and out limits

		// Histogram
		vector<int> hist(256, 0);
		for (int r = 0; r < src.rows; ++r) {
			for (int c = 0; c < src.cols; ++c) {
				hist[src(r, c)]++;
			}
		}

		// Cumulative histogram
		vector<int> cum = hist;
		for (int i = 1; i < hist.size(); ++i) {
			cum[i] = cum[i - 1] + hist[i];
		}

		// Compute bounds
		int total = src.rows * src.cols;
		int low_bound = total * tol / 100;
		int upp_bound = total * (100 - tol) / 100;
		in[0] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), low_bound));
		in[1] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), upp_bound));

	}

	// Stretching
	float scale = float(out[1] - out[0]) / float(in[1] - in[0]);
	for (int r = 0; r < dst.rows; ++r)
	{
		for (int c = 0; c < dst.cols; ++c)
		{
			int vs = max(src(r, c) - in[0], 0);
			int vd = min(int(vs * scale + 0.5f) + out[0], out[1]);
			dst(r, c) = saturate_cast<uchar>(vd);
		}
	}
}

Mat Utils::parseCSV()
{

	cout << "Parsing Labels from CSV" << endl;

	ifstream file("trainLabels.csv");

	int num_files = n_train_images - fails.size();
	Mat labels;// (num_files, 1, CV_32FC1);

	// Get and drop a line
	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	string line, id;

	int what_is_it = 0, index = 1;

	while (getline(file, line) && (index - 1) != n_train_images)
	{
		std::stringstream  lineStream(line);
		std::string        cell;
		while (getline(lineStream, cell, ','))
		{

			if ((find(fails.begin(), fails.end(), index) != fails.end())) { index++; break; }

			if (what_is_it == 0) {
				what_is_it++;
			}
			else {
				labels.push_back((float)names.at(cell));
				what_is_it = 0;
				index++;
			}
		}

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