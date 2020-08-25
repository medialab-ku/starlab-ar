#include <cstdlib>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main()
{
	// print test message
	cout << "Hello World!" << endl;

	// read test image
	const auto image = imread("test.bmp");

	// show test image
	imshow("Test Image", image);

	// close all windows
	waitKey(0);
	destroyAllWindows();

	// read labels
	vector<string> labels;
	{
		ifstream ifs("mscoco_labels.txt");
		string line;
		while (getline(ifs, line))
			labels.push_back(line);
	}

	// read colors
	vector<Scalar> colors;
	{
		ifstream ifs("instance_colors.txt");
		string line;
		while (getline(ifs, line)) {
			stringstream ss(line);
			int r, g, b;
			ss >> r;
			ss >> g;
			ss >> b;
			colors.push_back(Scalar(r, g, b, 255));
		}
	}

	// exit program
	return EXIT_SUCCESS;
}
