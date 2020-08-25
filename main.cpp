#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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

	// exit program
	return EXIT_SUCCESS;
}
