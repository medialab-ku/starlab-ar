#include "kf.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

int main()
{
	// print message
	std::cout << "Hello, World!" << std::endl;

	// set KinectFusion parameters
	unsigned int uiMapWidth = 640, uiMapHeight = 480;
	unsigned int uiPyramidLevel = 3;
	unsigned int uiVolumeWidth = 256, uiVolumeHeight = 256, uiVolumeDepth = 256;
	float fCenterX = 0.0f, fCenterY = 0.0f, fCenterZ = 2.0f;
	float fVoxelSize = 0.02f;

	// set TUM RGB-D dataset parameters
	kf::SIntrinsic intrinsic = { 591.1f, 590.1f, 331.0f, 234.0f };
	float fDepthScale = 1.0f / 5000.0f;

	// create frame, object, camera instance
	kf::CFrame frame(uiMapWidth, uiMapHeight, uiPyramidLevel, intrinsic);
	kf::CObject object(
		uiVolumeWidth, uiVolumeHeight, uiVolumeDepth,
		fCenterX, fCenterY, fCenterZ, fVoxelSize);
	kf::CCamera camera(uiMapWidth, uiMapHeight, intrinsic);

	// read TUM RGB-D dataset depth image list
	std::ifstream ifs("../data/tum/rgbd_dataset_freiburg1_xyz/depth.txt");
	std::string line;
	while (std::getline(ifs, line))
	{
		// skip comment line
		if (line.at(0) == '#')
		{
			continue;
		}

		// parse line
		std::istringstream iss(line);
		std::string strTime, strPath;
		if (!(iss >> strTime >> strPath))
		{
			throw std::runtime_error("invalid text file format");
		}

		// print depth image path
		std::cout << strPath << std::endl;

		// read depth map image
		auto depthImage = cv::imread(
			"../data/tum/rgbd_dataset_freiburg1_xyz/" + strPath,
			cv::IMREAD_ANYDEPTH);
		depthImage.convertTo(depthImage, CV_32F, fDepthScale);

		// preprocess by frame
		frame.clear();
		frame.preprocess(depthImage.data);

		// integrate frame to object
		object.integrate(frame);

		// raycast object from camera
		camera.clear();
		camera.raycast(object);

		// track camera with frame
		camera.track(frame);

		// print camera pose and sum of ICP error
		std::cout << camera.pose << std::endl;
		std::cout << camera.trackError << std::endl;
	}

	// exit program
	return 0;
}
