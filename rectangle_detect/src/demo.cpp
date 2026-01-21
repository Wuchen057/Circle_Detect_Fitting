#include "ImageProcessor.h"



int main66() {

	ImageProcessor processor;
	processor.setThresholdParams(130.0, 1000.0);

	cv::Mat img = cv::imread("data/28.png");
	std::vector<cv::Point2f> centers;
	cv::Mat result;
	bool flag = processor.extractReflectiveMarkers(img, centers, result);

	return 0;
}