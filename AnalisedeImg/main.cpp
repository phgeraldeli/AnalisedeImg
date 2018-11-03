#include "ShapeDetector.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char* argv[])
{
	const string imgpath = "t2.jpg";
	Mat image = imread(imgpath);
	if (image.empty())
	{
		return -1;
	}
	Mat gray;
	if (image.channels() == 3)
	{
		cvtColor(image, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = image.clone();
	}

	Mat blurred, thresh;
	GaussianBlur(gray, blurred, Size(5, 5), 0.0);
	threshold(blurred, thresh, 60, 255, THRESH_BINARY);

	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	ShapeDetector sd;
	vector<Point> c;
	for (size_t i = 0; i < contours.size(); i++)
	{
		c = contours[i];
		Rect crect = boundingRect(c);
		// compute the center of the contour, then detect the name of the
		// shape using only the contour
		Moments M = moments(c);
		int cX = static_cast<int>(M.m10 / M.m00);
		int cY = static_cast<int>(M.m01 / M.m00);
		sd.detect(Mat(c));
		string shape = sd.get_shape_type();
		drawContours(image, contours, i, Scalar(0, 255, 0), 2);
		Point pt(cX, cY);
		putText(image, shape, pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
		imshow("Image", image);
		
	}
	waitKey(0);
	return 0;
}