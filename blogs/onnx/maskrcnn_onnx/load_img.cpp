// load_img.cpp
#include "load_img.h"

// read a images, probably need to be resized
cv::Mat read_img(const char* img_name_str)
{
	cv::Mat img = cv::imread(img_name_str);
	// resize
	return img;
}

// reshape a Mat to vector of float
std::vector<float> convert_Mat_2_vec(cv::Mat& img)
{
	int chn = img.channels();//3
	int width = img.cols;//500
	int height = img.rows;//667
	std::vector<float> vector_1d(chn*width*height);
	for (int c = 0; c < chn; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				vector_1d[c*width*height + h * width + w] = img.at<cv::Vec3b>(h, w)[2 - c] / 255.0;
			}
		}
	}
	return vector_1d;
}

// reshape a vector of float to Mat image with 1 channel(CV_8UC1)
cv::Mat convert_vec_2_MatC1(float* vec_ptr, cv::Size sz)
{
	cv::Mat MatC1 = cv::Mat(sz, CV_8UC1);
	for (int h = 0; h < sz.height; h++)
	{
		for (int w = 0; w < sz.width; w++)
		{
			MatC1.data[h * sz.width + w] = vec_ptr[ h * sz.width + w] * 255.0;
		}
	}
	return MatC1;
}
