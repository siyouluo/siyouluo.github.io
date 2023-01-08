//load_img.h
#ifndef _LOAD_IMG_H
#define _LOAD_IMG_H
#include "opencv2/opencv.hpp"
extern cv::Mat read_img(const char* img_name_str);
extern std::vector<float> convert_Mat_2_vec(cv::Mat& img);
extern cv::Mat convert_vec_2_MatC1(float* vec_ptr, cv::Size sz);
#endif // !_LOAD_IMG_H
