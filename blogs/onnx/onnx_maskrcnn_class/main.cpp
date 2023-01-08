#include <iostream>
#include <sstream>
#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <typeinfo>
#include "opencv2/opencv.hpp"
#include "MaskRCNN_Session.h"
int main(int argc, char* argv[]) {

#ifdef _WIN32
	const wchar_t* model_path = L"./model_files/model_100.onnx";
#else
	const char* model_path = "./model_files/model_100.onnx";
#endif

	MaskRCNN_Session Model_Session(model_path);// read model file
	Model_Session.InputNode_Config();// configure input node infomation based on model file
	Model_Session.OutputNode_Config();// configure output node infomation based on model file
	cv::Mat img = Model_Session.read_img("./images/51_Color.png");
	printf("Using Onnxruntime C++ API\n");

	// reshape img to 1d-array & create input tensor object 
	std::vector<float> img_array = Model_Session.convert_Mat_2_vec(img);
	Model_Session.m_input_node_shape[0][0] = 1;
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(*Model_Session.m_memory_info_ptr, 
		img_array.data(), img_array.size(), 
		Model_Session.m_input_node_shape[0].data(), Model_Session.m_input_node_shape[0].size());
	
	// Inference
	auto output_tensors = Model_Session.run(&input_tensor);
	assert(output_tensors.size() == 4 && output_tensors.front().IsTensor());
	
	// reshape output tensor to vectors and matrices
	Model_Session.OutputNode_Config(output_tensors);
	float* boxes_ptr = output_tensors[0].GetTensorMutableData<float>();
	int* labels_ptr = output_tensors[1].GetTensorMutableData<int>();
	float* scores_ptr = output_tensors[2].GetTensorMutableData<float>();
	float* masks_ptr = output_tensors[3].GetTensorMutableData<float>();
	Model_Session.OutputNode_reshape(boxes_ptr, labels_ptr, scores_ptr, masks_ptr);

	// print and visualization
	for (size_t i = 0; i < Model_Session.m_n_predictions; i++)
	{
		if (Model_Session.m_outputscores[i] > 0.8) {

			std::cout << Model_Session.m_outputlabels[i] << ",\t";
			std::cout << "[";
			for (const auto& b : Model_Session.m_outputboxes[i]) {
				std::cout << b << "\t";
			}
			std::cout << "],\t";
			std::cout << Model_Session.m_outputscores[i] << std::endl;

			auto leftup_point = cv::Point(Model_Session.m_outputboxes[i][0], Model_Session.m_outputboxes[i][1]);
			auto rightdown_point = cv::Point(Model_Session.m_outputboxes[i][2], Model_Session.m_outputboxes[i][3]);
			cv::rectangle(img, leftup_point, rightdown_point, cv::Scalar(0, 255, 0));
			std::ostringstream text_str;
			text_str << "L" << labels_ptr[i] << "," << scores_ptr[i];
			cv::putText(img, text_str.str(),leftup_point,
				cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, 8);
			cv::Mat mask_3c;
			cv::cvtColor(Model_Session.m_outputmasks[i], mask_3c, cv::COLOR_GRAY2RGB);
			cv::addWeighted(img, 1, mask_3c, 1, 0, img);
		}
	}
	cv::imshow("mask", img);
	cv::waitKey(0);

	printf("\nDone!\n");
	return 0;
}