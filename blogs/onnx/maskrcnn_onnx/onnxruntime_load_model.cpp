//onnxruntime_load_model.cpp
#include <sstream>
#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <typeinfo>
#include "opencv2/opencv.hpp"
#include "load_img.h"
int main(int argc, char* argv[]) {
	cv::Mat img = read_img("./demo_resize.jpg");
	//cv::imshow("img", img);
	//cv::waitKey(0);
	std::vector<float> img_array = convert_Mat_2_vec(img);
//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

	// initialize session options if needed
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	// If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
	// session (we also need to include cuda_provider_factory.h above which defines it)
	// #include "cuda_provider_factory.h"
	// OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

	// Sets graph optimization level
	// Available levels are
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
	// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible opitmizations
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//*************************************************************************
	// create session and load model into memory
	// using squeezenet version 1.3
	// URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
	const wchar_t* model_path = L"model.onnx";
#else
	const char* model_path = "model.onnx";
#endif


	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);

//*************************************************************************
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
										   // Otherwise need vector<vector<>>

	printf("Number of inputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();

		//printf("tensor_info.GetElementCount(): %zu\n", tensor_info.GetElementCount());
		

		printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
		for (int j = 0; j < input_node_dims.size(); j++)
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
	}

		//Number of inputs = 1
		//Input 0 : name = input
		//Input 0 : type = 1
		//Input 0 : num_dims = 4
		//Input 0 : dim 0 = -1
		//Input 0 : dim 1 = 3
		//Input 0 : dim 2 = 500
		//Input 0 : dim 3 = 667

//*************************************************************************
	// Similar operations to get output node information.
	// Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
	// OrtSessionGetOutputTypeInfo() as shown above.
	// print number of model output nodes
	size_t num_output_nodes = session.GetOutputCount();
	std::vector<const char*> output_node_names(num_output_nodes);
	std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
										   // Otherwise need vector<vector<>>

	printf("Number of outputs = %zu\n", num_output_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_output_nodes; i++) {
		// print output node names
		char* output_name = session.GetOutputName(i, allocator);
		printf("Output %d : name=%s\n", i, output_name);
		output_node_names[i] = output_name;

		// print output node types
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Output %d : type=%d\n", i, type);

		// print output shapes/dims
		output_node_dims = tensor_info.GetShape();
		printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
		for (int j = 0; j < output_node_dims.size(); j++)
			printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
	}

  
//*************************************************************************
  // Score the model using sample data, and inspect values

	size_t input_tensor_size = 500 * 667 * 3;  // simplify ... using known dim values to calculate size
											   // use OrtGetTensorShapeElementCount() to get official size!
	
	//std::vector<const char*> output_node_names = { "boxes","labels","scores","3782" };
	for (int i = 0; i < 4; i++)
	{
		printf("%s\t", output_node_names[i]);
	}
	printf("\n");

	input_node_dims[0] = 1;
	printf("Input %d : num_dims=%zu\n", 0, input_node_dims.size());
	for (int j = 0; j < input_node_dims.size(); j++)
		printf("Input %d : dim %d=%jd\n", 0, j, input_node_dims[j]);

	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, img_array.data(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 4);
	assert(output_tensors.size() == 4 && output_tensors.front().IsTensor());
	printf("type of output_tensors: %s\t, size: %zu\n", 
		typeid(output_tensors).name(), output_tensors.size());
	for(int i=0;i< output_tensors.size();i++)
		printf("type of output_tensors[%d]: %s\n", i,typeid(output_tensors[i]).name());
	//std::vector<float>* boxes_ptr = output_tensors[0].GetTensorMutableData <std::vector<float>>();
	
	// print shape of all output node
	// output_node_[0:3]_shape: 15*4, 15, 15, 15*1*500*667, where 15 is the number of masks, which may be different in other cases.
	for (int i = 0; i < 4; i++)
	{
		std::vector<int64_t> output_node_i_shape 
			= output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
		printf("shape of output node %zu :\t[",i);
		for (auto j : output_node_i_shape)
		{
			printf("%d, ", j);
		}
		printf("]\n");
	}
	
	int n_predictions = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[0];
	float* boxes_ptr = output_tensors[0].GetTensorMutableData<float>();
	int* labels_ptr = output_tensors[1].GetTensorMutableData<int>();
	float* scores_ptr = output_tensors[2].GetTensorMutableData<float>();
	float* masks_ptr = output_tensors[3].GetTensorMutableData<float>();

	for (int i = 0; i < n_predictions; i++) {
		if (scores_ptr[i] > 0.4) 
		{
			printf("[%f,%f,%f,%f]\t", boxes_ptr[i*4], boxes_ptr[i * 4 +1], boxes_ptr[i * 4 +2], boxes_ptr[i * 4 +3]);
			printf("%zu\t", labels_ptr[i]);
			printf("%f\t", scores_ptr[i]);
			printf("\n");
			cv::rectangle(img,
				cv::Point(boxes_ptr[i * 4], boxes_ptr[i * 4 + 1]),
				cv::Point(boxes_ptr[i * 4 + 2], boxes_ptr[i * 4 + 3]),
				cv::Scalar(0, 255, 0));
			std::ostringstream text_str;
			text_str << "L" << labels_ptr[i] << "," << scores_ptr[i];
			cv::putText(img, text_str.str(), 
				cv::Point(boxes_ptr[i * 4]-20, boxes_ptr[i * 4 + 1]-10),
				cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0),1,8);
			auto sz = cv::Size(500, 667);
			auto mask_mat = convert_vec_2_MatC1(&masks_ptr[i*sz.area()], sz);
			printf("show img %zu\n", i);
			cv::Mat mask_3c;
			cv::cvtColor(mask_mat, mask_3c, cv::COLOR_GRAY2RGB);
			cv::addWeighted(img, 1, mask_3c, 0.2, 0, img);
		}
	}

	cv::imshow("mask", img);
	cv::waitKey(0);

	printf("\nDone!\n");
	return 0;
}

