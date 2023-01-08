//main.cpp
#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <typeinfo>
int main(int argc, char* argv[]) {
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
		//Input 0 : name = input_x
		//Input 0 : type = 1
		//Input 0 : num_dims = 1
		//Input 0 : dim 0 = -1

//*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.
		// print number of model input nodes
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

	size_t input_tensor_size = 1;  // simplify ... using known dim values to calculate size
											   // use OrtGetTensorShapeElementCount() to get official size!

	std::vector<float> input_tensor_values;
	//std::vector<const char*> output_node_names = { "boxes","labels","scores","3782" };
	for (int i = 0; i < num_output_nodes; i++)
	{
		printf("output name [%zu]: %s\t", i,output_node_names[i]);
	}
	printf("\n");

	input_node_dims[0] = 6;//-1
	printf("Input %d : num_dims=%zu\n", 0, input_node_dims.size());
	for (int j = 0; j < input_node_dims.size(); j++)
		printf("Input %d : dim %d=%jd\n", 0, j, input_node_dims[j]);

	// initialize input data with values in [0.0, 1.0]
	// 1 tensor, with shape = input_node_dims =[6,1]
	input_tensor_size *= input_node_dims[0] * input_node_dims[1];
	for (unsigned int i = 0; i < input_tensor_size; i++)
	{
		input_tensor_values.push_back((float)i / (input_tensor_size + 1));
	}
	input_tensor_values[0] = 0.5;
	printf("input_tensor_values:\n");
	for (unsigned int i = 0; i < input_tensor_size; i++)
		printf("%f\n", input_tensor_values[i]);

	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	//Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	printf("type of output_tensors: %s\t, size: %zu\n",
		typeid(output_tensors).name(), output_tensors.size());
	for (int i = 0; i < output_tensors.size(); i++)
	{
		printf("type of output_tensors[%d]: %s\n", i, typeid(output_tensors[i]).name());
		std::vector<int64_t> output_node_i_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
		printf("size of output_tensors[%zu]: %zu\n",i, output_node_i_shape.size());
		printf("shape of output_tensors[%zu]: [", i);
		for (int j = 0; j < output_node_i_shape.size(); j++)
		{
			printf("%zu\t", output_node_i_shape[j]);
		}
		printf("]\n");

		float* output_y = output_tensors[i].GetTensorMutableData<float>();
		printf("output_y:\n");
		for (int j = 0; j < output_node_i_shape[0]; j++)
		{
			for(int k=0;k<output_node_i_shape[1];k++){
				printf("%f\t", output_y[j*output_node_i_shape[1]+k]);
			}
			printf("\n");
		}

	}

	printf("\nDone!\n");
	return 0;
}
