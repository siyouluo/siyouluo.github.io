#include "MaskRCNN_Session.h"

MaskRCNN_Session::MaskRCNN_Session(const ORTCHAR_T* model_name) : m_model_name(model_name)
{
	Init();
}

void MaskRCNN_Session::Init()
{
	//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info
	//Ort::Env m_env(ORT_LOGGING_LEVEL_WARNING, "test");
	m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
	// initialize session options if needed
	//Ort::SessionOptions session_options;
	m_session_options.SetIntraOpNumThreads(1);

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
	m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//*************************************************************************
	// create session and load model into memory
	// using squeezenet version 1.3
	// URL = https://github.com/onnx/models/tree/master/squeezenet
	//Ort::Session m_session(m_env, m_model_name, session_options);
	m_session_ptr = std::make_shared<Ort::Session>(m_env, m_model_name, m_session_options);
	m_memory_info_ptr = std::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	printf("Initialize MaskRCNN_Session done!\n");
}

// read a images, probably need to be resized
cv::Mat MaskRCNN_Session::read_img(const char* img_name_str)
{
	cv::Mat img = cv::imread(img_name_str);
	// resize
	return img;
}

const ORTCHAR_T* MaskRCNN_Session::Get_ModelName(void) {
	return m_model_name;
}

// reshape a Mat to vector of float
std::vector<float> MaskRCNN_Session::convert_Mat_2_vec(cv::Mat& img)
{
	int chn = img.channels();//3
	int height = img.rows;//667
	int width = img.cols;//500
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
cv::Mat MaskRCNN_Session::convert_vec_2_MatC1(float* vec_ptr, cv::Size sz)
{
	cv::Mat MatC1 = cv::Mat(sz, CV_8UC1);
	for (int h = 0; h < sz.height; h++)
	{
		for (int w = 0; w < sz.width; w++)
		{
			MatC1.data[h * sz.width + w] = vec_ptr[h * sz.width + w] * 255.0;
		}
	}
	return MatC1;
}

void MaskRCNN_Session::InputNode_Config()
{
	m_num_input_nodes = m_session_ptr->GetInputCount();
	m_input_node_names.resize(m_num_input_nodes);
	m_input_node_shape.resize(m_num_input_nodes);
	m_input_node_type.resize(m_num_input_nodes);
	for (int i = 0; i < m_num_input_nodes; i++) {
		m_input_node_names[i] = m_session_ptr->GetInputName(i, m_allocator);
		Ort::TypeInfo type_info = m_session_ptr->GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> input_node_dims = tensor_info.GetShape();
		m_input_node_shape[i] = input_node_dims;
		m_input_node_type[i] = tensor_info.GetElementType();
	}

	m_imgsize = cv::Size(m_input_node_shape[0][3], m_input_node_shape[0][2]);
}

void MaskRCNN_Session::OutputNode_Config()
{
	m_num_output_nodes = m_session_ptr->GetOutputCount();
	m_output_node_names.resize(m_num_output_nodes);
	m_output_node_shape.resize(m_num_output_nodes);
	m_output_node_type.resize(m_num_output_nodes);
	for (int i = 0; i < m_num_output_nodes; i++) {
		m_output_node_names[i] = m_session_ptr->GetOutputName(i, m_allocator);
		Ort::TypeInfo type_info = m_session_ptr->GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> output_node_dims = tensor_info.GetShape();
		m_output_node_shape[i] = output_node_dims;
		m_output_node_type[i] = tensor_info.GetElementType();
	}
}

std::vector<Ort::Value> MaskRCNN_Session::run(Ort::Session& sess,const Ort::Value* input_values)
{
	return sess.Run(Ort::RunOptions{ nullptr },
		m_input_node_names.data(), input_values, 
		m_num_input_nodes, m_output_node_names.data(), m_num_output_nodes);
}
std::vector<Ort::Value> MaskRCNN_Session::run(const Ort::Value* input_values)
{
	return m_session_ptr->Run(Ort::RunOptions{ nullptr },
		m_input_node_names.data(), input_values,
		m_num_input_nodes, m_output_node_names.data(), m_num_output_nodes);
}



void MaskRCNN_Session::OutputNode_Config(std::vector<Ort::Value>& output_node)
{
	for (int i = 0; i < m_num_output_nodes; i++) {
		m_output_node_shape[i] = output_node[i].GetTensorTypeAndShapeInfo().GetShape();;
	}
	m_n_predictions = m_output_node_shape[0][0];
}

void MaskRCNN_Session::OutputNode_reshape(float* boxes_ptr, 
	int* labels_ptr, float* scores_ptr, float* masks_ptr,float score_th)
{
	m_outputlabels.resize(m_n_predictions);
	m_outputscores.resize(m_n_predictions);
	m_outputboxes.resize(m_n_predictions);
	m_outputmasks.resize(m_n_predictions);
	for (size_t i = 0; i < m_n_predictions; i++) 
	{
		m_outputboxes[i].resize(4);
		m_outputboxes[i][0] = boxes_ptr[i * 4];
		m_outputboxes[i][1] = boxes_ptr[i * 4+1];
		m_outputboxes[i][2] = boxes_ptr[i * 4+2];
		m_outputboxes[i][3] = boxes_ptr[i * 4+3];
		m_outputscores[i] = scores_ptr[i];
		m_outputlabels[i] = labels_ptr[i];
		m_outputmasks[i] = convert_vec_2_MatC1(&masks_ptr[i*m_imgsize.area()], m_imgsize);
	}
}
