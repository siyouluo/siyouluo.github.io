#ifndef MASKRCNN_SESSION_H_
#define MASKRCNN_SESSION_H_
#include <onnxruntime_cxx_api.h>
#include <typeinfo>
#include <string>
#include "opencv2/opencv.hpp"
class MaskRCNN_Session
{
public:
	MaskRCNN_Session(const ORTCHAR_T* model_name);
	const ORTCHAR_T* Get_ModelName(void);
	std::vector<float> convert_Mat_2_vec(cv::Mat& img);
	cv::Mat read_img(const char* img_name_str);
	cv::Mat convert_vec_2_MatC1(float* vec_ptr, cv::Size sz);
	void InputNode_Config();
	void OutputNode_Config();
	void OutputNode_Config(std::vector<Ort::Value>& output_node);
	std::vector<Ort::Value> run(Ort::Session& sess, const Ort::Value* input_values);
	std::vector<Ort::Value> run(const Ort::Value* input_values);
	void OutputNode_reshape(float* boxes_ptr, 
		int* labels_ptr, float* scores_ptr, float* masks_ptr, float score_th = 0);
	size_t m_n_predictions;
	std::vector<int> m_outputlabels;
	std::vector<float> m_outputscores;
	std::vector<std::vector<float>> m_outputboxes;
	std::vector<cv::Mat> m_outputmasks;
	std::shared_ptr<Ort::MemoryInfo> m_memory_info_ptr;
	std::vector<std::vector<int64_t> > m_input_node_shape;
private:
	void Init();
	const ORTCHAR_T* m_model_name;
	Ort::Env m_env;
	Ort::SessionOptions m_session_options;
	Ort::AllocatorWithDefaultOptions m_allocator;
	std::shared_ptr<Ort::Session> m_session_ptr;
	std::vector<int64_t> m_imageshape;// shape of image: CHW, channel, height, width

	size_t m_num_input_nodes;
	std::vector<const char*> m_input_node_names;
	
	std::vector<ONNXTensorElementDataType> m_input_node_type;

	size_t m_num_output_nodes;
	std::vector<const char*> m_output_node_names;
	std::vector<std::vector<int64_t> > m_output_node_shape;
	std::vector<ONNXTensorElementDataType> m_output_node_type;
	cv::Size m_imgsize;
};
#endif
