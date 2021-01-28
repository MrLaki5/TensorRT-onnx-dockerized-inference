#include "face_detector.hpp"
#include "utils.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>

void* FaceDetector::preprocess(void* data, int width, int height)
{
    // Every color is 8b and there are 3 channels so we use CV_8UC3
    cv::Mat frame = cv::Mat(height, width, CV_8UC3, data);
    // OpenCV needs BGR and we are getting RGB so we need to switch channels
    // cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

    cv::cuda::GpuMat gpu_frame;
    // Upload image to GPU
    gpu_frame.upload(frame);

    // Load model input sizes
    std::vector<TRTEngine::Dimension> dimensions = this->_engine.get_input_dimensions();
    if (dimensions.size() < 1)
    {
        std::cout << "FaceDetector: preprocess: warning: no input dimensions" << std::endl;
        return nullptr;
    }
    int model_input_channels = dimensions[0].dimension[1];
    int model_input_height = dimensions[0].dimension[2];
    int model_input_width = dimensions[0].dimension[3];
    cv::Size input_size = cv::Size(model_input_width, model_input_height);

    // Resize image
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    // Normalize image
    cv::cuda::GpuMat flt_image;
    // Range 0 to 1
    resized.convertTo(flt_image, CV_32FC3, 1.f / 128.f);
    // Subtract mean
    cv::cuda::subtract(flt_image, cv::Scalar(127.f/128.f, 127.f/128.f, 127.f/128.f), flt_image, cv::noArray(), -1);
    // Devide std
    //cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // Allocate memory for trt engine inference input
    float* gpu_input;
    cudaMalloc(&gpu_input, model_input_channels * model_input_width * model_input_height * sizeof(float));

    // Convert image to CHW format that is input for tensorRT and copy image to trt input buffer
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < model_input_channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * model_input_width * model_input_height));
    }
    cv::cuda::split(flt_image, chw);

    return gpu_input;
}

std::vector<Rect> FaceDetector::postprocess(std::vector<TRTEngine::OutputBuffer> buffer, int width, int height)
{
    auto clip = [](float x, float y) {return (x < 0 ? 0 : (x > y ? y : x));};

    auto w_h_list = {width, height};
    int num_featuremap = 4;
    const float center_variance = 0.1;
	const float size_variance = 0.2;
    std::vector<std::vector<float>> priors = {};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    // TODO: see what should be values
    float score_threshold = 0.7;
    float iou_threshold = 0.3;

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : this->_strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(this->_strides);
    }
    
    // Generate anchors
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = width / shrinkage_size[0][index];
        float scale_h = height / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : _min_boxes[index]) {
                    float w = k / width;
                    float h = k / height;
                    priors.push_back({clip(x_center, 1.f), clip(y_center, 1.f), clip(w, 1.f), clip(h, 1.f)});
                }
            }
        }
    }

    // Convert bounding boxes
    float *score_value = (float*)(buffer[0].buffer.data());
	float *bbox_value = (float*)(buffer[1].buffer.data());
    std::vector<Rect> bbox_collection;
	for (int i = 0; i < priors.size(); i++) {
		float score = score_value[2 * i + 1];
		if (score_value[2 * i + 1] > score_threshold) {
			Rect rect;
            // Calculate rect coordinates
			float x_center = bbox_value[i * 4] * center_variance * priors[i][2] + priors[i][0];
			float y_center = bbox_value[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
			float w = exp(bbox_value[i * 4 + 2] * size_variance) * priors[i][2];
			float h = exp(bbox_value[i * 4 + 3] * size_variance) * priors[i][3];

            // Add bbox
			rect.setX((int)(clip(x_center - w / 2.0, 1) * width));
			rect.setY((int)(clip(y_center - h / 2.0, 1) * height));
			rect.setWidth((int)(clip(w, 1) * width));
			rect.setHeight((int)(clip(h, 1) * height));
            rect.setScore(clip(score_value[2 * i + 1], 1));
            bbox_collection.push_back(rect);
		}
	}

    std::vector<Rect> result_collection;
    nms(bbox_collection, result_collection, iou_threshold, NmsType::hard);

    return result_collection;
}

std::vector<Rect> FaceDetector::execute(void* data, int width, int height)
{
    void* input_buffer = this->preprocess(data, width, height);
    void* input_array[1];
    input_array[0] = input_buffer;
    std::vector<TRTEngine::OutputBuffer> output_buffer = this->_engine.inference(input_array);
    std::vector<Rect> return_vector;
    if (output_buffer.size() > 0)
    {
        return_vector = this->postprocess(output_buffer, width, height);
    }
    return return_vector;
}
