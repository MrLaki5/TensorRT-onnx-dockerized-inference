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

std::vector<Rect<int>> FaceDetector::postprocess(std::vector<TRTEngine::OutputBuffer> buffer, int width, int height)
{
    int boxes_dim = 4;
    int conf_dim = 2;
    int size = 4420;
    float nms_threshold = 0.5;
    float threshold = 0.7;

    float *conf_data = (float*)(buffer[0].buffer.data());
	float *boxes_data = (float*)(buffer[1].buffer.data());

    std::vector<Rect<float>> input_anchors;
    for (int i = 0; i < size; i++) {
        if (conf_data[i * conf_dim + 1] < threshold) continue;

        input_anchors.emplace(input_anchors.end(), boxes_data[i * boxes_dim], 
                                                   boxes_data[i * boxes_dim + 1],
                                                   boxes_data[i * boxes_dim + 2] - boxes_data[i * boxes_dim],
                                                   boxes_data[i * boxes_dim + 3] - boxes_data[i * boxes_dim + 1],
                                                   conf_data[i * conf_dim + 1]);
    }

    std::vector<Rect<float>> nms_anchors;
    hard_nms<float>(input_anchors, nms_anchors, nms_threshold);

    std::vector<Rect<int>> results;
    std::transform(nms_anchors.begin(), nms_anchors.end(), std::back_inserter(results),
                   [width, height](const Rect<float>& detection) {
                       return Rect<int>(static_cast<int>(detection.x * width),
                                        static_cast<int>(detection.y * height),
                                        static_cast<int>(detection.w * width),
                                        static_cast<int>(detection.h * height),
                                        detection.conf);
                   });

    return results;
}

std::vector<Rect<int>> FaceDetector::execute(void* data, int width, int height)
{
    void* input_buffer = this->preprocess(data, width, height);
    void* input_array[1];
    input_array[0] = input_buffer;
    std::vector<TRTEngine::OutputBuffer> output_buffer = this->_engine.inference(input_array);
    std::vector<Rect<int>> return_vector;
    if (output_buffer.size() > 0)
    {
        return_vector = this->postprocess(output_buffer, width, height);
    }
    return return_vector;
}
