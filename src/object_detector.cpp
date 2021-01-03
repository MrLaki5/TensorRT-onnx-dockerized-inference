#include "object_detector.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>

void* ObjectDetector::preprocess(void* data, int width, int height)
{
    // Every color is 8b and there are 3 channels so we use CV_8UC3
    cv::Mat frame = cv::Mat(height, width, CV_8UC3, data);
    // OpenCV needs BGR and we are getting RGB so we need to switch channels
    cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

    cv::cuda::GpuMat gpu_frame;
    // Upload image to GPU
    gpu_frame.upload(frame);

    // Load model input sizes
    TRTEngine::Dimensions dimensions = this->_engine.get_input_dimensions();
    cv::Size input_size = cv::Size(dimensions.width, dimensions.height);

    // Resize image
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    // Normalize image
    cv::cuda::GpuMat flt_image;
    // Range 0 to 1
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    // Subtract mean
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    // Devide std
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // Allocate memory for trt engine inference input
    float* gpu_input;
    cudaMalloc(&gpu_input, dimensions.channels * dimensions.width * dimensions.height * sizeof(float));

    // Convert image to CHW format that is input for tensorRT and copy image to trt input buffer
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < dimensions.channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * dimensions.width * dimensions.height));
    }
    cv::cuda::split(flt_image, chw);

    return gpu_input;
}

void* ObjectDetector::postprocess(void* data)
{
    return nullptr;
}
