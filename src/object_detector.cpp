#include "object_detector.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

void* ObjectDetector::preprocess(void* data, int width, int height)
{
    cv::Mat frame;

    cv::cuda::GpuMat gpu_frame;
    // Upload image to GPU
    gpu_frame.upload(frame);

    auto input_size = cv::Size(width, height);
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

    return nullptr;
}

void* ObjectDetector::postprocess(void* data)
{
    return nullptr;
}
