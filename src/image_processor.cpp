#include "image_processor.hpp"

#include <fstream>

bool ImageProcessor::init(nlohmann::json& config)
{
    // Check if config is valid
    if (config["onnx_model_path"] == nullptr || config["trt_model_path"] == nullptr)
    {
        std::cout << "ImageProcessor: init: error: bad config" << std::endl;
        return false;
    }

    // Check if TensorRT exported model exists
    std::ifstream f(config["trt_model_path"].dump());
    bool trt_model_exists = f.good();
    f.close();

    // If model is not exported, export it
    if (!trt_model_exists)
    {
        bool convert_status = TRTEngine::convert_onnx_to_trt_model(config["onnx_model_path"].dump(), config["trt_model_path"].dump());
        if (!convert_status)
        {
            std::cout << "ImageProcessor: init: error: conversion from onnx to trt failed" << std::endl;
            return false;
        }
    }

    // Init engine with trt model
    return this->_engine.init(config["trt_model_path"].dump());    
}
