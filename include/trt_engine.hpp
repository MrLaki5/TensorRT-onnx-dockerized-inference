#pragma once

#include <NvInfer.h>
#include <iostream>
#include <mutex>
#include <vector>

class Logger : public nvinfer1::ILogger           
{
    public:
        void log(Severity severity, const char* msg) override
        {
            // suppress info-level messages
            if (severity != Severity::kINFO)
                std::cout << msg << std::endl;
        }
};


class TRTEngine
{
    public:

        struct Dimension
        {
            std::vector<int> dimension;
        };

        struct OutputBuffer
        {
            std::vector<float> buffer;
        };

        TRTEngine();

        ~TRTEngine();

        /// Converts onnx model to tensorRT model
        /// @param input_model_file name of onnx model that will be converted
        /// @param output_model_file name of tensorRT model that will be out of conversion
        /// @return return the status of conversion
        static bool convert_onnx_to_trt_model(std::string input_model_file, std::string output_model_file);

        /// Init engine with tensorRT model
        /// @param trt_model_file name of tensorRT model that will be used in inference
        /// @return return the status of init
        bool init(std::string trt_model_file);

        /// Inference of input image with init model from buffer on CPU
        /// @param input array of pointers to image buffer loaded on CPU
        /// @param input_size size of image buffers
        /// @return return the output of inference
        std::vector<OutputBuffer> inference(float* input[], int input_sizes[]);

        /// Inference of input image with init model from buffer loaded on GPU
        /// @param input array of pointers to image buffer loaded on GPU
        /// @return return the output of inference
        std::vector<OutputBuffer> inference(void* input[]);

        /// Get input dimensions of loaded model
        /// @return return input model dimensions
        std::vector<Dimension> get_input_dimensions();

        /// Get output dimensions of loaded model
        /// @return return output model dimensions
        std::vector<Dimension> get_output_dimensions();

    private:
        Logger _gLogger;
        std::mutex _engine_mutex;
        bool _engine_init_status = false;
        nvinfer1::IRuntime* _runtime = nullptr;
        nvinfer1::ICudaEngine* _engine = nullptr;
        std::vector<int> _input_indexes;
        std::vector<int> _output_indexes;
        std::vector<nvinfer1::Dims> _input_dimensions;
        std::vector<nvinfer1::Dims> _output_dimensions;
};
