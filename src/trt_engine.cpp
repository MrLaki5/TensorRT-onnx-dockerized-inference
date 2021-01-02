#include "trt_engine.hpp"

#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>

TRTEngine::TRTEngine()
{}

TRTEngine::~TRTEngine()
{
    if (this->_engine)
    {
        this->_engine->destroy();
        this->_engine = nullptr;
    }
    if (this->_runtime)
    {
        this->_runtime->destroy();
        this->_runtime = nullptr;
    }
}

std::size_t get_size_by_dim(const nvinfer1::Dims& dims)
{
    std::size_t size = 1;
    for (std::size_t i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }
    return size;
}


std::vector<float> TRTEngine::inference(float* image, int image_size)
{
    // Create execution context
    this->_engine_mutex.lock();
    nvinfer1::IExecutionContext* context = this->_engine->createExecutionContext();
    this->_engine_mutex.unlock();

    // Create buffers
    std::vector<void*> buffers(2);

    // Allocate GPU buffers for input and output
    cudaMalloc(&buffers[this->_input_index], get_size_by_dim(this->_input_dimensions) * sizeof(float));
    cudaMalloc(&buffers[this->_output_index], get_size_by_dim(this->_output_dimensions) * sizeof(float));

    // Copy input image from CPU to GPU
    cudaMemcpy(buffers[this->_input_index], image, image_size, cudaMemcpyHostToDevice);

    // Do inference
    bool inference_status = context->executeV2(buffers.data());
    if (!inference_status)
    {
        // Free memory
        cudaFree(buffers[this->_input_index]);
        cudaFree(buffers[this->_output_index]);
        context->destroy();

        return std::vector<float>();
    }
    
    // Copy inference results from GPU to CPU
    std::vector<float> cpu_output(get_size_by_dim(this->_output_dimensions));
    cudaMemcpy(cpu_output.data(), (float*)buffers[this->_output_index], cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(buffers[this->_input_index]);
    cudaFree(buffers[this->_output_index]);
    context->destroy();

    return cpu_output;
}

bool TRTEngine::init(std::string trt_model_file)
{
    const std::lock_guard<std::mutex> lock(this->_engine_mutex);
    if (this->_engine_init_status)
        return false;
    
    // Create runtime
    this->_runtime = nvinfer1::createInferRuntime(gLogger);
    if (!this->_runtime)
    {
        std::string msg = "Failed to create runtime";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        return false;
    }

    // Load data from file
    std::ifstream p(trt_model_file);  
    p.seekg( 0, std::ios::end );  
    size_t model_size = p.tellg();  
    char *model_data = new char[model_size];  
    p.seekg(0, std::ios::beg);   
    p.read(model_data, model_size);  
    p.close();

    // Create engine
    this->_engine = this->_runtime->deserializeCudaEngine(model_data, model_size, nullptr);
    if (!this->_engine)
    {
        std::string msg = "Failed to create engine";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        this->_runtime->destroy();
        this->_runtime = nullptr;
        delete model_data;
        return false;
    }

    // Get input and output buffer indexes
    this->_input_index = 0;
    this->_output_index = 1;
    if (!this->_engine->bindingIsInput(0))
    {
        this->_input_index = 1;
        this->_output_index = 0;
    }

    // Get input and output buffer dimensions
    this->_input_dimensions = this->_engine->getBindingDimensions(this->_input_index);
    this->_output_dimensions = this->_engine->getBindingDimensions(this->_output_index);

    delete model_data;
    this->_engine_init_status = true;
    return true;
}

TRTEngine::Dimensions TRTEngine::get_input_dimensions()
{
    TRTEngine::Dimensions ret_dimensions;

    ret_dimensions.width = this->_input_dimensions.d[2];
    ret_dimensions.height = this->_input_dimensions.d[1];
    ret_dimensions.channels = this->_input_dimensions.d[0];

    return ret_dimensions;
}

TRTEngine::Dimensions TRTEngine::get_output_dimensions()
{
    TRTEngine::Dimensions ret_dimensions;

    ret_dimensions.width = this->_output_dimensions.d[2];
    ret_dimensions.height = this->_output_dimensions.d[1];
    ret_dimensions.channels = this->_output_dimensions.d[0];

    return ret_dimensions;
}

bool TRTEngine::convert_onnx_to_trt_model(std::string input_model_file, std::string output_model_file)
{
    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    if (!builder)
    {
        std::string msg = "Failed to create builder";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        return false; 
    }

    // Create network definition
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network)
    {
        std::string msg = "Failed create network definition";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        builder->destroy();
        return false; 
    }

    // Create ONNX parser and parse network
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser || !parser->parseFromFile(input_model_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR)))
    {
        std::string msg = "Failed to parse onnx file";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // Create engine from parsed network
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // Maximum size that any layer in network can use
    config->setMaxWorkspaceSize(1 << 20);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        std::string msg = "Failed to create engine from parsed network";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();
        return false;
    }

    // Free memory (can be deleted after creation of engine)
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

    // Serialize engine and store it to file
    nvinfer1::IHostMemory* serialized_engine = engine->serialize();
    std::ofstream p(output_model_file.c_str());
    p.write((const char*)serialized_engine->data(),serialized_engine->size());
    p.close();

    // Free memory
    serialized_engine->destroy();
    engine->destroy();

    return true;
}
