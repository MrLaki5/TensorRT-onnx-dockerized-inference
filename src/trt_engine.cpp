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

std::vector<TRTEngine::OutputBuffer> TRTEngine::inference(float* input[], int input_sizes[])
{
    std::vector<void*> input_buffers(this->_input_dimensions.size());
 
    // Allocate GPU buffers for input
    for (int i = 0; i < this->_input_dimensions.size(); i++)
    {
        cudaMalloc(&input_buffers[i], get_size_by_dim(this->_input_dimensions[i]) * sizeof(float));
        // Copy input image from CPU to GPU
        cudaMemcpy(&input_buffers[i], input[i], input_sizes[i], cudaMemcpyHostToDevice);
    }
    
    // Do inference
    return this->inference(input_buffers.data());
}

std::vector<TRTEngine::OutputBuffer> TRTEngine::inference(void* input[])
{
    // Create execution context
    this->_engine_mutex.lock();
    nvinfer1::IExecutionContext* context = this->_engine->createExecutionContext();
    this->_engine_mutex.unlock();

    // Create buffers
    std::vector<void*> buffers(this->_engine->getNbBindings());
    
    // Map input buffers
    for (int i = 0; i < this->_input_indexes.size(); i++)
    {
        buffers[this->_input_indexes[i]] = input[i];
    }

    // Allocate GPU buffers for output
    for (int i = 0; i < this->_output_dimensions.size(); i++)
    {
        cudaMalloc(&buffers[this->_output_indexes[i]], get_size_by_dim(this->_output_dimensions[i]) * sizeof(float));
    }

    // Do inference
    bool inference_status = context->executeV2(buffers.data());
    if (!inference_status)
    {
        // Free memory
        for (int i=0; i < this->_engine->getNbBindings(); i++)
        {
            cudaFree(buffers[i]);
        }
        context->destroy();

        return std::vector<TRTEngine::OutputBuffer>();
    }
    
    // Copy inference results from GPU to CPU
    std::vector<TRTEngine::OutputBuffer> cpu_result(this->_output_dimensions.size());
    for (int i = 0; i < this->_output_dimensions.size(); i++)
    {
        std::vector<float> cpu_output(get_size_by_dim(this->_output_dimensions[i]));
        cudaMemcpy(cpu_output.data(), (float*)buffers[this->_output_indexes[i]], cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cpu_result[i].buffer = cpu_output;
    }

    // Free memory
    for (int i=0; i < this->_engine->getNbBindings(); i++)
    {
        cudaFree(buffers[i]);
    }
    context->destroy();

    return cpu_result;
}

bool TRTEngine::init(std::string trt_model_file)
{
    const std::lock_guard<std::mutex> lock(this->_engine_mutex);
    if (this->_engine_init_status)
        return false;
    
    // Create runtime
    this->_runtime = nvinfer1::createInferRuntime(this->_gLogger);
    if (!this->_runtime)
    {
        std::string msg = "Failed to create runtime";
        this->_gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
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
        this->_gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        this->_runtime->destroy();
        this->_runtime = nullptr;
        delete model_data;
        return false;
    }

    // Get input and output buffer indexes and dimensions
    for (size_t i = 0; i < this->_engine->getNbBindings(); i++)
    {
        if (this->_engine->bindingIsInput(i))
        {
            this->_input_dimensions.emplace_back(this->_engine->getBindingDimensions(i));
            this->_input_indexes.emplace_back(i);
        }
        else
        {
            this->_output_dimensions.emplace_back(this->_engine->getBindingDimensions(i));
            this->_output_indexes.emplace_back(i);
        }
    }

    delete model_data;
    this->_engine_init_status = true;
    return true;
}

std::vector<TRTEngine::Dimension> TRTEngine::get_input_dimensions()
{
    std::vector<TRTEngine::Dimension> ret_dimensions;

    for (int i = 0; i < this->_input_dimensions.size(); ++i)
    {
        TRTEngine::Dimension input_dimension;
        for (std::size_t j = 0; j < this->_input_dimensions[i].nbDims; j++)
        {
            input_dimension.dimension.emplace_back(this->_input_dimensions[i].d[j]);
        }
        ret_dimensions.emplace_back(input_dimension);
    }

    return ret_dimensions;
}

std::vector<TRTEngine::Dimension> TRTEngine::get_output_dimensions()
{
    std::vector<TRTEngine::Dimension> ret_dimensions;

    for (int i = 0; i < this->_output_dimensions.size(); ++i)
    {
        TRTEngine::Dimension output_dimension;
        for (std::size_t j = 0; j < this->_output_dimensions[i].nbDims; j++)
        {
            output_dimension.dimension.emplace_back(this->_output_dimensions[i].d[j]);
        }
        ret_dimensions.emplace_back(output_dimension);
    }

    return ret_dimensions;
}

bool TRTEngine::convert_onnx_to_trt_model(std::string input_model_file, std::string output_model_file)
{
    // Create logger
    Logger gLogger;

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
    config->setMaxWorkspaceSize(1ULL << 30);
    // Set max batch size to one image
    builder->setMaxBatchSize(1);
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
