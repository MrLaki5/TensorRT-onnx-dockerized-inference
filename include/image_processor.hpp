#pragma once

#include "trt_engine.hpp"
#include "json.hpp"

class ImageProcessor
{
    public:

        virtual void* preprocess(void* data, int width, int height) = 0;

        virtual std::vector<float> postprocess(std::vector<TRTEngine::OutputBuffer> buffer) = 0;

        bool init(nlohmann::json& config);

    protected:
        TRTEngine _engine;
};
