#pragma once

#include "trt_engine.hpp"
#include "json.hpp"

class ImageProcessor
{
    public:

        bool init(nlohmann::json& config);

    protected:
        TRTEngine _engine;
};
