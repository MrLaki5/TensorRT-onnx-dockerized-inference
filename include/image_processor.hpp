#pragma once

#include "trt_engine.hpp"

class ImageProcessor
{
    public:

        virtual void* preprocess(void* data, int width, int height) = 0;

        virtual void* postprocess(void* data) = 0;

    protected:
        TRTEngine _engine;
};
