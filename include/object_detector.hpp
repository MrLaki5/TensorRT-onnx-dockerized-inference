#pragma once

#include "image_processor.hpp"

class ObjectDetector: public ImageProcessor
{
    public:

        void* preprocess(void* data, int width, int height) override;

        void* postprocess(void* data) override;
};
