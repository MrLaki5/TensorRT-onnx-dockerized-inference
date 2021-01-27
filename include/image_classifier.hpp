#pragma once

#include "image_processor.hpp"

class ImageClassifier: public ImageProcessor
{
    public:

        std::vector<float> execute(void* data, int width, int height);

        void* preprocess(void* data, int width, int height);

        std::vector<float> postprocess(std::vector<TRTEngine::OutputBuffer> buffer);
};
