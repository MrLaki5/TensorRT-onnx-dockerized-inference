#pragma once

#include "image_processor.hpp"
#include "data_structures.hpp"

class FaceDetector: public ImageProcessor
{
    public:

        std::vector<Rect<int>> execute(void* data, int width, int height);

        void* preprocess(void* data, int width, int height);

        std::vector<Rect<int>> postprocess(std::vector<TRTEngine::OutputBuffer> buffer, int width, int height);

    private:
        const std::vector<float> _strides = {8.0, 16.0, 32.0, 64.0};

        const std::vector<std::vector<float>> _min_boxes = {{10.0f,  16.0f,  24.0f},
                                                           {32.0f,  48.0f},
                                                           {64.0f,  96.0f},
                                                           {128.0f, 192.0f, 256.0f}};
};
