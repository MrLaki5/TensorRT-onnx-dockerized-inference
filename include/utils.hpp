#pragma once

#include "data_structures.hpp"

#include <vector>

enum NmsType
{ 
    hard, 
    blending
};

void nms(std::vector<Rect> &input, std::vector<Rect> &output, const float& iou_threshold, NmsType nms_type);
