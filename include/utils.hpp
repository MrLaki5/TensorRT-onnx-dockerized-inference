#pragma once

#include "data_structures.hpp"

#include <vector>
#include <algorithm>

template <typename T>
float iou(const Rect<T>& rect1, const Rect<T>& rect2)
{
    T base_val = 0;

    // Find intersection area coordinates
    T x_a = std::max(rect1.x, rect2.x);
    T y_a = std::max(rect1.y, rect2.y);
    T x_b = std::min(rect1.x + rect1.w, rect2.x + rect2.w);
    T y_b = std::min(rect1.y + rect1.h, rect2.y + rect2.h);

    // Calculate intersection area
    float inter_area = (std::max(base_val, x_b - x_a + 1)) * (std::max(base_val, y_b - y_a + 1));

    // Calculate rect areas
    float rect1_area = (rect1.w + 1) * (rect1.h + 1);
    float rect2_area = (rect2.w + 1) * (rect2.h + 1);

    // Calculate intersection over union
    return inter_area / (rect1_area + rect2_area - inter_area);
}

template <typename T>
void hard_nms(std::vector<Rect<T>>& input, std::vector<Rect<T>>& output, const float iou_threshold)
{
    output.clear();
    if(input.empty()) return;

    std::vector<unsigned> ids(input.size());
    for(unsigned i = 0; i < ids.size(); i++)
        ids[i] = i;

	std::sort(input.begin(), input.end(), [](const Rect<T> &a, const Rect<T> &b) { return a.conf > b.conf; });

    while (ids.size() > 0) {
        int curr_id = ids[0];
        output.push_back(input[curr_id]);

        std::vector<unsigned> next_ids;
        for(auto it = ids.begin() + 1; it != ids.end(); ++it) {
            float iou_res = iou(input[curr_id], input[*it]);
            
            if (iou_res <= iou_threshold) next_ids.push_back(*it);
        }
        ids = std::move(next_ids);
    }
}
