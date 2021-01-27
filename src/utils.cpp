#include "utils.hpp"

#include <iostream>
#include <math.h>
#include <algorithm>

void nms(std::vector<Rect> &input, std::vector<Rect> &output, const float& iou_threshold, NmsType nms_type) {
	std::sort(input.begin(), input.end(), [](const Rect &a, const Rect &b) { return a.getScore() > b.getScore(); });

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++) {
		if (merged[i])
			continue;
		std::vector<Rect> buf;

		buf.push_back(input[i]);
		merged[i] = 1;

		for (int j = i + 1; j < box_num; j++) {
			if (merged[j])
				continue;

			float score = input[i].iou(input[j]);

			if (score > iou_threshold) {
				merged[j] = 1;
				buf.push_back(input[j]);
			}
		}
		switch (nms_type) {
            case NmsType::hard: {
                output.push_back(buf[0]);
                break;
            }
            case NmsType::blending: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].getScore());
                }
                Rect rects(0, 0, 0, 0, 0);
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].getScore()) / total;
                    rects.setX(buf[i].getX() * rate + rects.getX());
                    rects.setY(buf[i].getY() * rate + rects.getY());
                    rects.setWidth(buf[i].getWidth() * rate + rects.getWidth());
                    rects.setHeight(buf[i].getHeight() * rate + rects.getHeight());
                    rects.setScore(buf[i].getScore() * rate);
                }
                output.push_back(rects);
                break;
            }
            default: {
                std::cout << "Utils: nms: Wrong type of nms" << std::endl;
                exit(-1);
            }
		}
	}
}