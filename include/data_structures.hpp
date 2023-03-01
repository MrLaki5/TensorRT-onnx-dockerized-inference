#pragma once

template <typename T>
struct Rect
{
        T x;
        T y;
        T w;
        T h;
        float conf;

        Rect(T in_x, T in_y, T in_w, T in_h, float in_conf): x(in_x), y(in_y), w(in_w), h(in_h), conf(in_conf) {}
};
