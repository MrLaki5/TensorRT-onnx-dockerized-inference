#pragma once

class Point2D
{
    private:
        int _x;
        int _y;

    public:

        Point2D();

        Point2D(const int& x, const int& y);

        int getX() const;

        int getY() const;

        void setX(const int& x);

        void setY(const int& y);
};

class Rect
{
    private:
        Point2D _top_left_corner;
        int _width;
        int _height;

    public:

        Rect();

        Rect(const int& x, const int& y, const int& width, const int& height);

        Rect(const Point2D& top_left_corner, const int& width, const int& height);

        Point2D getTopLeftCorner() const;

        int getX() const;

        int getY() const;

        int getWidth() const;

        int getHeight() const;

        void setTopLeftCorner(const Point2D& top_left_corner);

        void setX(const int& x);

        void setY(const int& y);

        void setWidth(const int& width);

        void setHeight(const int& height);
};
