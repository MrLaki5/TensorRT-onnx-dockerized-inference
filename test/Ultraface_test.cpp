#include "face_detector.hpp"

#include <fstream>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    nlohmann::json detector_config;
    detector_config["onnx_model_path"] = "ultraface-RFB-320.onnx";
    detector_config["trt_model_path"] = "ultraface-RFB-320.trt";

    FaceDetector face_detector;
    bool init_status = face_detector.init(detector_config);
    if (init_status)
    {
        std::cout << "Detector init successful" << std::endl;

        cv::VideoCapture cam;
        
        // Open camera
        if (!cam.open(2)) 
        {
            std::cout << "Error opening the camera" << std::endl;
            return -1;
        }

        while(true) 
        {
            cv::Mat image;
            cv::Mat image_infer;
            cam >> image;
            // Conver image to 3 chanel RGB, byte per chanel
            image.convertTo(image, CV_8UC3);
            cv::cvtColor(image, image_infer, CV_BGR2RGB);

            cv::Size s = image_infer.size();
            int image_height = s.height;
            int image_width = s.width;
            std::vector<Rect<int>> results = face_detector.execute(image_infer.data, image_width, image_height);

            // Draw bounding boxes
            for (auto result: results)
            {
                cv::Rect rect(result.x, result.y, result.w, result.h);
                cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
            }

            // Show image
            cv::imshow("Ultraface test", image);
            char c = (char)cv::waitKey(25);
            // Escape key
            if(c == 27)
            {
                break;
            }
        }
        std::cout << "Stoping camera..." << std::endl;
        cam.release();
    }
    else
    {
        std::cout << "Detector init failed" << std::endl;
    }
    return 0;
}