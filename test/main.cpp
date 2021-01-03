#include "object_detector.hpp"

#include <fstream>

int main()
{
    bool convert_status = TRTEngine::convert_onnx_to_trt_model("retinanet-9.onnx", "retinanet-9.trt");
    if (convert_status)
    {
        std::cout << "Model converted!" << std::endl;

        TRTEngine engine;
        ObjectDetector object_detector;
        bool init_status = engine.init("retinanet-9.trt");
        if (init_status)
        {
            std::cout << "Engine init successful" << std::endl;

            int image_width = 1280;
            int image_height = 720;
            // Load image from file
            std::ifstream p("test.rgb");  
            p.seekg( 0, std::ios::end );  
            size_t image_size = p.tellg();  
            char* image_data = new char[image_size];  
            p.seekg(0, std::ios::beg);   
            p.read(image_data, image_size);  
            p.close();

            //for(size_t i=0; i < image_size; i++)
            //{
            //    unsigned char temp_var = image_data[i];
            //    std::cout << (int)(temp_var) << std::endl;
            //}

            object_detector.preprocess(image_data, image_width, image_height);

            delete image_data;
        }
        else
        {
            std::cout << "Engine init failed" << std::endl;
        }
    }
    else
    {
        std::cout << "Failed to convert model!" << std::endl;
    }
    return 0;
}