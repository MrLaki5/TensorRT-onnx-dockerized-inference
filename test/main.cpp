#include "image_classifier.hpp"

#include <fstream>

int main()
{
    nlohmann::json detector_config;
    detector_config["onnx_model_path"] = "resnet50-v1-7.onnx";
    detector_config["trt_model_path"] = "resnet50-v1-7.trt";

    ImageClassifier image_classifier;
    bool init_status = image_classifier.init(detector_config);
    if (init_status)
    {
        std::cout << "Detector init successful" << std::endl;

        int image_width = 1280;
        int image_height = 853;
        // Load image from file
        std::ifstream p("cat.rgb");  
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

        std::vector<float> result = image_classifier.execute(image_data, image_width, image_height);
        for (int i=0; i < result.size(); i++)
        {
            std::cout << "Class " << i << ": " << result[i] << std::endl;
        }

        delete image_data;
    }
    else
    {
        std::cout << "Detector init failed" << std::endl;
    }
    return 0;
}