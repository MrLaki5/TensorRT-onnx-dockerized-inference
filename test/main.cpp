#include "trt_engine.hpp"
#include "object_detector.hpp"


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
            std::cout << "Engin init successful" << std::endl;
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