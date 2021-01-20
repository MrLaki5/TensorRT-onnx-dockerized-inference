## TensorRT-onnx-dockerized-inference
* <b>TensorRT</b> engine inference with <b>ONNX model conversion</b>
* <b>Dockerized</b> environment with: CUDA 10.2, TensorRT 7, OpenCV 3.4 built with CUDA
* <b>ResNet50</b> preprocessing and postprocessing implementation

### Quick docker setup:
* Download TensorRT 7 installation from [link](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.1/local_repos/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb)
* Place downloaded deb file into root dir of this repo
* Run:
``` bash
cd ./docker
./build.sh
./run.sh
```

### ResNet50 inference test

``` bash
./ResNet50_test
```
* [Input image](https://pixabay.com/photos/cat-siamese-cat-fur-kitten-2068462)
* Output: Siamese cat, Siamese (confidence: 0.995392)

<img src="img/cat.jpg" width="400"/><img src="img/out.png" width="400"/>
