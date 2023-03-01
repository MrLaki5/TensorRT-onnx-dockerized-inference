# TensorRT-onnx-dockerized-inference
* <b>TensorRT</b> engine inference with <b>ONNX model conversion</b>
* <b>Dockerized</b> environment with: CUDA 10.2, TensorRT 7, OpenCV 3.4 built with CUDA
* <b>ResNet50</b> preprocessing and postprocessing implementation
* <b>Ultraface</b> preprocessing and postprocessing implementation

## Requirements
* [docker](https://docs.docker.com/get-docker/)
* [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Build
### Pull docker image
* Pull container image from the repo packages
``` bash
docker pull ghcr.io/mrlaki5/tensorrt-onnx-dockerized-inference:latest
```

### Build docker image from sources
* Download TensorRT 7 installation from [link](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.1/local_repos/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb)
* Place downloaded TensorRT 7 deb file into root dir of this repo
* Build
``` bash
cd ./docker
./build.sh
```

## Run
From the root of the repo start docker container with the command below
``` bash
./docker/run.sh
```
### ResNet50 inference test
``` bash
./ResNet50_test
```
* [Input image](https://pixabay.com/photos/cat-siamese-cat-fur-kitten-2068462)
* Output: Siamese cat, Siamese (confidence: 0.995392)
<img src="img/cat.jpg" width="400"/><img src="img/out.png" width="400"/>

### Ultraface detector inference test
* Note: for this test, camera device is required. Test will start GUI showing camera stream overlaped with face detections.
``` bash
./Ultraface_test
```
<img src="img/face.png" width="400"/>
