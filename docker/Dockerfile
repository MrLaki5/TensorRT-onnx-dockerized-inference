FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

COPY ./nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb .

RUN apt update && apt -y install cmake wget unzip ffmpeg zlib1g-dev libv4l-dev

# Install tensorrt
RUN dpkg -x nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb /tmpLocal && \
    apt-get install -y --allow-downgrades --allow-change-held-packages /tmpLocal/var/nv-tensorrt-repo-cuda10.2-trt7.2.1.6-ga-20201006/*.deb && \
    # cleanup
    rm nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb

# Install OpenCV with cuda support
RUN mkdir -p /tmp && \
    cd /tmp && \
    wget --no-check-certificate -O opencv.zip https://github.com/opencv/opencv/archive/3.4.0.zip && \
    unzip opencv.zip && \
    mkdir opencv-3.4.0/build && \
    cd opencv-3.4.0/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D WITH_CUDA=ON \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -D WITH_TBB=ON \
        .. && \
    make -j$(nproc) && \
    make install && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf && \
    ldconfig && \
    cd /tmp && \
    rm -rf opencv-3.4.0 opencv.zip && \
    cd /

# Build project
RUN mkdir wspace
COPY . ./wspace
RUN mkdir build
WORKDIR ./build
RUN cmake ../wspace
RUN make install
