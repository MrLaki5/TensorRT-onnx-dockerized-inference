FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

COPY ./nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb .

RUN apt update && apt -y install cmake

# Install tensorrt
RUN dpkg -x nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb /tmpLocal && \
    apt-get install -y --allow-downgrades --allow-change-held-packages /tmpLocal/var/nv-tensorrt-repo-cuda10.2-trt7.2.1.6-ga-20201006/*.deb && \
    # cleanup
    rm nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb

# Build project
RUN mkdir wspace
COPY . ./wspace
RUN mkdir build
WORKDIR ./build
RUN cmake ../wspace
RUN make install
