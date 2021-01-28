#!/bin/bash

devices=""
for device in /dev/video*; do
    devices="${devices} --device ${device}"
done

xhost local:root
docker run -it --rm --gpus all ${devices} -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name tensorrt-engine-test tensorrt-engine-test:latest /bin/bash
