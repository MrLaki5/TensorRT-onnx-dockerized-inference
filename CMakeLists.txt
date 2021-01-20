cmake_minimum_required(VERSION 3.1)

project(tensorrt-engine)

#================================================================================
# Set compiler
#================================================================================
set(CMAKE_CXX_FLAGS "-std=c++11 -g -O3 -fPIC")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

#================================================================================
# Build options
#================================================================================
option(BUILD_RESNET50_TEST "Build test binary for ResNet50 model" OFF)

#================================================================================
# Find packages
#================================================================================
find_package(CUDA)
find_package(OpenCV)

#================================================================================
# Build tensorrt-engine lib
#================================================================================
set(LIB "tensorrt-engine")
# Cuda link libs dir
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
# Store all cpp files into SRC variable
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# Lib
add_library(${LIB} SHARED ${SRC})
# Include header files
target_include_directories(${LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                  PUBLIC ${CUDA_INCLUDE_DIRS}
                                  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ext/pkg-json)
# Link libs
target_link_libraries(${LIB} PRIVATE cudart 
                             PRIVATE nvinfer 
                             PRIVATE nvonnxparser
                             PRIVATE ${OpenCV_LIBS})

#================================================================================
# Build engine tests
#================================================================================
if(BUILD_RESNET50_TEST)
    set(BIN "ResNet50_test")
    # Create executable
    add_executable(${BIN} ${CMAKE_CURRENT_SOURCE_DIR}/test/ResNet50_test.cpp)
    # Link libs
    target_link_libraries(${BIN} PRIVATE ${LIB})
    # Include header files
    target_include_directories(${BIN} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    # Install test dataset
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/cat.rgb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

#================================================================================
# Install models and necesery extra files
#================================================================================
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/models/ResNet50/resnet50-v1-7.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/models/ResNet50/resnet_classes.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
