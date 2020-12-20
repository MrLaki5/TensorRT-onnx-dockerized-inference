cmake_minimum_required(VERSION 3.1)

project(tensorrt-engine)

# Set compiler
set(CMAKE_CXX_FLAGS "-std=c++11 -g -O3 -fPIC")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

find_package(CUDA)

message("Cuda libs: " ${CUDA_TOOLKIT_ROOT_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)

set(LIB "tensorrt-engine")
# Store all cpp files into SRC variable
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# Lib
add_library(${LIB} SHARED ${SRC})
# Include header files
target_include_directories(${LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                  PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${LIB} PRIVATE cudart nvinfer nvonnxparser)


set(BIN "tensorrt-engine-test")
add_executable(${BIN} ${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp)
target_link_libraries(${BIN} PRIVATE ${LIB})
target_include_directories(${BIN} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/retinanet-9.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
