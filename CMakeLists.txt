cmake_minimum_required(VERSION 3.29)
project(lstm)

set(CMAKE_CXX_STANDARD 20)

# Eigen kütüphanesinin yolunu belirtin
include_directories("./include/eigen-3.4.0")

add_executable(lstm main.cpp
        include/lstm_layer.h
        src/lstm_layer.cpp
        include/activation_functions.h
        src/activation_functions.cpp)
