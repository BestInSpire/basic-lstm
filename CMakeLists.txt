cmake_minimum_required(VERSION 3.29)
project(lstm)

set(CMAKE_CXX_STANDARD 20)

add_executable(lstm main.cpp
        LSTMCell/LSTMCell.h
        LSTMCell/LSTMCell.cpp)
