cmake_minimum_required(VERSION 3.16)
project(panic)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(panic
        cl2.hpp
        input.txt
        lab2.pdf
        main.cpp
        output.txt
        README.md
        scan.cl)
