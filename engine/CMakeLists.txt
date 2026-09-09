cmake_minimum_required(VERSION 3.16)
project(chlorine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(chlorine
  src/main.cpp
  src/hgn.cpp
  src/serve.cpp)
target_compile_options(chlorine PRIVATE -Wall -Wextra -O2)
