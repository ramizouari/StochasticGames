cmake_minimum_required(VERSION 3.24)

project(mpgcpp_examples)
set(CMAKE_CXX_STANDARD 20)
include_directories(..)

add_executable(MeanPayoffs MeanPayoffs.cpp ../csp/Variable.cpp )