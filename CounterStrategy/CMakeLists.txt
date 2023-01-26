cmake_minimum_required(VERSION 3.24)
project(CounterStrategy)

set(CMAKE_CXX_STANDARD 20)

add_executable(CounterStrategy counter.cpp extended_integer.h types.h operation.h tropical_matrix.h games.h)
add_executable(OptimalStrategy optimal.cpp extended_integer.h types.h operation.h tropical_matrix.h games.h)
