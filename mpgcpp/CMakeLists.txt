cmake_minimum_required(VERSION 3.24)
project(mpgcpp)

set(CMAKE_CXX_STANDARD 20)
set(HEADER_FILES MeanPayoffGame.h Variable.h MaxAtomSystem.h MinMaxSystem.h IntRange.h)
set(SOURCE_FILES Variable.cpp)
include_directories(CPLibrary/include)

add_executable(mpgcpp main.cpp ${HEADER_FILES} ${SOURCE_FILES})
