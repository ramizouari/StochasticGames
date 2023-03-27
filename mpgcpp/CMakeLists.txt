cmake_minimum_required(VERSION 3.24)
project(mpgcpp)

set(CMAKE_CXX_STANDARD 20)
set(HEADER_FILES game/MeanPayoffGame.h csp/Variable.h csp/MaxAtomSystem.h csp/MinMaxSystem.h IntRange.h mpgio/MPGReader.h Writer.h ProgramOptions.h)
set(SOURCE_FILES csp/Variable.cpp Writer.cpp ProgramOptions.cpp)
#Headers
include_directories(.)
include_directories(CPLibrary/include)
add_library(CPLibrary STATIC IMPORTED)
# Include boost headers
find_package(Boost COMPONENTS iostreams program_options log python REQUIRED)

add_executable(MPGSolver MPGSolver.cpp ${HEADER_FILES} ${SOURCE_FILES})
add_executable(MPGGenerator MPGGenerator.cpp ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(MPGSolver Boost::iostreams Boost::program_options Boost::log pthread)
target_link_libraries(MPGGenerator Boost::iostreams Boost::program_options Boost::log pthread)

execute_process(COMMAND python3-config --includes OUTPUT_VARIABLE PYTHON_INCLUDE RESULT_VARIABLE PYTHON_INCLUDE_RESULT)
if (NOT ${PYTHON_INCLUDE_RESULT} EQUAL 0)
    message(WARNING "Unable to get python include directory. Please set PYTHON_INCLUDE manually. Otherwise, the python wrapper will not be built.")
    if(NOT PYTHON_INCLUDE)
        message(WARNING "PYTHON_INCLUDE is not set. The python wrapper will not be built.")
    endif()
else()
    # Remove -I from the output of python3-config --includes
    string(REPLACE "-I" "" PYTHON_INCLUDE ${PYTHON_INCLUDE})
    message(STATUS "Python include: ${PYTHON_INCLUDE}")
    # Split the string into a list
    separate_arguments(PYTHON_INCLUDE)
    # Add the python include directory to the include directories
    include_directories(${PYTHON_INCLUDE})

    set(WRAPPER_NAME mpgcpp)
    set(LIBRARY_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}${WRAPPER_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(mpgcpp.template "${WRAPPER_NAME}.py")

    add_library(MPGPyWrapper SHARED PyMPGWrapper.cpp ${HEADER_FILES} ${SOURCE_FILES})

    set_target_properties(MPGPyWrapper PROPERTIES OUTPUT_NAME ${WRAPPER_NAME})
    target_link_libraries(MPGPyWrapper Boost::iostreams Boost::program_options Boost::log pthread Boost::python)

    # Set the default install location for the python wrapper
    if (NOT MPG_PYTHON)
        set(PARENT "${CMAKE_CURRENT_SOURCE_DIR}/..")
        set(MPG_PYTHON "${PARENT}/mpg/wrapper")
        message("MPG_PYTHON is not set. The python wrapper will be installed to ${MPG_PYTHON}. Set MPG_PYTHON to change this.")
    else()
        message(STATUS "Python wrapper will be installed to ${MPG_PYTHON}")
    endif()

    # Install the python wrapper
    install(TARGETS MPGPyWrapper DESTINATION ${MPG_PYTHON})
    # Install the python wrapper script
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${WRAPPER_NAME}.py" DESTINATION ${MPG_PYTHON})
endif()

set(MPG_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(example)