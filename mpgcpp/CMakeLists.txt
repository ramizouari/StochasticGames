cmake_minimum_required(VERSION 3.24)
project(mpgcpp)
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}")
message(STATUS "Install directory: ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_CXX_STANDARD 20)
set(HEADER_FILES include/game/MeanPayoffGame.h include/csp/Variable.h include/csp/MaxAtomSystem.h
        include/csp/MaxAtomSolver.h include/csp/MinMaxSystem.h include/IntRange.h include/ProgramOptions.h
        include/csp/MaxAtomSolver.h include/csp/solver/ParallelMaxAtomSolver.h
        include/csp/solver/DenseSolver.h)
set(SOURCE_FILES src/csp/Variable.cpp)
#Headers
include_directories(include)
include_directories(external/CPLibrary/include)
include_directories(external)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Include boost headers
find_package(Boost COMPONENTS iostreams program_options log REQUIRED)

add_executable(MPGSolver src/MPGSolver.cpp ${HEADER_FILES} ${SOURCE_FILES} src/Writer.cpp src/ProgramOptions.cpp)
add_executable(MPGGenerator src/MPGGenerator.cpp ${HEADER_FILES} ${SOURCE_FILES}
        src/Writer.cpp src/ProgramOptions.cpp)

target_link_libraries(MPGSolver Boost::iostreams Boost::program_options Boost::log pthread)
target_link_libraries(MPGGenerator Boost::iostreams Boost::program_options Boost::log pthread)

execute_process(COMMAND python3-config --includes OUTPUT_VARIABLE PYTHON_INCLUDE RESULT_VARIABLE PYTHON_INCLUDE_RESULT)
if (NOT ${PYTHON_INCLUDE_RESULT} EQUAL 0)
    message(WARNING "Unable to get python include directory. Please set PYTHON_INCLUDE manually. Otherwise, the python wrapper will not be built.")
    if(NOT PYTHON_INCLUDE)
        message(WARNING "PYTHON_INCLUDE is not set. The python wrapper will not be built.")
    endif()
else()
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    set(boost_python "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
    message(STATUS "Boost python library: ${boost_python}")
    find_package(Boost COMPONENTS ${boost_python} REQUIRED)    # Remove -I from the output of python3-config --includes
    string(REPLACE "-I" "" PYTHON_INCLUDE ${PYTHON_INCLUDE})
    message(STATUS "Python include: ${PYTHON_INCLUDE}")
    # Split the string into a list
    separate_arguments(PYTHON_INCLUDE)
    # Add the python include directory to the include directories
    include_directories(${PYTHON_INCLUDE})

    set(WRAPPER_NAME mpgcpp)
    set(LIBRARY_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}${WRAPPER_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(mpgcpp.template "${WRAPPER_NAME}.py")

    add_library(MPGPyWrapper SHARED src/PyMPGWrapper.cpp ${HEADER_FILES} ${SOURCE_FILES} include/csp/MaxAtomSolver.h)

    set_target_properties(MPGPyWrapper PROPERTIES OUTPUT_NAME ${WRAPPER_NAME})
    target_link_libraries(MPGPyWrapper Boost::iostreams Boost::program_options Boost::log pthread Boost::${boost_python})

    # Set the default install location for the python wrapper
    if (NOT MPG_PYTHON)
        set(PARENT "${CMAKE_CURRENT_SOURCE_DIR}/..")
        set(MPG_PYTHON "${PARENT}/mpg/wrapper")
        message("MPG_PYTHON is not set. The python wrapper will be installed to ${MPG_PYTHON}. Set MPG_PYTHON to change this.")
    else()
        message(STATUS "Python wrapper will be installed to ${MPG_PYTHON}")
    endif()

    install(TARGETS MPGSolver DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    # Install the python wrapper
    install(TARGETS MPGPyWrapper DESTINATION ${MPG_PYTHON})
    # Install the python wrapper script
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${WRAPPER_NAME}.py" DESTINATION ${MPG_PYTHON})
endif()

set(MPG_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(example)

if(${DISABLE_TESTS})
    message(STATUS "Tests are disabled.")
else()
    FetchContent_MakeAvailable(googletest)
    if (NOT TARGET GTest::gtest_main)
        message(WARNING "Unable to find gtest. Tests will not be built.")
    else()
        message(STATUS "Found gtest. Tests will be built.")
        enable_testing()

        add_executable(
                TernaryMaxAtomTest
                tests/ternary_max_test.cpp
                tests/ternary_max_double_test.cpp
                src/csp/Variable.cpp
                include/csp/MaxAtomSolver.h)
        add_executable(
                MaxAtomTest
                tests/nary_max_test.cpp
                tests/nary_max_double_test.cpp
                tests/utils.cpp
                src/csp/Variable.cpp
                include/csp/MaxAtomSolver.h tests/utils.h tests/utils.cpp)
        target_link_libraries(
                TernaryMaxAtomTest
                GTest::gtest_main
        )
        target_link_libraries(
                MaxAtomTest
                GTest::gtest_main
        )

        include(GoogleTest)
        gtest_discover_tests(TernaryMaxAtomTest)
        gtest_discover_tests(MaxAtomTest)
    endif()

endif()

