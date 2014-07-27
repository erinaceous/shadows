cmake_minimum_required(VERSION 2.8)

# Problem with CMake detecting my Linux setup correctly (on Arch).
# Unlikely you'll need to change the line below unless you're compiling this on
# a really small system (like an embedded one) which really doesn't support
# dynamic linking of libraries.
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
SET_PROPERTY(GLOBAL PROPERTY CMAKE_BUILD_TYPE "RelWithDebInfo")

include(FindOpenCV.cmake)
#include(FindCWD.cmake)
project(shadow_methods)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)
#find_package(CWD)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})

set(CMAKE_CXX_FLAGS              "-Wall -std=c++11")
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
set(BOOST_ALL_DYN_LINK           ON)

file(GLOB files src/*.cpp)
foreach(source ${files})
    GET_FILENAME_COMPONENT(basename ${source} NAME_WE)
    message("Will compile ${basename}")
    add_executable(${basename} ${source})
    target_link_libraries(${basename} ${OpenCV_LIBS} ${Boost_LIBRARIES})
endforeach(source)

