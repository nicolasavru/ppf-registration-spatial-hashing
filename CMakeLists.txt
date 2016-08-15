# include(/opt/pcl/src/pcl/cmake/pcl_utils.cmake)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)
# SET(CMAKE_BUILD_TYPE Release)

project(alignment)

# set(PCL_DIR "/opt/pcl/src/pcl/build")
find_package(PCL 1.7 REQUIRED)

add_definitions(-DBOOST_ALL_DYN_LINK)
find_package(Boost REQUIRED COMPONENTS program_options log)

set(CMAKE_MODULE_PATH "/opt/pcl/src/pcl/cmake/Modules/" ${CMAKE_MODULE_PATH})
find_package(Eigen REQUIRED)
# find_package(Eigen3 REQUIRED)

#cuda info:
# http://stackoverflow.com/questions/13073717/building-cuda-object-files-using-cmake
# https://devtalk.nvidia.com/default/topic/487573/cuda-programming-and-performance/linker-error-usr-bin-ld-cannot-find-lcudart/post/3497498/#3497498
find_package(CUDA 5.0 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS OFF)
# set(CUDA_VERBOSE_BUILD ON)


FILE(GLOB srcs src/*.cpp)
FILE(GLOB cuda src/cuda/*.cu)

source_group("Source Files\\cuda" FILES ${cuda})
source_group("Source Files" FILES ${srcs})

set(LIB_NAME alignment_cuda)

find_library(CUDADEVRT
   NAMES cudadevrt
   PATHS "/opt/" "/usr/local/" 
   PATH_SUFFIXES "cuda/lib64"
   DOC "Location of cudadevrt library"
   NO_DEFAULT_PATH
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

# list(APPEND CUDA_LIBRARIES -lcudadevrt)
set(CUDA_HOST_COMPILER "/usr/bin/g++-5")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-5")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -D_FORCE_INLINES -O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -O0 -g --std=c++11")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ftz=true;--prec-div=false;--prec-sqrt=false;-arch=sm_35;--expt-extended-lambda)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DTHRUST_DEBUG;-g)
# set(CUDA_NVCC_FLAGS -G)

# CUDA_COMPILE(cuda_objs ${cuda})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -I/home/avru/Documents/Senior_Project/thrust )

set(CUDA_SEPARABLE_COMPILATION ON)

# set(CUDA_64_BIT_DEVICE_CODE ON)

CUDA_ADD_LIBRARY(alignment_lib ${cuda} ${srcs})

add_executable (alignment src/alignment.cpp)
# set_property(TARGET alignment PROPERTY CXX_STANDARD 11)

target_link_libraries (alignment alignment_lib ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_LIBRARIES} ${CUDADEVRT})
# target_link_libraries (test test_lib ${PCL_LIBRARIES} ${CUDA_LIBRARIES} ${CUDADEVRT})
