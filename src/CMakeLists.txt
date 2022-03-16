include("../build_utils/CMakeLists.txt")
SetRequiredCMakeVersion()
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})



project(GSOC_DPC_EXERCISE CXX)
SetCompilerFlags()
SetBuildType()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gline-tables-only")

add_executable(dpc_exercise vectMatvect.cpp)
target_include_directories(dpc_exercise
  PRIVATE "${PROJECT_SOURCE_DIR}/../utils")
if(CMAKE_INCLUDE_PATH)
  target_include_directories(dpc_exercise
     PUBLIC "${CMAKE_INCLUDE_PATH}")
endif()

FindOpenCLLibrary(dpc_exercise)
FindOpenCLHeaders(dpc_exercise)
