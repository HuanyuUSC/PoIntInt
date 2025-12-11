cmake_minimum_required(VERSION 3.20)
project(poIntInt_demo LANGUAGES CXX CUDA)

# ---- Build options ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# ---- Find dependencies ----
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)

# Eigen: try config mode first, then module mode
find_package(Eigen3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
  find_package(Eigen3 REQUIRED)
endif()

# ---- Path to cpp project ----
set(POINTINT_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")
set(POINTINT_BUILD_DIR "${POINTINT_CPP_DIR}/build")

# Check if poIntInt_core library exists
if(WIN32)
  set(POINTINT_CORE_LIB "${POINTINT_BUILD_DIR}/Release/poIntInt_core.lib")
  if(NOT EXISTS "${POINTINT_CORE_LIB}")
    set(POINTINT_CORE_LIB "${POINTINT_BUILD_DIR}/Debug/poIntInt_core.lib")
  endif()
  if(NOT EXISTS "${POINTINT_CORE_LIB}")
    set(POINTINT_CORE_LIB "${POINTINT_BUILD_DIR}/poIntInt_core.lib")
  endif()
else()
  set(POINTINT_CORE_LIB "${POINTINT_BUILD_DIR}/libpoIntInt_core.a")
endif()

if(NOT EXISTS "${POINTINT_CORE_LIB}")
  message(FATAL_ERROR
    "poIntInt_core library not found at ${POINTINT_CORE_LIB}\n"
    "Please build the cpp project first:\n"
    "  cd ${POINTINT_CPP_DIR} && mkdir -p build && cd build\n"
    "  cmake .. -DBUILD_EXE=OFF && cmake --build . --target poIntInt_core"
  )
endif()

message(STATUS "Found poIntInt_core: ${POINTINT_CORE_LIB}")

# ---- Create Python module ----
pybind11_add_module(pointint_cpp bindings.cpp)

target_include_directories(pointint_cpp PRIVATE
  ${POINTINT_CPP_DIR}/include
)

target_link_libraries(pointint_cpp PRIVATE
  ${POINTINT_CORE_LIB}
  Eigen3::Eigen
  TBB::tbb
  CUDA::cudart_static
)

# Link cudadevrt if available
if(CUDAToolkit_FOUND AND CUDAToolkit_LIBRARY_DIR)
  target_link_directories(pointint_cpp PRIVATE ${CUDAToolkit_LIBRARY_DIR})
  target_link_libraries(pointint_cpp PRIVATE cudadevrt)
endif()

# ---- Output to demo directory ----
set_target_properties(pointint_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# For multi-config generators (Visual Studio, Xcode)
foreach(CONFIG_TYPE ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${CONFIG_TYPE} CONFIG_TYPE_UPPER)
  set_target_properties(pointint_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPE_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endforeach()

message(STATUS "Python module will be output to: ${CMAKE_CURRENT_SOURCE_DIR}")
