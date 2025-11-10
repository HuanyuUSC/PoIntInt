cmake_minimum_required(VERSION 3.20)

# ---- Project ----
project(PoIntInt
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

# ---- Build options ----
option(BUILD_EXE "Build the main executable from src/main.cpp" ON)
option(FETCH_EIGEN "Fetch Eigen with FetchContent if not found locally" ON)
option(FETCH_LIBIGL "Fetch libigl with FetchContent" ON)
# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# C++ / CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (override on cmd line: -DCMAKE_CUDA_ARCHITECTURES="86;89")
set(CMAKE_CUDA_ARCHITECTURES 89)

# Position-independent code (useful for static libs too)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- CUDA Toolkit ----
find_package(CUDAToolkit REQUIRED)

# ---- Dependencies: Eigen, libigl ----
include(FetchContent)

# Try system Eigen first (quiet); fall back to FetchContent if requested.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND AND FETCH_EIGEN)
  message(STATUS "Eigen3 not found; fetching with FetchContent...")
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    FIND_PACKAGE_ARGS # exposes Eigen3::Eigen target
  )
  FetchContent_MakeAvailable(eigen)
endif()
if(NOT Eigen3_FOUND)
  message(FATAL_ERROR "Eigen3 not found and FETCH_EIGEN=OFF. Provide Eigen3 or enable fetching.")
endif()

# libigl (header-only core)
if(FETCH_LIBIGL)
  message(STATUS "Fetching libigl (core)...")
  FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0
  )
  set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE BOOL "" FORCE)
  set(LIBIGL_WITH_OPENGL OFF CACHE BOOL "" FORCE)
  set(LIBIGL_WITH_OPENGL_GLFW OFF CACHE BOOL "" FORCE)
  set(LIBIGL_WITH_TUTORIAL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(libigl)
else()
  # If you vendor libigl yourself, expose target igl::core before this point
  find_package(igl CONFIG REQUIRED) # expects igl::core
endif()

# ---- Sources ----
# Split source and headers so we can place them under separate VS filters.
file(GLOB_RECURSE POINTINT_SRC  CONFIGURE_DEPENDS
  src/*.cpp
  src/*.cu
  src/*.cxx
  src/*.cc
)
file(GLOB_RECURSE POINTINT_HDR  CONFIGURE_DEPENDS
  include/*.h
  include/*.hpp
  include/*.hh
  include/*.cuh
)
set(POINTINT_SOURCES ${POINTINT_SRC} ${POINTINT_HDR})

# ---- Visual Studio organization (mirrors folder structure) ----
# Enable solution folders for targets
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Place files under "src/..." and "include/..." filters matching their real paths
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"     PREFIX "src"     FILES ${POINTINT_SRC})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "include" FILES ${POINTINT_HDR})

# ---- Library target ----
add_library(poIntInt_core ${POINTINT_SOURCES})
target_include_directories(poIntInt_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(poIntInt_core
  PUBLIC
    Eigen3::Eigen
    igl::core
)
set_target_properties(poIntInt_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  FOLDER "lib"  # shows up under a 'lib' folder in VS Solution Explorer
)

# Note: IntelliSense for CUDA files in Visual Studio may show red underlines
# even when the code compiles correctly. This is a known limitation.
# The includes use quotes for project headers to help IntelliSense find them.
# Link CUDA runtime libraries
# For separable compilation, we need both cudart_static and cudadevrt
target_link_libraries(poIntInt_core
  PUBLIC
    CUDA::cudart_static
)
# cudadevrt is needed for device linking with separable compilation
# Add library directory and link the library
if(CUDAToolkit_FOUND AND CUDAToolkit_LIBRARY_DIR)
  target_link_directories(poIntInt_core PUBLIC ${CUDAToolkit_LIBRARY_DIR})
  target_link_libraries(poIntInt_core PUBLIC cudadevrt)
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(poIntInt_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>
  )
endif()

# Warnings:
if(MSVC)
  target_compile_options(poIntInt_core PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
)
else()
  target_compile_options(poIntInt_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Example executable (optional) ----
if(BUILD_EXE)
  # Only add if a main exists; avoids config errors on empty projects
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/app/main.cpp")
    add_executable(poIntInt_app app/main.cpp)
    target_link_libraries(poIntInt_app PRIVATE poIntInt_core)
    set_target_properties(poIntInt_app PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "app"
    )
  endif()
endif()

# ---- Nice status output ----
message(STATUS "CMAKE_BUILD_TYPE                = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES        = ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Eigen3::Eigen                   = ${Eigen3_DIR}")
