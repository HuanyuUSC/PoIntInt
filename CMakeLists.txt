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

# CUDA architectures (override with -DCMAKE_CUDA_ARCHITECTURES="86;89")
set(CMAKE_CUDA_ARCHITECTURES 89)

# Position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- CUDA Toolkit ----
find_package(CUDAToolkit REQUIRED)

# ---- TBB (Intel Threading Building Blocks) ----
find_package(TBB REQUIRED)

# Function to copy TBB DLLs to target's output directory
function(copy_tbb_dlls target)
  if(WIN32 AND TBB_FOUND)
    # Try to get TBB library location
    get_target_property(TBB_LIBRARY_LOCATION TBB::tbb IMPORTED_LOCATION)
    if(NOT TBB_LIBRARY_LOCATION)
      get_target_property(TBB_LIBRARY_LOCATION TBB::tbb IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT TBB_LIBRARY_LOCATION)
      get_target_property(TBB_LIBRARY_LOCATION TBB::tbb IMPORTED_LOCATION_DEBUG)
    endif()
    
    if(TBB_LIBRARY_LOCATION)
      get_filename_component(TBB_LIBRARY_DIR "${TBB_LIBRARY_LOCATION}" DIRECTORY)
      
      # TBB DLLs are typically in a 'bin' or 'redist' directory at the same level as 'lib'
      # Try common locations
      get_filename_component(TBB_ROOT "${TBB_LIBRARY_DIR}" DIRECTORY)
      set(TBB_BIN_DIRS
        "${TBB_ROOT}/bin"
        "${TBB_ROOT}/redist"
        "${TBB_ROOT}/bin/intel64/vc14"
        "${TBB_ROOT}/bin/intel64/vc_mt"
        "${TBB_LIBRARY_DIR}"
        "${TBB_ROOT}/redist/intel64/vc14"
        "${TBB_ROOT}/redist/intel64/vc_mt"
      )
      
      # Determine build type suffix
      set(TBB_DLL_PATTERNS)
      if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES)
        list(APPEND TBB_DLL_PATTERNS "tbb*_debug.dll" "tbb*d.dll" "tbb12_debug.dll" "tbbmalloc*_debug.dll" "tbbmalloc*d.dll")
      else()
        list(APPEND TBB_DLL_PATTERNS "tbb*.dll" "tbb12.dll" "tbbmalloc*.dll")
      endif()
      
      # Find and copy DLLs from all possible locations
      foreach(bin_dir ${TBB_BIN_DIRS})
        if(EXISTS "${bin_dir}")
          foreach(pattern ${TBB_DLL_PATTERNS})
            file(GLOB TBB_DLLS "${bin_dir}/${pattern}")
            foreach(dll ${TBB_DLLS})
              add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:${target}>
                COMMENT "Copying TBB DLL: ${dll}"
              )
            endforeach()
          endforeach()
        endif()
      endforeach()
    endif()
  endif()
endfunction()

# ---- Dependencies: Eigen, libigl ----
include(FetchContent)

# ---------------------------------------------------------------------------
# Fixed Eigen3 logic (final)
# ---------------------------------------------------------------------------
unset(Eigen3_DIR CACHE)
unset(Eigen3_FOUND CACHE)

if(FETCH_EIGEN)
  message(STATUS "Fetching Eigen via FetchContent...")
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)

  # Eigen is header-only; make sure include path is available
  set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
  include_directories(${EIGEN3_INCLUDE_DIR})

  # Only create the target if it doesn't already exist
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
  endif()
else()
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# ---------------------------------------------------------------------------
# libigl (header-only core)
# ---------------------------------------------------------------------------
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
  find_package(igl CONFIG REQUIRED)
endif()

# ---- Sources ----
file(GLOB_RECURSE POINTINT_SRC CONFIGURE_DEPENDS
  src/*.cpp
  src/*.cu
  src/*.cxx
  src/*.cc
)
file(GLOB_RECURSE POINTINT_HDR CONFIGURE_DEPENDS
  include/*.h
  include/*.hpp
  include/*.hh
  include/*.cuh
)
set(POINTINT_SOURCES ${POINTINT_SRC} ${POINTINT_HDR})

# ---- Visual Studio organization ----
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${POINTINT_SRC})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "include" FILES ${POINTINT_HDR})

# ---- Library target ----
add_library(poIntInt_core ${POINTINT_SOURCES})
target_include_directories(poIntInt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(poIntInt_core PUBLIC Eigen3::Eigen igl::core TBB::tbb)
set_target_properties(poIntInt_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  FOLDER "lib"
)

# ---- CUDA runtime linking ----
target_link_libraries(poIntInt_core PUBLIC CUDA::cudart_static)
if(CUDAToolkit_FOUND AND CUDAToolkit_LIBRARY_DIR)
  target_link_directories(poIntInt_core PUBLIC ${CUDAToolkit_LIBRARY_DIR})
  target_link_libraries(poIntInt_core PUBLIC cudadevrt)
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(poIntInt_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>
  )
endif()

# ---- Warnings ----
if(MSVC)
  target_compile_options(poIntInt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(poIntInt_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Example executable (optional) ----
if(BUILD_EXE)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/app/main.cpp")
    add_executable(poIntInt_app app/main.cpp)
    target_link_libraries(poIntInt_app PRIVATE poIntInt_core)
    set_target_properties(poIntInt_app PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "app"
    )
    copy_tbb_dlls(poIntInt_app)
  endif()
endif()

# ---- Unit Tests ----
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_unit_cube.cpp")
    add_executable(poIntInt_test_triangle_mesh test/test_unit_cube.cpp)
    target_link_libraries(poIntInt_test_triangle_mesh PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_triangle_mesh PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_triangle_mesh)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_sphere_pointcloud.cpp")
    add_executable(poIntInt_test_point_cloud test/test_sphere_pointcloud.cpp)
    target_link_libraries(poIntInt_test_point_cloud PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_point_cloud PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_point_cloud)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_gaussian_splats.cpp")
    add_executable(poIntInt_test_gaussian_splat test/test_gaussian_splats.cpp)
    target_link_libraries(poIntInt_test_gaussian_splat PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_gaussian_splat PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_gaussian_splat)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_multi_object.cpp")
    add_executable(poIntInt_test_multi_object test/test_multi_object.cpp)
    target_link_libraries(poIntInt_test_multi_object PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_multi_object PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_multi_object)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_dof_parameterization.cpp")
    add_executable(poIntInt_test_differentiability test/test_dof_parameterization.cpp)
    target_link_libraries(poIntInt_test_differentiability PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_differentiability PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_differentiability)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_intersection_volume_gradient.cpp")
    # Configure header file with source directory path
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_config.h.in"
      "${CMAKE_CURRENT_BINARY_DIR}/test_config.h"
      @ONLY
    )
    add_executable(poIntInt_test_intersection_volume_gradient test/test_intersection_volume_gradient.cpp)
    target_link_libraries(poIntInt_test_intersection_volume_gradient PRIVATE poIntInt_core)
    # Add include directory for generated config header
    target_include_directories(poIntInt_test_intersection_volume_gradient PRIVATE 
      "${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties(poIntInt_test_intersection_volume_gradient PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_intersection_volume_gradient)
  endif()
endif()

# ---- Status Output ----
message(STATUS "CMAKE_BUILD_TYPE                = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES        = ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Eigen3::Eigen                   = ${EIGEN3_INCLUDE_DIR}")
