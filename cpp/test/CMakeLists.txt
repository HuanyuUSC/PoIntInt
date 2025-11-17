# CMakeLists.txt for test/ directory
# Handles all unit test executables

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
  # Test: Triangle mesh (unit cube)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_unit_cube.cpp")
    add_executable(poIntInt_test_triangle_mesh test_unit_cube.cpp)
    target_link_libraries(poIntInt_test_triangle_mesh PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_triangle_mesh PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_triangle_mesh)
  endif()
  
  # Test: Point cloud (sphere)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_sphere_pointcloud.cpp")
    add_executable(poIntInt_test_point_cloud test_sphere_pointcloud.cpp)
    target_link_libraries(poIntInt_test_point_cloud PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_point_cloud PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_point_cloud)
  endif()
  
  # Test: Gaussian splats
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_gaussian_splats.cpp")
    add_executable(poIntInt_test_gaussian_splat test_gaussian_splats.cpp)
    target_link_libraries(poIntInt_test_gaussian_splat PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_gaussian_splat PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_gaussian_splat)
  endif()
  
  # Test: Multi-object
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_multi_object.cpp")
    add_executable(poIntInt_test_multi_object test_multi_object.cpp)
    target_link_libraries(poIntInt_test_multi_object PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_multi_object PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_multi_object)
  endif()
  
  # Test: Volume unified interface
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_volume_unified.cpp")
    # Configure header file with source directory path
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/test_config.h.in"
      "${CMAKE_CURRENT_BINARY_DIR}/test_config.h"
      @ONLY
    )
    add_executable(poIntInt_test_volume_unified test_volume_unified.cpp)
    target_link_libraries(poIntInt_test_volume_unified PRIVATE poIntInt_core)
    # Add include directory for generated config header
    target_include_directories(poIntInt_test_volume_unified PRIVATE 
      "${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties(poIntInt_test_volume_unified PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_volume_unified)
  endif()
  
  # Test: DoF parameterization (differentiability)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_dof_parameterization.cpp")
    add_executable(poIntInt_test_differentiability test_dof_parameterization.cpp)
    target_link_libraries(poIntInt_test_differentiability PRIVATE poIntInt_core)
    set_target_properties(poIntInt_test_differentiability PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "test"
    )
    copy_tbb_dlls(poIntInt_test_differentiability)
  endif()
  
  # Test: Intersection volume gradient
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_intersection_volume_gradient.cpp")
    # Configure header file with source directory path
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/test_config.h.in"
      "${CMAKE_CURRENT_BINARY_DIR}/test_config.h"
      @ONLY
    )
    add_executable(poIntInt_test_intersection_volume_gradient test_intersection_volume_gradient.cpp)
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

