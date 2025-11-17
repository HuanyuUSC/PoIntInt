# CMakeLists.txt for app/ directory
# Handles main app executables (not in subdirectories)

# ---- Example executables (optional) ----
if(BUILD_EXE)
  # Main app
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(poIntInt_app main.cpp)
    target_link_libraries(poIntInt_app PRIVATE poIntInt_core)
    set_target_properties(poIntInt_app PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "app"
    )
    copy_tbb_dlls(poIntInt_app)
  endif()
  
  # Visualization demo (requires libigl viewer with OpenGL/GLFW)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/visualize_gradient.cpp")
    add_executable(poIntInt_visualize visualize_gradient.cpp)
    target_link_libraries(poIntInt_visualize PRIVATE poIntInt_core)
    
    # Define source directory for finding data files
    # Use CMAKE_SOURCE_DIR to get the root project directory
    # Convert Windows backslashes to forward slashes for the preprocessor string
    string(REPLACE "\\" "/" POINTINT_SOURCE_DIR_NORMALIZED "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(poIntInt_visualize PRIVATE 
      POINTINT_SOURCE_DIR="${POINTINT_SOURCE_DIR_NORMALIZED}"
    )
    
    # Link against libigl viewer components (if available)
    # Note: igl::glfw should automatically pull in igl::opengl and glad::glad
    if(TARGET igl::glfw)
      target_link_libraries(poIntInt_visualize PRIVATE igl::glfw)
    endif()
    if(TARGET igl::imgui)
      # Use PUBLIC linking to ensure include directories are propagated
      target_link_libraries(poIntInt_visualize PUBLIC igl::imgui)
    endif()
    
    set_target_properties(poIntInt_visualize PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "app"
    )
    copy_tbb_dlls(poIntInt_visualize)
  endif()
endif()

