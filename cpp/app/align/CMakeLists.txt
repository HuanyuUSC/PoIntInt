# CMakeLists.txt for app/align/ directory
# Handles the mesh alignment application

if(BUILD_EXE)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    # Main executable with all source files
    add_executable(poIntInt_align
      main.cpp
      similarity_measure.cpp
      optimizer.cpp
      warm_start.cpp
    )
    target_link_libraries(poIntInt_align PRIVATE poIntInt_core)
    target_include_directories(poIntInt_align PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    # Define POINTINT_SOURCE_DIR for finding data directory
    # Use CMAKE_SOURCE_DIR to get the root project directory
    get_filename_component(CMAKE_CURRENT_SOURCE_DIR_NORMALIZED "${CMAKE_SOURCE_DIR}" ABSOLUTE)
    string(REPLACE "\\" "/" CMAKE_CURRENT_SOURCE_DIR_NORMALIZED "${CMAKE_CURRENT_SOURCE_DIR_NORMALIZED}")
    target_compile_definitions(poIntInt_align PRIVATE 
      POINTINT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR_NORMALIZED}")
    
    # Link libigl if available
    if(TARGET igl::glfw)
      target_link_libraries(poIntInt_align PRIVATE igl::glfw)
    endif()
    if(TARGET igl::imgui)
      target_link_libraries(poIntInt_align PUBLIC igl::imgui)
    endif()
    
    set_target_properties(poIntInt_align PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      FOLDER "app"
    )
    copy_tbb_dlls(poIntInt_align)
  endif()
endif()

