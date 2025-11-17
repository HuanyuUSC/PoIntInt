#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iomanip>

#include "compute_intersection_volume.hpp"
#include "compute_volume.hpp"
#include "dof/affine_dof.hpp"
#include "geometry/packing.hpp"
#include "geometry/geometry_helpers.hpp"
#include "quadrature/lebedev_io.hpp"
#include "quadrature/kgrid.hpp"
#include "similarity_measure.hpp"
#include "optimizer.hpp"
#include "warm_start.hpp"

using namespace PoIntInt;

// ============================================================================
// Helper Functions
// ============================================================================

// Helper function to find the project data directory
std::string find_data_directory() {
  std::vector<std::string> candidates;
  
#ifdef POINTINT_SOURCE_DIR
  std::string source_dir = POINTINT_SOURCE_DIR;
  for (size_t i = 0; i < source_dir.length(); ++i) {
    if (source_dir[i] == '\\') {
      source_dir[i] = '/';
    }
  }
  candidates.push_back(source_dir + "/data/lebedev");
#endif
  
  candidates.push_back("data/lebedev");
  candidates.push_back("../data/lebedev");
  candidates.push_back("../../data/lebedev");
  
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  
  return "";
}

// Format Lebedev number to 3-digit string (e.g., 29 -> "029")
std::string format_lebedev_number(int number) {
  if (number < 10) {
    return "00" + std::to_string(number);
  } else if (number < 100) {
    return "0" + std::to_string(number);
  } else {
    return std::to_string(number);
  }
}

// Get Lebedev file path from number
std::string get_lebedev_file_path(int lebedev_number) {
  std::string data_dir = find_data_directory();
  if (data_dir.empty()) {
    throw std::runtime_error("Could not find data/lebedev directory");
  }
  std::string filename = "lebedev_" + format_lebedev_number(lebedev_number) + ".txt";
  return data_dir + "/" + filename;
}

int get_default_lebedev_number() {
  return 29;
}

// ============================================================================
// Main Application
// ============================================================================

// Global state
Eigen::MatrixXd V1, V2;
Eigen::MatrixXi F1, F2;
Geometry ref_geom1, ref_geom2;
std::shared_ptr<AffineDoF> affine_dof;
Eigen::VectorXd dofs2;
KGrid kgrid;
double V1_volume = 0.0;
double V2_volume = 0.0;
double intersection_volume = 0.0;
double similarity = 0.0;  // Similarity value (0-1)
double similarity_loss = 0.0;  // -log(similarity), the objective to minimize
SimilarityMeasure current_measure = SIM_DICE;
bool optimization_running = false;
bool optimization_complete = false;
bool optimization_step_mode = false;  // Step-by-step mode
OptimizationState opt_state;
int current_opt_iter = 0;
int max_opt_iterations = 50;
float opt_tolerance = 1e-6;
float opt_damping = 0.1;

// Forward declaration
void update_geometry_and_volumes();

// Function to update geometry and compute volumes
void update_geometry_and_volumes() {
  // Compute volumes
  // V1 is fixed, so compute once using identity DoF
  static double V1_computed = 0.0;
  static bool V1_initialized = false;
  static Eigen::VectorXd identity_dofs(12);
  if (!V1_initialized) {
    identity_dofs << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    auto identity_dof = std::make_shared<AffineDoF>();
    V1_computed = compute_intersection_volume_cuda(
      ref_geom1, ref_geom1, kgrid, 256, false
    );
    // V1_computed = identity_dof->compute_volume(ref_geom1, identity_dofs);
    V1_initialized = true;
  }
  V1_volume = V1_computed;
  V2_volume = compute_intersection_volume_cuda(
    ref_geom2, ref_geom2, affine_dof, affine_dof, dofs2, dofs2,
    kgrid, ComputationFlags::VOLUME_ONLY, 256, false
  ).volume;
  // V2_volume = affine_dof->compute_volume(ref_geom2, dofs2);
  
  // Compute intersection volume
  // Use identity DoF for mesh 1 (fixed)
  static Eigen::VectorXd identity_dofs_static(12);
  identity_dofs_static << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  auto vol_result = compute_intersection_volume_cuda(
    ref_geom1, ref_geom2,
    std::make_shared<AffineDoF>(), affine_dof,
    identity_dofs_static, dofs2,
    kgrid, ComputationFlags::VOLUME_ONLY, 256, false
  );
  intersection_volume = vol_result.volume;
  
  // Compute similarity (function only for display)
  auto sim_result = compute_similarity(
    ref_geom1, ref_geom2, affine_dof, dofs2, kgrid,
    current_measure, V1_volume, SIM_FUNC_ONLY
  );
  similarity = sim_result.similarity;  // Original similarity (0-1)
  similarity_loss = sim_result.value;  // -log(similarity), the objective
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <mesh1.off> <mesh2.off> [lebedev_number] [Nrad]" << std::endl;
    std::cerr << "  lebedev_number: odd number in [3, 131] (default: 29)" << std::endl;
    std::cerr << "  Nrad: number of radial quadrature points (default: 32)" << std::endl;
    return 1;
  }
  
  std::string mesh1_file = argv[1];
  std::string mesh2_file = argv[2];
  std::string leb_file;
  
  // Parse Lebedev number
  int lebedev_number = (argc > 3) ? std::atoi(argv[3]) : get_default_lebedev_number();
  if (lebedev_number < 3 || lebedev_number > 131 || lebedev_number % 2 == 0) {
    std::cerr << "Error: lebedev_number must be an odd number in range [3, 131]" << std::endl;
    return 1;
  }
  
  leb_file = get_lebedev_file_path(lebedev_number);
  int Nrad = (argc > 4) ? std::atoi(argv[4]) : 32;
  
  // Load meshes
  std::cout << "Loading mesh 1: " << mesh1_file << std::endl;
  if (!igl::readOFF(mesh1_file, V1, F1)) {
    std::cerr << "Failed to load mesh 1: " << mesh1_file << std::endl;
    return 1;
  }
  
  std::cout << "Loading mesh 2: " << mesh2_file << std::endl;
  if (!igl::readOFF(mesh2_file, V2, F2)) {
    std::cerr << "Failed to load mesh 2: " << mesh2_file << std::endl;
    return 1;
  }
  
  std::cout << "Mesh 1: " << V1.rows() << " vertices, " << F1.rows() << " faces" << std::endl;
  std::cout << "Mesh 2: " << V2.rows() << " vertices, " << F2.rows() << " faces" << std::endl;
  
  // Create geometries
  ref_geom1 = make_triangle_mesh(V1, F1);
  ref_geom2 = make_triangle_mesh(V2, F2);
  
  // Create affine DoF for mesh 2 (mesh 1 stays fixed)
  affine_dof = std::make_shared<AffineDoF>();
  
  // Initialize DoFs to identity transform
  dofs2 = Eigen::VectorXd(12);
  dofs2 << 0.0, 0.0, 0.0,  // Translation
           1.0, 0.0, 0.0,  // Matrix row 1
           0.0, 1.0, 0.0,  // Matrix row 2
           0.0, 0.0, 1.0;  // Matrix row 3
  
  // Compute warm start: fit affine transform to match moments
  std::cout << "\n=== Computing Warm Start ===" << std::endl;
  dofs2 = compute_warm_start(ref_geom1, ref_geom2, affine_dof);
  std::cout << "=== Warm Start Complete ===\n" << std::endl;
  
  // Load Lebedev grid and build KGrid (needed for similarity computation)
  std::cout << "Loading Lebedev grid: " << leb_file << std::endl;
  LebedevGrid L = load_lebedev_txt(leb_file);
  kgrid = build_kgrid(L.dirs, L.weights, Nrad);
  std::cout << "KGrid: " << kgrid.kmag.size() << " k-nodes" << std::endl;
  
  // Compute V1_volume (needed for similarity computation)
  // V1 is fixed, so compute once using identity DoF
  static Eigen::VectorXd identity_dofs_static(12);
  identity_dofs_static << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  V1_volume = compute_intersection_volume_cuda(
    ref_geom1, ref_geom1, kgrid, 256, false
  );
  
  // Initial volume computation
  update_geometry_and_volumes();
  std::cout << "Initial volumes:" << std::endl;
  std::cout << "  Mesh 1: " << V1_volume << std::endl;
  std::cout << "  Mesh 2: " << V2_volume << std::endl;
  std::cout << "  Intersection: " << intersection_volume << std::endl;
  std::cout << "  Similarity: " << (similarity * 100.0) << "%" << std::endl;
  std::cout << "  Similarity Loss (-log): " << similarity_loss << std::endl;
  
  // Set up viewer
  igl::opengl::glfw::Viewer viewer;
  
  // Set mesh 1
  viewer.data().set_mesh(V1, F1);
  viewer.data().set_colors(Eigen::MatrixXd::Ones(V1.rows(), 3) * 0.8);
  
  // Append mesh 2
  viewer.append_mesh();
  viewer.data().set_mesh(V2, F2);
  viewer.data().set_colors(Eigen::MatrixXd::Ones(V2.rows(), 3) * 0.8);
  
  auto& data1 = viewer.data_list[0];
  auto& data2 = viewer.data_list[1];
  
  // Set up ImGui
  igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
  viewer.plugins.push_back(&imgui_plugin);
  
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  imgui_plugin.widgets.push_back(&menu);
  
  // Function to update mesh 2 visualization
  auto update_mesh2_visualization = [&]() {
    // Apply affine transform to mesh 2 vertices
    Eigen::Matrix3d A;
    Eigen::Vector3d t;
    A << dofs2(3), dofs2(4), dofs2(5),
         dofs2(6), dofs2(7), dofs2(8),
         dofs2(9), dofs2(10), dofs2(11);
    t << dofs2(0), dofs2(1), dofs2(2);
    
    Eigen::MatrixXd V2_transformed = (V2 * A.transpose()).rowwise() + t.transpose();
    data2.set_mesh(V2_transformed, F2);
    data2.set_colors(Eigen::MatrixXd::Ones(V2.rows(), 3) * 0.8);
  };
  
  // Function to test gradient against finite differencing
  auto test_gradient = [&]() {
    std::cout << "\n=== Testing Similarity Loss Gradient ===" << std::endl;
    std::cout << "Current DoFs: [" << dofs2.transpose() << "]" << std::endl;
    
    // Compute analytical gradient
    auto sim_result = compute_similarity(
      ref_geom1, ref_geom2, affine_dof, dofs2, kgrid,
      current_measure, V1_volume, SIM_FUNC_GRAD
    );
    Eigen::VectorXd grad_analytical = sim_result.gradient;
    double loss0 = sim_result.value;
    
    std::cout << "Analytical gradient norm: " << grad_analytical.norm() << std::endl;
    
    // Compute finite difference gradient using central differences
    double eps = 1e-6;
    Eigen::VectorXd grad_fd = Eigen::VectorXd::Zero(12);
    
    for (int i = 0; i < 12; ++i) {
      // Forward perturbation
      Eigen::VectorXd dofs_forward = dofs2;
      dofs_forward(i) += eps;
      auto sim_forward = compute_similarity(
        ref_geom1, ref_geom2, affine_dof, dofs_forward, kgrid,
        current_measure, V1_volume, SIM_FUNC_ONLY
      );
      double loss_forward = sim_forward.value;
      
      // Backward perturbation
      Eigen::VectorXd dofs_backward = dofs2;
      dofs_backward(i) -= eps;
      auto sim_backward = compute_similarity(
        ref_geom1, ref_geom2, affine_dof, dofs_backward, kgrid,
        current_measure, V1_volume, SIM_FUNC_ONLY
      );
      double loss_backward = sim_backward.value;
      
      // Central difference
      grad_fd(i) = (loss_forward - loss_backward) / (2.0 * eps);
    }
    
    std::cout << "Finite difference gradient norm: " << grad_fd.norm() << std::endl;
    
    // Compute errors
    Eigen::VectorXd grad_error = grad_analytical - grad_fd;
    double max_abs_error = grad_error.cwiseAbs().maxCoeff();
    double max_rel_error = 0.0;
    int max_error_idx = -1;
    
    for (int i = 0; i < 12; ++i) {
      double abs_val = std::abs(grad_analytical(i));
      double rel_error = (abs_val > 1e-10) ? std::abs(grad_error(i)) / abs_val : std::abs(grad_error(i));
      if (rel_error > max_rel_error) {
        max_rel_error = rel_error;
        max_error_idx = i;
      }
    }
    
    std::cout << "\nGradient Comparison:" << std::endl;
    std::cout << "  Max absolute error: " << max_abs_error << std::endl;
    std::cout << "  Max relative error: " << max_rel_error << " (at DoF " << max_error_idx << ")" << std::endl;
    std::cout << "\nPer-DoF comparison:" << std::endl;
    std::cout << "  DoF | Analytical | Finite Diff | Error | Rel Error" << std::endl;
    std::cout << "  ----|------------|------------|-------|----------" << std::endl;
    
    for (int i = 0; i < 12; ++i) {
      double abs_val = std::abs(grad_analytical(i));
      double rel_error = (abs_val > 1e-10) ? std::abs(grad_error(i)) / abs_val : std::abs(grad_error(i));
      std::cout << "  " << std::setw(3) << i << " | " 
                << std::scientific << std::setprecision(6)
                << std::setw(11) << grad_analytical(i) << " | "
                << std::setw(11) << grad_fd(i) << " | "
                << std::setw(6) << grad_error(i) << " | "
                << std::setw(9) << rel_error << std::endl;
    }
    
    // Determine if test passed
    double tolerance_abs = 1e-4;
    double tolerance_rel = 1e-3;
    bool passed = (max_abs_error < tolerance_abs) && (max_rel_error < tolerance_rel);
    
    std::cout << "\nTest Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Tolerance: abs < " << tolerance_abs << ", rel < " << tolerance_rel << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    return passed;
  };
  
  // Helper function to apply sign flip to transform
  auto apply_sign_flip = [&](bool flip_x, bool flip_y, bool flip_z) {
    // Extract current transform from dofs2
    Eigen::Vector3d t = dofs2.segment<3>(0);
    Eigen::Matrix3d A = Eigen::Map<const Eigen::Matrix3d>(dofs2.data() + 3).transpose();
    
    // Create sign flip matrix
    Eigen::Matrix3d flip_mat = Eigen::Matrix3d::Identity();
    if (flip_x) flip_mat(0, 0) = -1.0;
    if (flip_y) flip_mat(1, 1) = -1.0;
    if (flip_z) flip_mat(2, 2) = -1.0;
    
    // Apply sign flip: A_new = A * flip_mat
    Eigen::Matrix3d A_flipped = A * flip_mat;
    
    // Pack back into DoF vector
    dofs2.segment<3>(0) = t;
    dofs2.segment<3>(3) = A_flipped.row(0);
    dofs2.segment<3>(6) = A_flipped.row(1);
    dofs2.segment<3>(9) = A_flipped.row(2);
    
    // Update geometry and volumes
    update_geometry_and_volumes();
    update_mesh2_visualization();
  };
  
  // Custom menu callback
  menu.callback_draw_viewer_menu = [&]() {
    if (ImGui::CollapsingHeader("Mesh Alignment", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Display volumes
      ImGui::Text("Volumes:");
      ImGui::Text("  Mesh 1: %.6f", V1_volume);
      ImGui::Text("  Mesh 2: %.6f", V2_volume);
      ImGui::Text("  Intersection: %.6f", intersection_volume);
      
      ImGui::Separator();
      
      // Similarity measure selection
      ImGui::Text("Similarity Measure:");
      const char* measure_names[] = { "Ochiai", "Jaccard", "Dice" };
      int current_idx = static_cast<int>(current_measure);
      if (ImGui::Combo("Measure", &current_idx, measure_names, 3)) {
        current_measure = static_cast<SimilarityMeasure>(current_idx);
        update_geometry_and_volumes();
      }
      ImGui::Text("Current Similarity: %.2f%%", similarity * 100.0);
      ImGui::Text("Similarity Loss (-log): %.6f", similarity_loss);
      
      ImGui::Separator();
      
      // Sign flip controls
      ImGui::Text("Sign Flip (Flip axes of mesh 2):");
      if (ImGui::Button("Flip X")) {
        apply_sign_flip(true, false, false);
      }
      ImGui::SameLine();
      if (ImGui::Button("Flip Y")) {
        apply_sign_flip(false, true, false);
      }
      ImGui::SameLine();
      if (ImGui::Button("Flip Z")) {
        apply_sign_flip(false, false, true);
      }
      
      ImGui::Separator();
      
      // Manual transform controls
      if (ImGui::CollapsingHeader("Manual Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool transform_changed = false;
        
        // Translation controls
        ImGui::Text("Translation:");
        float tx = static_cast<float>(dofs2(0));
        float ty = static_cast<float>(dofs2(1));
        float tz = static_cast<float>(dofs2(2));
        
        if (ImGui::DragFloat("X", &tx, 0.01f, -10.0f, 10.0f, "%.3f")) {
          dofs2(0) = tx;
          transform_changed = true;
        }
        if (ImGui::DragFloat("Y", &ty, 0.01f, -10.0f, 10.0f, "%.3f")) {
          dofs2(1) = ty;
          transform_changed = true;
        }
        if (ImGui::DragFloat("Z", &tz, 0.01f, -10.0f, 10.0f, "%.3f")) {
          dofs2(2) = tz;
          transform_changed = true;
        }
        
        ImGui::Separator();
        
        // Matrix controls (3x3)
        ImGui::Text("Matrix (3x3):");
        float matrix[9];
        for (int i = 0; i < 9; ++i) {
          matrix[i] = static_cast<float>(dofs2(3 + i));
        }
        
        // Display as 3x3 grid
        if (ImGui::DragFloat3("Row 0", &matrix[0], 0.01f, -10.0f, 10.0f, "%.3f")) {
          for (int i = 0; i < 3; ++i) {
            dofs2(3 + i) = matrix[i];
          }
          transform_changed = true;
        }
        if (ImGui::DragFloat3("Row 1", &matrix[3], 0.01f, -10.0f, 10.0f, "%.3f")) {
          for (int i = 0; i < 3; ++i) {
            dofs2(6 + i) = matrix[3 + i];
          }
          transform_changed = true;
        }
        if (ImGui::DragFloat3("Row 2", &matrix[6], 0.01f, -10.0f, 10.0f, "%.3f")) {
          for (int i = 0; i < 3; ++i) {
            dofs2(9 + i) = matrix[6 + i];
          }
          transform_changed = true;
        }
        
        // Update if transform changed
        if (transform_changed) {
          update_geometry_and_volumes();
          update_mesh2_visualization();
        }
      }
      
      ImGui::Separator();
      
      // Gradient test button
      if (ImGui::Button("Test Gradient (Analytical vs Finite Diff)")) {
        test_gradient();
      }
      
      ImGui::Separator();
      
      // Optimization controls
      ImGui::Separator();
      ImGui::Text("Optimization:");
      
      ImGui::Checkbox("Step-by-step mode", &optimization_step_mode);
      ImGui::DragFloat("Tolerance", &opt_tolerance, 1e-7f, 1e-8f, 1e-3f, "%.2e");
      ImGui::DragInt("Max Iterations", &max_opt_iterations, 1, 1, 200);
      ImGui::DragFloat("Damping", &opt_damping, 0.01f, 0.01f, 1.0f, "%.3f");
      
      if (ImGui::Button("Run Full Optimization") && !optimization_running) {
        optimization_running = true;
        optimization_complete = false;
        current_opt_iter = 0;
        
        // Run full optimization
        opt_state = optimize_alignment(
          ref_geom1, ref_geom2, affine_dof, dofs2, kgrid,
          current_measure, V1_volume, opt_tolerance, max_opt_iterations, opt_damping
        );
        
        dofs2 = opt_state.dofs;
        update_geometry_and_volumes();
        update_mesh2_visualization();
        
        optimization_running = false;
        optimization_complete = true;
      }
      
      if (optimization_step_mode) {
        if (ImGui::Button("Step One Iteration") && !optimization_running) {
          optimization_running = true;
          
          // Initialize if needed
          if (current_opt_iter == 0) {
            opt_state.dofs = dofs2;
          }
          
          // Run one iteration using the unified optimizer (max_iterations = 1)
          opt_state = optimize_alignment(
            ref_geom1, ref_geom2, affine_dof, opt_state.dofs, kgrid,
            current_measure, V1_volume, opt_tolerance, 1, opt_damping
          );
          
          current_opt_iter = opt_state.iteration;
          if (opt_state.converged || current_opt_iter >= max_opt_iterations - 1) {
            optimization_complete = true;
          }
          
          dofs2 = opt_state.dofs;
          update_geometry_and_volumes();
          update_mesh2_visualization();
          
          optimization_running = false;
        }
        
        if (ImGui::Button("Reset Optimization")) {
          current_opt_iter = 0;
          optimization_complete = false;
          opt_state.converged = false;
        }
      }
      
      if (optimization_complete || current_opt_iter > 0) {
        ImGui::Separator();
        ImGui::Text("Optimization Status:");
        if (opt_state.converged) {
          ImGui::TextColored(ImVec4(0, 1, 0, 1), "  %s", opt_state.status.c_str());
        } else {
          ImGui::Text("  %s", opt_state.status.c_str());
        }
        ImGui::Text("  Iteration: %d / %d", opt_state.iteration, max_opt_iterations);
        ImGui::Text("  Similarity: %.2f%%", opt_state.similarity * 100.0);
        ImGui::Text("  Similarity Loss (-log): %.6f", opt_state.similarity_loss);
      }
      
      if (ImGui::Button("Reset Transform")) {
        dofs2 << 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0;
        update_geometry_and_volumes();
        update_mesh2_visualization();
        optimization_complete = false;
      }
    }
    
    // Mesh info
    if (ImGui::CollapsingHeader("Mesh Info")) {
      ImGui::Text("Mesh 1: %d vertices, %d faces", (int)V1.rows(), (int)F1.rows());
      ImGui::Text("Mesh 2: %d vertices, %d faces", (int)V2.rows(), (int)F2.rows());
      ImGui::Text("KGrid: %d k-nodes", (int)kgrid.kmag.size());
    }
  };
  
  // Initial visualization update
  update_mesh2_visualization();
  
  std::cout << R"(
Controls:
  [Use ImGui menu to run optimization]
)" << std::endl;
  
  // Launch viewer
  viewer.launch();
  
  return 0;
}

