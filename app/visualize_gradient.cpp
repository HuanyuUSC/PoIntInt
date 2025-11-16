#include <Eigen/Dense>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <igl/combine.h>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "compute_intersection_volume.hpp"
#include "compute_volume.hpp"
#include "dof/affine_dof.hpp"
#include "dof/triangle_mesh_dof.hpp"
#include "geometry/packing.hpp"
#include "geometry/geometry_helpers.hpp"
#include "quadrature/lebedev_io.hpp"
#include "quadrature/kgrid.hpp"

using namespace PoIntInt;

// Helper function to find the project data directory
std::string find_data_directory() {
  // Try multiple possible locations
  std::vector<std::string> candidates;
  
  // 1. Check if POINTINT_SOURCE_DIR is defined (from CMake)
#ifdef POINTINT_SOURCE_DIR
  std::string source_dir = POINTINT_SOURCE_DIR;
  // Normalize path separators
  for (size_t i = 0; i < source_dir.length(); ++i) {
    if (source_dir[i] == '\\') {
      source_dir[i] = '/';
    }
  }
  candidates.push_back(source_dir + "/data/lebedev");
#endif
  
  // 2. Try relative to current working directory
  candidates.push_back("data/lebedev");
  candidates.push_back("../data/lebedev");
  candidates.push_back("../../data/lebedev");
  
  // 3. Try relative to executable location (if available)
  // Note: This would require platform-specific code, so we skip it for simplicity
  
  // Check each candidate
  for (const auto& candidate : candidates) {
    std::filesystem::path path(candidate);
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
      return candidate;
    }
  }
  
  // Default fallback
  return "data/lebedev";
}

// Helper function to format Lebedev number as 3-digit string (e.g., 29 -> "029")
std::string format_lebedev_number(int number) {
  if (number < 10) {
    return "00" + std::to_string(number);
  } else if (number < 100) {
    return "0" + std::to_string(number);
  } else {
    return std::to_string(number);
  }
}

// Helper function to get Lebedev file path from number
std::string get_lebedev_file_path(int lebedev_number) {
  std::string data_dir = find_data_directory();
  std::string filename = "lebedev_" + format_lebedev_number(lebedev_number) + ".txt";
  return (std::filesystem::path(data_dir) / filename).string();
}

// Helper function to get default Lebedev number (29 is a good default)
int get_default_lebedev_number() {
  return 29;
}

// Global state
Eigen::MatrixXd V1, V2;
Eigen::MatrixXi F1, F2;
Geometry geom1, geom2;
std::shared_ptr<TriangleMeshDoF> dof1, dof2;
Eigen::VectorXd dofs1, dofs2;
Eigen::VectorXd dofs2_base;  // Base DoFs for mesh 2 (original positions)
KGrid kgrid;
Eigen::VectorXd grad1, grad2;  // Gradients for visualization (per-vertex)
Eigen::VectorXd grad1_magnitude, grad2_magnitude;  // Gradient magnitudes per vertex
double intersection_volume = 0.0;
bool gradients_computed = false;
double self_volume1 = 0.0;
double self_volume1_gt = 0.0;
double self_volume2 = 0.0;
double self_volume2_gt = 0.0;
Eigen::Vector3d translation2(0.0, 0.0, 0.0);  // Translation for mesh 2

// Color mapping: map gradient magnitude to vertex colors
Eigen::MatrixXd gradient_to_colors(const Eigen::VectorXd& grad, double min_val = 0.0, double max_val = 1.0) {
  Eigen::MatrixXd C(grad.size(), 3);
  
  // Normalize gradient to [0, 1]
  double grad_min = grad.minCoeff();
  double grad_max = grad.maxCoeff();
  double range = grad_max - grad_min;
  
  if (range < 1e-10) {
    // Uniform color if gradient is constant
    C.setConstant(0.5);
    return C;
  }
  
  // Map to [0, 1] then to color
  for (int i = 0; i < grad.size(); ++i) {
    double normalized = (grad(i) - grad_min) / range;
    // Use a colormap: blue (low) -> white (middle) -> red (high)
    if (normalized < 0.5) {
      // Blue to white
      double t = normalized * 2.0;
      C(i, 0) = t;
      C(i, 1) = t;
      C(i, 2) = 1.0;
    } else {
      // White to red
      double t = (normalized - 0.5) * 2.0;
      C(i, 0) = 1.0;
      C(i, 1) = 1.0 - t;
      C(i, 2) = 1.0 - t;
    }
  }
  
  return C;
}

// Compute gradients
void compute_gradients() {
  if (!gradients_computed) {
    std::cout << "Computing intersection volume gradients..." << std::endl;
    
    // Compute intersection volume with gradients
    auto result = compute_intersection_volume_cuda(
      geom1, geom2, dof1, dof2, dofs1, dofs2, kgrid,
      ComputationFlags::VOLUME_ONLY | ComputationFlags::GRADIENT,
      256, false
    );
    
    intersection_volume = result.volume;
    grad1 = result.grad_geom1;
    grad2 = result.grad_geom2;
    
    gradients_computed = true;
    std::cout << "Intersection volume: " << intersection_volume << std::endl;
    std::cout << "Gradient 1 norm: " << grad1.norm() << std::endl;
    std::cout << "Gradient 2 norm: " << grad2.norm() << std::endl;
  }
}

void test_gradient_correctness()
{
  // Finite difference check for gradient correctness
  double epsilon = 1e-6;
  compute_gradients();

  for (int j = 0; j < 3; j++)
  {
    Eigen::VectorXd dofs1_perturbed = dofs1;
    for (int i = 0; i < V1.rows(); i++) {
      dofs1_perturbed(i * 3 + j) += epsilon;
    }
    double vplus = compute_intersection_volume_cuda(
      geom1, geom2, dof1, dof2, dofs1_perturbed, dofs2, kgrid,
      ComputationFlags::VOLUME_ONLY,
      256, false
    ).volume;

    for (int i = 0; i < V1.rows(); i++) {
      dofs1_perturbed(i * 3 + j) -= 2.0 * epsilon;
    }
    double vminus = compute_intersection_volume_cuda(
      geom1, geom2, dof1, dof2, dofs1_perturbed, dofs2, kgrid,
      ComputationFlags::VOLUME_ONLY,
      256, false
    ).volume;

    double finite_diff_grad = (vplus - vminus) / (2.0 * epsilon);
    double analytic_grad = 0.0;
    for (int i = 0; i < V1.rows(); i++) {
      analytic_grad += grad1[3 * i + j];
    }
    printf("Object 1 %d-grad: %g vs %g, rel err %g%%\n", j, analytic_grad, finite_diff_grad,
      abs(analytic_grad - finite_diff_grad) / finite_diff_grad * 100.0);
  }

}

int main(int argc, char* argv[]) {
  // Parse command line arguments
  std::string mesh1_file, mesh2_file, leb_file;
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <mesh1.off> <mesh2.off> [lebedev_number] [Nrad]" << std::endl;
    std::cerr << "  lebedev_number: odd number in range [3, 131] (default: 29)" << std::endl;
    std::cerr << "  Nrad: number of radial quadrature points (default: 32)" << std::endl;
    std::cerr << "  Example: " << argv[0] << " mesh1.off mesh2.off 29 32" << std::endl;
    return 1;
  }
  
  mesh1_file = argv[1];
  mesh2_file = argv[2];
  
  // Parse Lebedev number (odd number in [3, 131])
  int lebedev_number = (argc > 3) ? std::atoi(argv[3]) : get_default_lebedev_number();
  if (lebedev_number < 3 || lebedev_number > 131 || lebedev_number % 2 == 0) {
    std::cerr << "Error: lebedev_number must be an odd number in range [3, 131]" << std::endl;
    std::cerr << "  Got: " << lebedev_number << std::endl;
    return 1;
  }
  
  // Construct Lebedev file path from number
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
  geom1 = make_triangle_mesh(V1, F1);
  geom2 = make_triangle_mesh(V2, F2);
  
  // Create TriangleMeshDoF for both meshes (allows per-vertex gradients)
  dof1 = std::make_shared<TriangleMeshDoF>(V1.rows());
  dof2 = std::make_shared<TriangleMeshDoF>(V2.rows());
  
  // Set DoFs as flattened vertex positions
  dofs1 = Eigen::VectorXd(V1.rows() * 3);
  dofs2 = Eigen::VectorXd(V2.rows() * 3);
  dofs2_base = Eigen::VectorXd(V2.rows() * 3);
  for (int i = 0; i < V1.rows(); ++i) {
    dofs1(i * 3 + 0) = V1(i, 0);
    dofs1(i * 3 + 1) = V1(i, 1);
    dofs1(i * 3 + 2) = V1(i, 2);
  }
  for (int i = 0; i < V2.rows(); ++i) {
    dofs2_base(i * 3 + 0) = V2(i, 0);
    dofs2_base(i * 3 + 1) = V2(i, 1);
    dofs2_base(i * 3 + 2) = V2(i, 2);
    dofs2(i * 3 + 0) = V2(i, 0);
    dofs2(i * 3 + 1) = V2(i, 1);
    dofs2(i * 3 + 2) = V2(i, 2);
  }
  
  // Load Lebedev grid and build KGrid
  std::cout << "Loading Lebedev grid: " << leb_file << std::endl;
  LebedevGrid L = load_lebedev_txt(leb_file);
  kgrid = build_kgrid(L.dirs, L.weights, Nrad);
  std::cout << "KGrid: " << kgrid.kmag.size() << " k-nodes" << std::endl;

  self_volume1 = compute_intersection_volume_cuda(geom1, geom1, dof1, dof1, dofs1, dofs1, 
    kgrid, ComputationFlags::VOLUME_ONLY, 256, false).volume;
  self_volume1_gt = compute_volume_cpu(geom1, false);
  printf("Object 1, self volume: %g vs %g, rel err %g%%.\n", self_volume1, self_volume1_gt, 
    abs(self_volume1 - self_volume1_gt) / self_volume1_gt * 100.0);

  self_volume2 = compute_intersection_volume_cuda(geom2, geom2, dof2, dof2, dofs2, dofs2,
    kgrid, ComputationFlags::VOLUME_ONLY, 256, false).volume;
  self_volume2_gt = compute_volume_cpu(geom2, false);
  printf("Object 2, self volume: %g vs %g, rel err %g%%.\n", self_volume2, self_volume2_gt,
    abs(self_volume2 - self_volume2_gt) / self_volume2_gt * 100.0);

  // Compute initial intersection volume
  auto initial_result = compute_intersection_volume_cuda(
    geom1, geom2, dof1, dof2, dofs1, dofs2, kgrid,
    ComputationFlags::VOLUME_ONLY,
    256, false
  );
  intersection_volume = initial_result.volume;
  printf("Initial intersection volume: %.6f\n", intersection_volume);
  
  // Initialize gradients (will be computed on first draw)
  grad1 = Eigen::VectorXd::Zero(V1.rows() * 3);
  grad2 = Eigen::VectorXd::Zero(V2.rows() * 3);
  grad1_magnitude = Eigen::VectorXd::Zero(V1.rows());
  grad2_magnitude = Eigen::VectorXd::Zero(V2.rows());
  
  // Set up viewer
  igl::opengl::glfw::Viewer viewer;
  
  // Set mesh 1 in the first data slot
  viewer.data().set_mesh(V1, F1);
  viewer.data().set_colors(Eigen::MatrixXd::Ones(V1.rows(), 3) * 0.8);  // Light gray default
  
  // Append mesh 2 as a separate data object
  viewer.append_mesh();
  viewer.data().set_mesh(V2, F2);
  viewer.data().set_colors(Eigen::MatrixXd::Ones(V2.rows(), 3) * 0.8);  // Light gray default
  
  // Store references to both mesh data objects for easy access
  auto& data1 = viewer.data_list[0];  // Mesh 1
  auto& data2 = viewer.data_list[1];  // Mesh 2
  
  // Track visibility state
  bool mesh1_visible = true;
  bool mesh2_visible = true;
  
  // Set up ImGui
  igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
  viewer.plugins.push_back(&imgui_plugin);
  
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  imgui_plugin.widgets.push_back(&menu);
  
  // Function to update mesh 2 translation and recompute intersection volume
  auto update_translation_and_volume = [&]() {
    // Update dofs2 with translation
    for (int i = 0; i < V2.rows(); ++i) {
      dofs2(i * 3 + 0) = dofs2_base(i * 3 + 0) + translation2(0);
      dofs2(i * 3 + 1) = dofs2_base(i * 3 + 1) + translation2(1);
      dofs2(i * 3 + 2) = dofs2_base(i * 3 + 2) + translation2(2);
    }
    
    // Update mesh 2 geometry with new positions
    Eigen::MatrixXd V2_translated = V2;
    for (int i = 0; i < V2_translated.rows(); ++i) {
      V2_translated.row(i) += translation2.transpose();
    }
    
    // Update mesh 2 in viewer
    data2.set_mesh(V2_translated, F2);
    if (gradients_computed) {
      // Preserve colors if gradients were computed
      Eigen::MatrixXd C2(V2.rows(), 3);
      for (int i = 0; i < V2.rows(); ++i) {
        double normalized = std::min(1.0, std::max(0.0, grad2_magnitude(i) / 
          std::max(grad1_magnitude.maxCoeff(), grad2_magnitude.maxCoeff())));
        if (normalized < 0.5) {
          double t = normalized * 2.0;
          C2(i, 0) = t; C2(i, 1) = t; C2(i, 2) = 1.0;
        } else {
          double t = (normalized - 0.5) * 2.0;
          C2(i, 0) = 1.0; C2(i, 1) = 1.0 - t; C2(i, 2) = 1.0 - t;
        }
      }
      data2.set_colors(C2);
    } else {
      data2.set_colors(Eigen::MatrixXd::Ones(V2.rows(), 3) * 0.8);
    }
    
    // Recompute intersection volume (without gradient)
    auto result = compute_intersection_volume_cuda(
      geom1, geom2, dof1, dof2, dofs1, dofs2, kgrid,
      ComputationFlags::VOLUME_ONLY,
      256, false
    );
    intersection_volume = result.volume;
    
    // Invalidate gradients since geometry changed
    gradients_computed = false;
  };
  
  // Custom menu callback
  menu.callback_draw_viewer_menu = [&]() {
    if (ImGui::CollapsingHeader("Intersection Volume Gradient", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Display intersection volume
      ImGui::Text("Intersection Volume: %.6f", intersection_volume);
      
      ImGui::Separator();
      ImGui::Text("Mesh 2 Translation:");
      bool translation_changed = false;
      float tx = static_cast<float>(translation2(0));
      float ty = static_cast<float>(translation2(1));
      float tz = static_cast<float>(translation2(2));
      
      // Use DragFloat for input boxes with incremental changing (drag or type)
      if (ImGui::DragFloat("X", &tx, 0.01f, -5.0f, 5.0f, "%.3f")) {
        translation2(0) = static_cast<double>(tx);
        translation_changed = true;
      }
      if (ImGui::DragFloat("Y", &ty, 0.01f, -5.0f, 5.0f, "%.3f")) {
        translation2(1) = static_cast<double>(ty);
        translation_changed = true;
      }
      if (ImGui::DragFloat("Z", &tz, 0.01f, -5.0f, 5.0f, "%.3f")) {
        translation2(2) = static_cast<double>(tz);
        translation_changed = true;
      }
      
      if (translation_changed) {
        update_translation_and_volume();
      }
      
      if (ImGui::Button("Reset Translation")) {
        translation2.setZero();
        update_translation_and_volume();
      }
      
      ImGui::Separator();
      
      if (ImGui::Button("Compute Gradients")) {
        compute_gradients();
        
        // Compute per-vertex gradient magnitudes
        // grad1 and grad2 are 3*num_vertices vectors: [v0_x, v0_y, v0_z, v1_x, ...]
        for (int i = 0; i < V1.rows(); ++i) {
          Eigen::Vector3d g(grad1(i * 3 + 0), grad1(i * 3 + 1), grad1(i * 3 + 2));
          grad1_magnitude(i) = g.norm();
        }
        for (int i = 0; i < V2.rows(); ++i) {
          Eigen::Vector3d g(grad2(i * 3 + 0), grad2(i * 3 + 1), grad2(i * 3 + 2));
          grad2_magnitude(i) = g.norm();
        }
        
        // Normalize gradient magnitudes for color mapping
        double max_grad1 = grad1_magnitude.maxCoeff();
        double max_grad2 = grad2_magnitude.maxCoeff();
        double max_grad = std::max(max_grad1, max_grad2);
        if (max_grad < 1e-10) max_grad = 1.0;  // Avoid division by zero
        
        // Update colors: mesh1 vertices get color based on grad1 magnitude, mesh2 based on grad2 magnitude
        Eigen::MatrixXd C1(V1.rows(), 3);
        Eigen::MatrixXd C2(V2.rows(), 3);
        
        for (int i = 0; i < V1.rows(); ++i) {
          double normalized = std::min(1.0, std::max(0.0, grad1_magnitude(i) / max_grad));
          // Blue (low) -> white (middle) -> red (high) colormap
          if (normalized < 0.5) {
            double t = normalized * 2.0;
            C1(i, 0) = t;
            C1(i, 1) = t;
            C1(i, 2) = 1.0;
          } else {
            double t = (normalized - 0.5) * 2.0;
            C1(i, 0) = 1.0;
            C1(i, 1) = 1.0 - t;
            C1(i, 2) = 1.0 - t;
          }
        }
        
        for (int i = 0; i < V2.rows(); ++i) {
          double normalized = std::min(1.0, std::max(0.0, grad2_magnitude(i) / max_grad));
          if (normalized < 0.5) {
            double t = normalized * 2.0;
            C2(i, 0) = t;
            C2(i, 1) = t;
            C2(i, 2) = 1.0;
          } else {
            double t = (normalized - 0.5) * 2.0;
            C2(i, 0) = 1.0;
            C2(i, 1) = 1.0 - t;
            C2(i, 2) = 1.0 - t;
          }
        }
        
        data1.set_colors(C1);
        data2.set_colors(C2);
      }

      if (gradients_computed) {
        ImGui::Separator();
        ImGui::Text("Gradient Statistics:");
        if (grad1_magnitude.size() > 0) {
          ImGui::Text("Mesh 1:");
          ImGui::Text("  Min magnitude: %.6f", grad1_magnitude.minCoeff());
          ImGui::Text("  Max magnitude: %.6f", grad1_magnitude.maxCoeff());
          ImGui::Text("  Mean magnitude: %.6f", grad1_magnitude.mean());
        }
        if (grad2_magnitude.size() > 0) {
          ImGui::Text("Mesh 2:");
          ImGui::Text("  Min magnitude: %.6f", grad2_magnitude.minCoeff());
          ImGui::Text("  Max magnitude: %.6f", grad2_magnitude.maxCoeff());
          ImGui::Text("  Mean magnitude: %.6f", grad2_magnitude.mean());
        }
        if (ImGui::Button("Test Gradient Correctness"))
        {
          test_gradient_correctness();
        }
      }
      
      if (ImGui::Button("Reset Colors")) {
        data1.set_colors(Eigen::MatrixXd::Ones(V1.rows(), 3) * 0.8);
        data2.set_colors(Eigen::MatrixXd::Ones(V2.rows(), 3) * 0.8);
      }
      
      ImGui::Separator();
      ImGui::Text("Visibility Controls:");
      if (ImGui::Checkbox("Mesh 1 Visible", &mesh1_visible)) {
        data1.is_visible = mesh1_visible ? 1 : 0;
      }
      if (ImGui::Checkbox("Mesh 2 Visible", &mesh2_visible)) {
        data2.is_visible = mesh2_visible ? 1 : 0;
      }
    }
    
    // Mesh info
    if (ImGui::CollapsingHeader("Mesh Info")) {
      ImGui::Text("Mesh 1: %d vertices, %d faces", (int)V1.rows(), (int)F1.rows());
      ImGui::Text("Mesh 2: %d vertices, %d faces", (int)V2.rows(), (int)F2.rows());
      ImGui::Text("KGrid: %d k-nodes", (int)kgrid.kmag.size());
    }
  };
  
  // Keyboard callback
  viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key, int) {
    switch (key) {
      case 'G':
      case 'g':
        compute_gradients();
        return true;
      case 'R':
      case 'r':
        data1.set_colors(Eigen::MatrixXd::Ones(V1.rows(), 3) * 0.8);
        data2.set_colors(Eigen::MatrixXd::Ones(V2.rows(), 3) * 0.8);
        return true;
      case '1':
        mesh1_visible = !mesh1_visible;
        data1.is_visible = mesh1_visible ? 1 : 0;
        std::cout << "Mesh 1 visibility: " << (mesh1_visible ? "ON" : "OFF") << std::endl;
        return true;
      case '2':
        mesh2_visible = !mesh2_visible;
        data2.is_visible = mesh2_visible ? 1 : 0;
        std::cout << "Mesh 2 visibility: " << (mesh2_visible ? "ON" : "OFF") << std::endl;
        return true;
    }
    return false;
  };
  
  std::cout << R"(
Controls:
  G/g     Compute intersection volume gradients
  R/r     Reset colors to default
  1       Toggle mesh 1 visibility
  2       Toggle mesh 2 visibility
  [Click "Compute Gradients" button in menu]
)" << std::endl;
  
  // Launch viewer
  viewer.launch();
  
  return 0;
}

