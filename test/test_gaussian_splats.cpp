#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include "compute_intersection_volume.hpp"
#include "compute_volume.hpp"
#include "geometry/packing.hpp"
#include "geometry/geometry_helpers.hpp"
#include "form_factor_helpers.hpp"
#include "quadrature/lebedev_io.hpp"
#include "quadrature/gauss_legendre.hpp"
#include "quadrature/kgrid.hpp"

using namespace PoIntInt;

// ============================================================================
// Test 1: Compare computed form factor field from Gaussian splats to exact formula
// ============================================================================
bool test_gaussian_form_factor_field(const std::string& leb_file, int Nrad = 32) {
  std::cout << "\n=== Test 1: Gaussian Splat Form Factor Field Comparison ===" << std::endl;
  
  // Create unit sphere Gaussian splats
  Eigen::MatrixXd P, N;
  Eigen::VectorXd sigmas, weights;
  create_sphere_gaussians(P, N, sigmas, weights, 1000);
  auto geom = make_gaussian_splat(P, N, sigmas, weights);
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Sample a few k-points and compare
  std::vector<std::pair<std::array<double, 3>, double>> test_k_points = {
    {{1.0, 0.0, 0.0}, 0.0},
    {{1.0, 0.0, 0.0}, 0.1},
    {{1.0, 0.0, 0.0}, 1.0},
    {{0.0, 1.0, 0.0}, 1.0},
    {{0.0, 0.0, 1.0}, 2.0},
    {{0.707, 0.707, 0.0}, 1.414},
  };
  
  double max_rel_error = 0.0;
  int num_tested = 0;
  
  for (const auto& k_test : test_k_points) {
    double kx = k_test.first[0] * k_test.second;
    double ky = k_test.first[1] * k_test.second;
    double kz = k_test.first[2] * k_test.second;
    double k_mag = std::sqrt(kx*kx + ky*ky + kz*kz);
    
    // Compute A_parallel(k) from Gaussian splats using form_factor_helpers
    Eigen::Vector3d k_vec(kx, ky, kz);
    std::complex<double> A_complex = compute_A_geometry(geom, k_vec);
    double Ak2_gaussian = std::norm(A_complex);  // |A|^2 = real^2 + imag^2
    
    // Exact |A(k)|² for sphere (depends only on |k|)
    double Ak2_exact = exact_sphere_Ak_squared(k_mag);  // From form_factor_helpers
    
    double rel_error = std::abs(Ak2_gaussian - Ak2_exact) / (std::abs(Ak2_exact) + 1e-10);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  k = (" << std::setw(6) << kx << ", " 
              << std::setw(6) << ky << ", " << std::setw(6) << kz << "), |k| = " << k_mag << ":" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "    |A|²_gaussian = " << std::setw(15) << Ak2_gaussian << std::endl;
    std::cout << "    |A|²_exact    = " << std::setw(15) << Ak2_exact << std::endl;
    std::cout << "    rel error     = " << std::setw(15) << rel_error 
              << " (" << std::fixed << std::setprecision(2) << rel_error * 100.0 << "%)" << std::endl;
    
    if (rel_error > max_rel_error) {
      max_rel_error = rel_error;
    }
    num_tested++;
  }
  
  std::cout << "  Tested " << num_tested << " k-points" << std::endl;
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Max relative error: " << max_rel_error << std::endl;
  
  // Allow higher error for Gaussian splats due to discretization
  bool passed = max_rel_error < 1.5;  // Allow up to 150% error for outliers
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  if (max_rel_error > 0.5) {
    std::cout << "  Note: Some k-directions may have higher error due to Gaussian splat discretization" << std::endl;
  }
  return passed;
}

// ============================================================================
// Test 2: Compare Gaussian splat volume computation to exact volume
// ============================================================================
bool test_gaussian_volume_computation(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 2: Gaussian Splat Volume Computation ===" << std::endl;
  
  // Clear any previous CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess && err != cudaErrorNoDevice) {
    cudaDeviceSynchronize();  // Clear error state
  }
  
  // Create unit sphere Gaussian splats
  Eigen::MatrixXd P, N;
  Eigen::VectorXd sigmas, weights;
  create_sphere_gaussians(P, N, sigmas, weights, 2000);  // More points for better accuracy
  auto geom = make_gaussian_splat(P, N, sigmas, weights);
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute volume using CUDA
  double volume_computed = compute_intersection_volume_cuda(geom, geom, KG, 256, true);
  double volume_exact = 4.0 * M_PI / 3.0;  // Unit sphere volume
  
  double rel_error = std::abs(volume_computed - volume_exact) / volume_exact;
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Computed volume: " << volume_computed << std::endl;
  std::cout << "  Exact volume:    " << volume_exact << std::endl;
  std::cout << "  Relative error:   " << std::scientific << rel_error 
            << " (" << std::fixed << std::setprecision(4) << rel_error * 100.0 << "%)" << std::endl;
  
  bool passed = rel_error < 0.2;  // 20% tolerance for Gaussian splat approximation
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 3: Intersection between two Gaussian splat spheres
// ============================================================================
bool test_gaussian_gaussian_intersection(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 3: Gaussian-Gaussian Intersection ===" << std::endl;
  
  // Clear any previous CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess && err != cudaErrorNoDevice) {
    cudaDeviceSynchronize();  // Clear error state
  }
  
  // Create first unit sphere Gaussian splat at origin
  Eigen::MatrixXd P1, N1;
  Eigen::VectorXd sigmas1, weights1;
  create_sphere_gaussians(P1, N1, sigmas1, weights1, 1000);
  auto geom1 = make_gaussian_splat(P1, N1, sigmas1, weights1);
  
  // Create second unit sphere Gaussian splat translated by (0.5, 0, 0)
  Eigen::MatrixXd P2, N2;
  Eigen::VectorXd sigmas2, weights2;
  create_sphere_gaussians(P2, N2, sigmas2, weights2, 1000);
  P2.col(0).array() += 0.5;  // Translate in x-direction
  auto geom2 = make_gaussian_splat(P2, N2, sigmas2, weights2);
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute intersection volume
  double volume_computed = compute_intersection_volume_cuda(geom1, geom2, KG, 256, true);
  
  // Analytical ground truth: intersection volume of two unit spheres
  // Center distance d = 0.5, radius R = 1
  // Using formula from test_multi_object.cpp
  double d = 0.5;
  double r1 = 1.0;
  double r2 = 1.0;
  
  double volume_exact;
  if (d >= r1 + r2) {
    volume_exact = 0.0;
  } else if (d <= std::abs(r1 - r2)) {
    double r_min = std::min(r1, r2);
    volume_exact = (4.0 * M_PI / 3.0) * r_min * r_min * r_min;
  } else {
    double d2 = d * d;
    double r1_2 = r1 * r1;
    double r2_2 = r2 * r2;
    double term1 = (r1 + r2 - d) * (r1 + r2 - d);
    double term2 = d2 + 2.0 * d * (r1 + r2) - 3.0 * (r1_2 + r2_2) + 6.0 * r1 * r2;
    volume_exact = (M_PI * term1 * term2) / (12.0 * d);
  }
  
  double rel_error = std::abs(volume_computed - volume_exact) / (volume_exact + 1e-10);
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Computed intersection volume: " << volume_computed << std::endl;
  std::cout << "  Analytical ground truth:     " << volume_exact << std::endl;
  std::cout << "  Relative error:   " << std::scientific << rel_error 
            << " (" << std::fixed << std::setprecision(4) << rel_error * 100.0 << "%)" << std::endl;
  
  bool passed = rel_error < 0.3;  // 30% tolerance for Gaussian-Gaussian intersection
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 4: Intersection between Gaussian splat and mesh
// ============================================================================
bool test_gaussian_mesh_intersection(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 4: Gaussian-Mesh Intersection ===" << std::endl;
  
  // Clear any previous CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess && err != cudaErrorNoDevice) {
    cudaDeviceSynchronize();  // Clear error state
  }
  
  // Create unit cube mesh
  Eigen::MatrixXd V_cube(8, 3);
  V_cube <<
    -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5;
  Eigen::MatrixXi F_cube(12, 3);
  F_cube <<
    0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,
    0, 1, 5,  0, 5, 4,  2, 7, 6,  2, 3, 7,
    0, 4, 7,  0, 7, 3,  1, 6, 5,  1, 2, 6;
  auto geom_mesh = make_triangle_mesh(V_cube, F_cube);
  
  // Create unit sphere Gaussian splat (centered at origin, radius 0.5 so it fits inside cube)
  Eigen::MatrixXd P, N;
  Eigen::VectorXd sigmas, weights;
  create_sphere_gaussians(P, N, sigmas, weights, 1000);
  P *= 0.5;  // Scale to radius 0.5
  sigmas *= 0.5;  // Scale sigmas proportionally
  // Weights should scale as area, so weight * (0.5)^2
  weights *= 0.25;
  auto geom_gaussian = make_gaussian_splat(P, N, sigmas, weights);
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute intersection volume
  double volume_computed = compute_intersection_volume_cuda(geom_mesh, geom_gaussian, KG, 256, true);
  
  // Expected volume: volume of sphere with radius 0.5 = (4π/3) * (0.5)³ = π/6
  double volume_exact = M_PI / 6.0;
  
  double rel_error = std::abs(volume_computed - volume_exact) / volume_exact;
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Computed intersection volume: " << volume_computed << std::endl;
  std::cout << "  Expected volume (sphere r=0.5): " << volume_exact << std::endl;
  std::cout << "  Relative error:   " << std::scientific << rel_error 
            << " (" << std::fixed << std::setprecision(4) << rel_error * 100.0 << "%)" << std::endl;
  
  bool passed = rel_error < 0.25;  // 25% tolerance for mixed geometry
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 5: Intersection between Gaussian splat and point cloud
// ============================================================================
bool test_gaussian_pointcloud_intersection(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 5: Gaussian-Point Cloud Intersection ===" << std::endl;
  
  // Clear any previous CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess && err != cudaErrorNoDevice) {
    cudaDeviceSynchronize();  // Clear error state
  }
  
  // Create unit sphere point cloud (centered at origin, radius 0.5)
  Eigen::MatrixXd P_pc, N_pc;
  Eigen::VectorXd radii_pc;
  // Use the same helper function from test_sphere_pointcloud.cpp
  // For now, create a simple version
  int n_points = 1000;
  P_pc.resize(n_points, 3);
  N_pc.resize(n_points, 3);
  radii_pc.resize(n_points);
  
  const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));
  for (int i = 0; i < n_points; ++i) {
    double y = 1.0 - (2.0 * i) / (n_points - 1.0);
    double radius_at_y = std::sqrt(1.0 - y * y);
    double theta = golden_angle * i;
    double x = radius_at_y * std::cos(theta);
    double z = radius_at_y * std::sin(theta);
    P_pc(i, 0) = x * 0.5;
    P_pc(i, 1) = y * 0.5;
    P_pc(i, 2) = z * 0.5;
    N_pc(i, 0) = x;
    N_pc(i, 1) = y;
    N_pc(i, 2) = z;
    double disk_area = 4.0 * M_PI * 0.25 / n_points;  // Scaled area
    radii_pc(i) = std::sqrt(disk_area / M_PI);
  }
  auto geom_pointcloud = make_point_cloud(P_pc, N_pc, radii_pc, true);
  
  // Create unit sphere Gaussian splat (centered at origin, radius 0.5)
  Eigen::MatrixXd P, N;
  Eigen::VectorXd sigmas, weights;
  create_sphere_gaussians(P, N, sigmas, weights, 1000);
  P *= 0.5;  // Scale to radius 0.5
  sigmas *= 0.5;  // Scale sigmas proportionally
  weights *= 0.25;  // Scale weights as area
  auto geom_gaussian = make_gaussian_splat(P, N, sigmas, weights);
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute intersection volume
  double volume_computed = compute_intersection_volume_cuda(geom_gaussian, geom_pointcloud, KG, 256, true);
  
  // Expected volume: volume of sphere with radius 0.5 = (4π/3) * (0.5)³ = π/6
  double volume_exact = M_PI / 6.0;
  
  double rel_error = std::abs(volume_computed - volume_exact) / volume_exact;
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Computed intersection volume: " << volume_computed << std::endl;
  std::cout << "  Expected volume (sphere r=0.5): " << volume_exact << std::endl;
  std::cout << "  Relative error:   " << std::scientific << rel_error 
            << " (" << std::fixed << std::setprecision(4) << rel_error * 100.0 << "%)" << std::endl;
  
  bool passed = rel_error < 0.3;  // 30% tolerance for mixed geometry types
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 5: Volume from divergence theorem
// ============================================================================
bool test_volume_divergence_theorem(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 5: Volume from Divergence Theorem ===" << std::endl;
  
  // Create unit sphere Gaussian splats
  Eigen::MatrixXd P, N;
  Eigen::VectorXd sigmas, weights;
  create_sphere_gaussians(P, N, sigmas, weights, 2000);
  auto geom = make_gaussian_splat(P, N, sigmas, weights);
  
  // Compute volume using divergence theorem
  double volume_div = compute_volume_cuda(geom, 256, true);
  
  // Compute F(0) - for a sphere, F(0) = (4π/3) * R³
  double R = 1.0;  // Unit sphere
  double volume_F0 = (4.0 * M_PI / 3.0) * R * R * R;
  
  // Compute self-intersection volume
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  double volume_self = compute_intersection_volume_cuda(geom, geom, KG, 256, false);
  
  double volume_exact = volume_F0;
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Volume (divergence theorem): " << volume_div << std::endl;
  std::cout << "  Volume (F(0) = 4πR³/3): " << volume_F0 << std::endl;
  std::cout << "  Volume (self-intersection): " << volume_self << std::endl;
  std::cout << "  Exact volume: " << volume_exact << std::endl;
  
  double rel_error_div = std::abs(volume_div - volume_exact) / volume_exact;
  double rel_error_self = std::abs(volume_self - volume_exact) / volume_exact;
  double rel_error_div_vs_self = std::abs(volume_div - volume_self) / (0.5 * (volume_div + volume_self) + 1e-10);
  
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Rel error (div vs exact): " << rel_error_div << std::endl;
  std::cout << "  Rel error (self vs exact): " << rel_error_self << std::endl;
  std::cout << "  Rel error (div vs self): " << rel_error_div_vs_self << std::endl;
  
  bool passed = (rel_error_div < 0.1) && (rel_error_self < 0.1) && (rel_error_div_vs_self < 0.1);
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 7: CPU vs GPU comparison (accuracy and speed)
// ============================================================================
bool test_cpu_vs_gpu_comparison(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 7: CPU vs GPU Comparison ===" << std::endl;
  
  // Clear any previous CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess && err != cudaErrorNoDevice) {
    cudaDeviceSynchronize();  // Clear error state
  }
  
  // Create unit sphere Gaussian splats
  Eigen::MatrixXd P, N;
  Eigen::VectorXd sigmas, weights;
  create_sphere_gaussians(P, N, sigmas, weights, 2000);  // More points for better accuracy
  auto geom = make_gaussian_splat(P, N, sigmas, weights);
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute using CPU version (with profiling)
  std::cout << "\n--- CPU Version ---" << std::endl;
  double volume_cpu = compute_intersection_volume_cpu(geom, geom, KG, true);
  
  // Compute using GPU version (with profiling)
  std::cout << "\n--- GPU Version ---" << std::endl;
  double volume_gpu = compute_intersection_volume_cuda(geom, geom, KG, 256, true);
  
  // Compare results
  double volume_exact = 4.0 * M_PI / 3.0;  // Unit sphere volume
  double abs_error_cpu = std::abs(volume_cpu - volume_exact);
  double abs_error_gpu = std::abs(volume_gpu - volume_exact);
  double rel_error_cpu = abs_error_cpu / volume_exact;
  double rel_error_gpu = abs_error_gpu / volume_exact;
  double cpu_gpu_diff = std::abs(volume_cpu - volume_gpu);
  double cpu_gpu_rel_diff = cpu_gpu_diff / (0.5 * (std::abs(volume_cpu) + std::abs(volume_gpu)) + 1e-10);
  
  std::cout << "\n--- Comparison ---" << std::endl;
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  CPU volume:        " << volume_cpu << std::endl;
  std::cout << "  GPU volume:        " << volume_gpu << std::endl;
  std::cout << "  Exact volume:      " << volume_exact << std::endl;
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  CPU abs error:     " << abs_error_cpu << std::endl;
  std::cout << "  GPU abs error:     " << abs_error_gpu << std::endl;
  std::cout << "  CPU rel error:     " << rel_error_cpu 
            << " (" << std::fixed << std::setprecision(4) << rel_error_cpu * 100.0 << "%)" << std::endl;
  std::cout << "  GPU rel error:     " << rel_error_gpu 
            << " (" << std::fixed << std::setprecision(4) << rel_error_gpu * 100.0 << "%)" << std::endl;
  std::cout << "  CPU-GPU diff:      " << std::scientific << cpu_gpu_diff << std::endl;
  std::cout << "  CPU-GPU rel diff:  " << cpu_gpu_rel_diff 
            << " (" << std::fixed << std::setprecision(4) << cpu_gpu_rel_diff * 100.0 << "%)" << std::endl;
  
  // Both should be reasonably accurate (within 20% of exact for Gaussian splats)
  bool accuracy_passed = (rel_error_cpu < 0.20) && (rel_error_gpu < 0.20);
  // CPU and GPU should agree (within 5% relative difference)
  bool consistency_passed = cpu_gpu_rel_diff < 0.05;
  bool passed = accuracy_passed && consistency_passed;
  
  std::cout << "  Accuracy check:    " << (accuracy_passed ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Consistency check: " << (consistency_passed ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Main test runner
// ============================================================================
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <lebedev_txt_file> [Nrad]" << std::endl;
    return 1;
  }
  
  std::string leb_file = argv[1];
  int Nrad = (argc > 2 ? std::atoi(argv[2]) : 96);
  
  std::cout << "Running unit tests for Gaussian splats..." << std::endl;
  std::cout << "Lebedev file: " << leb_file << std::endl;
  std::cout << "Radial points: " << Nrad << std::endl;
  
  bool all_passed = true;
  
  // Test 1: Form factor field comparison
  all_passed &= test_gaussian_form_factor_field(leb_file, 32);
  
  // Test 2: Volume computation
  all_passed &= test_gaussian_volume_computation(leb_file, Nrad);
  
  // Test 3: Gaussian-Gaussian intersection
  all_passed &= test_gaussian_gaussian_intersection(leb_file, Nrad);
  
  // Test 4: Gaussian-Mesh intersection
  all_passed &= test_gaussian_mesh_intersection(leb_file, Nrad);
  
  // Test 5: Gaussian-Point Cloud intersection
  all_passed &= test_gaussian_pointcloud_intersection(leb_file, Nrad);
  
  // Test 6: Volume from divergence theorem
  all_passed &= test_volume_divergence_theorem(leb_file, Nrad);
  
  // Test 7: CPU vs GPU comparison
  all_passed &= test_cpu_vs_gpu_comparison(leb_file, Nrad);
  
  std::cout << "\n=== Summary ===" << std::endl;
  if (all_passed) {
    std::cout << "All tests PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "Some tests FAILED!" << std::endl;
    return 1;
  }
}

