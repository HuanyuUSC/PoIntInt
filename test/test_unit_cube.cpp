#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <array>
#include <cassert>
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
// Test 1: Compare computed form factor field A(k) to exact formula
// ============================================================================
bool test_form_factor_field(const std::string& leb_file, int Nrad = 32) {
  std::cout << "\n=== Test 1: Form Factor Field Comparison ===" << std::endl;

  // Create unit cube mesh
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  create_unit_cube_mesh(V, F);
  auto geom = make_triangle_mesh(V, F);

  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);

  // Sample a few k-points and compare
  std::vector<std::pair<std::array<double, 3>, double>> test_k_points = {
    {{1.0, 0.0, 0.0}, 0.0},
    {{1.0, 0.0, 0.0}, 0.1},
    {{1.0, 0.0, 0.0}, 1.0},
    {{0.0, 1.0, 0.0}, 1.0},
    {{0.0, 0.0, 1.0}, 1.0},
    {{0.0, 1.0, 0.0}, 2.0},
  };
  
  double max_rel_error = 0.0;
  int num_tested = 0;
  
  for (const auto& k_test : test_k_points) {
    double kx = k_test.first[0] * k_test.second;
    double ky = k_test.first[1] * k_test.second;
    double kz = k_test.first[2] * k_test.second;
    
    // Compute A(k) from mesh using form_factor_helpers
    Eigen::Vector3d k_vec(kx, ky, kz);
    std::complex<double> A_complex = compute_A_geometry(geom, k_vec);
    double Ak2_mesh = std::norm(A_complex);  // |A|^2 = real^2 + imag^2
    
    // Exact |A(k)|²
    double Ak2_exact = exact_cube_Ak_squared(kx, ky, kz);  // From form_factor_helpers
    
    double rel_error = std::abs(Ak2_mesh - Ak2_exact) / (std::abs(Ak2_exact) + 1e-10);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  k = (" << std::setw(6) << kx << ", " 
              << std::setw(6) << ky << ", " << std::setw(6) << kz << "):" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "    |A|²_mesh  = " << std::setw(15) << Ak2_mesh << std::endl;
    std::cout << "    |A|²_exact = " << std::setw(15) << Ak2_exact << std::endl;
    std::cout << "    rel error  = " << std::setw(15) << rel_error 
              << " (" << std::fixed << std::setprecision(2) << rel_error * 100.0 << "%)" << std::endl;
    
    if (rel_error > max_rel_error) {
      max_rel_error = rel_error;
    }
    num_tested++;
  }
  
  std::cout << "  Tested " << num_tested << " k-points" << std::endl;
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Max relative error: " << max_rel_error << std::endl;
  
  bool passed = max_rel_error < 0.1;  // 10% tolerance for mesh computation
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 2: Integrate |F(k)|² over k-space and compare to volume
// ============================================================================
bool test_volume_from_form_factor_integral(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 2: Volume from |F(k)|² Integral ===" << std::endl;
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // According to theory (form_factor_intersection_theory.md, section 1):
  // V_∩(t=0) = (1/(2π)³) ∫ |F(k)|² d³k for self-intersection
  // 
  // The weights KG.w are set up for the A(k) formulation used in the CUDA code:
  //   KG.w[q] = w_angular * w_radial * sec²(t)
  // where k = tan(t), and this accounts for dk = sec²(t) dt
  //
  // For the F(k) integral, we need: ∫ |F(k)|² d³k
  // In spherical coordinates: d³k = k² dk dΩ
  // With k = tan(t), dk = sec²(t) dt, so: d³k = k² sec²(t) dt dΩ
  //
  // So the weight should be: w_angular * w_radial * k² * sec²(t)
  // But KG.w = w_angular * w_radial * sec²(t) (missing k² factor)
  // Therefore we multiply by k² to get the correct d³k measure
  
  double integral = 0.0;
  
  for (int q = 0; q < KG.kmag.size(); ++q) {
    double k = KG.kmag[q];
    const auto& dir = KG.dirs[q];
    double kx = k * dir[0];
    double ky = k * dir[1];
    double kz = k * dir[2];
    
    // |F(k)|² using exact formula for unit cube
    double F = exact_cube_form_factor(kx, ky, kz);  // From form_factor_helpers
    double F2 = F * F;
    
    // Weight includes w_angular * w_radial * sec²(t)
    // We need k² factor for d³k = k² sec²(t) dt dΩ
    integral += KG.w[q] * F2 * k * k;
  }
  
  // Use the formula from theory: V = (1/(2π)³) ∫ |F(k)|² d³k
  double volume_computed = integral / (8.0 * M_PI * M_PI * M_PI);
  double volume_exact = 1.0;  // Unit cube volume
  
  double rel_error = std::abs(volume_computed - volume_exact) / volume_exact;
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Computed volume: " << volume_computed << std::endl;
  std::cout << "  Exact volume:    " << volume_exact << std::endl;
  std::cout << "  Relative error:   " << std::scientific << rel_error 
            << " (" << std::fixed << std::setprecision(4) << rel_error * 100.0 << "%)" << std::endl;
  
  bool passed = rel_error < 0.01;  // 1% tolerance
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 3: Compare mesh-based volume computation to exact volume
// ============================================================================
bool test_mesh_volume_computation(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 3: Mesh Volume Computation ===" << std::endl;
  
  // Create unit cube mesh
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  create_unit_cube_mesh(V, F);
  auto geom = make_triangle_mesh(V, F);
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute volume using CUDA
  double volume_computed = compute_intersection_volume_cuda(geom, geom, KG, 256, true);
  double volume_exact = 1.0;  // Unit cube volume
  
  double rel_error = std::abs(volume_computed - volume_exact) / volume_exact;
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Computed volume: " << volume_computed << std::endl;
  std::cout << "  Exact volume:    " << volume_exact << std::endl;
  std::cout << "  Relative error:   " << std::scientific << rel_error 
            << " (" << std::fixed << std::setprecision(4) << rel_error * 100.0 << "%)" << std::endl;
  
  bool passed = rel_error < 0.05;  // 5% tolerance for mesh computation
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 4: Volume from divergence theorem
// ============================================================================
bool test_volume_divergence_theorem(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 4: Volume from Divergence Theorem ===" << std::endl;
  
  // Create unit cube mesh
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  create_unit_cube_mesh(V, F);
  auto geom = make_triangle_mesh(V, F);
  
  // Compute volume using divergence theorem (unified interface with identity AffineDoF)
  double volume_div = compute_volume_cpu(geom);
  
  // Compute F(0) - form factor at k=0 should equal volume
  // F(0) = ∫_Ω dx = V
  // For a triangle mesh, we can compute this directly
  Eigen::Vector3d k_zero(0.0, 0.0, 0.0);
  std::complex<double> A_zero = compute_A_geometry(geom, k_zero);
  // At k=0, A(0) = ∫_∂Ω n dS = 0 for closed surfaces
  // But F(0) = volume, so we need to compute it differently
  // Actually, for k=0, the form factor is just the volume
  // We can compute it by summing signed volumes of tetrahedra
  double volume_F0 = 0.0;
  for (const auto& tri : geom.tris) {
    Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
    Eigen::Vector3d e1(tri.e1.x, tri.e1.y, tri.e1.z);
    Eigen::Vector3d e2(tri.e2.x, tri.e2.y, tri.e2.z);
    Eigen::Vector3d b = a + e1;
    Eigen::Vector3d c = a + e2;
    // Signed volume of tetrahedron (0, a, b, c) = (1/6) * a · (b × c)
    double vol_tet = (1.0/6.0) * a.dot(e1.cross(e2));
    volume_F0 += vol_tet;
  }
  
  // Compute self-intersection volume
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  double volume_self = compute_intersection_volume_cuda(geom, geom, KG, 256, false);
  
  double volume_exact = 1.0;  // Unit cube volume
  
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  Volume (divergence theorem): " << volume_div << std::endl;
  std::cout << "  Volume (F(0) from tetrahedra): " << volume_F0 << std::endl;
  std::cout << "  Volume (self-intersection): " << volume_self << std::endl;
  std::cout << "  Exact volume: " << volume_exact << std::endl;
  
  double rel_error_div = std::abs(volume_div - volume_exact) / volume_exact;
  double rel_error_F0 = std::abs(volume_F0 - volume_exact) / volume_exact;
  double rel_error_self = std::abs(volume_self - volume_exact) / volume_exact;
  double rel_error_div_vs_self = std::abs(volume_div - volume_self) / (0.5 * (volume_div + volume_self) + 1e-10);
  
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Rel error (div vs exact): " << rel_error_div << std::endl;
  std::cout << "  Rel error (F0 vs exact): " << rel_error_F0 << std::endl;
  std::cout << "  Rel error (self vs exact): " << rel_error_self << std::endl;
  std::cout << "  Rel error (div vs self): " << rel_error_div_vs_self << std::endl;
  
  bool passed = (rel_error_div < 0.01) && (rel_error_F0 < 0.01) && 
                (rel_error_self < 0.05) && (rel_error_div_vs_self < 0.05);
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

// ============================================================================
// Test 5: CPU vs GPU comparison (accuracy and speed)
// ============================================================================
bool test_cpu_vs_gpu_comparison(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test 5: CPU vs GPU Comparison ===" << std::endl;
  
  // Create unit cube mesh
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  create_unit_cube_mesh(V, F);
  auto geom = make_triangle_mesh(V, F);
  
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
  double volume_exact = 1.0;  // Unit cube volume
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
  
  // Both should be accurate (within 5% of exact)
  bool accuracy_passed = (rel_error_cpu < 0.05) && (rel_error_gpu < 0.05);
  // CPU and GPU should agree (within 1% relative difference)
  bool consistency_passed = cpu_gpu_rel_diff < 0.01;
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
  
  std::cout << "Running unit tests for unit cube..." << std::endl;
  std::cout << "Lebedev file: " << leb_file << std::endl;
  std::cout << "Radial points: " << Nrad << std::endl;
  
  bool all_passed = true;
  
  // Test 1: Form factor field (simplified - verifies exact formula)
  all_passed &= test_form_factor_field(leb_file, 32);
  
  // Test 2: Volume from exact form factor integral
  all_passed &= test_volume_from_form_factor_integral(leb_file, Nrad);
  
  // Test 3: Mesh-based volume computation
  all_passed &= test_mesh_volume_computation(leb_file, Nrad);
  
  // Test 4: Volume from divergence theorem
  all_passed &= test_volume_divergence_theorem(leb_file, Nrad);
  
  // Test 5: CPU vs GPU comparison
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

