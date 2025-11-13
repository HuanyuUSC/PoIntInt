#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <array>
#include <cassert>
#include "compute_volume.hpp"
#include "geometry_packing.hpp"
#include "lebedev_io.hpp"
#include "gauss_legendre.hpp"

using namespace PoIntInt;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact formulas for unit cube [-1/2, 1/2]^3
// ============================================================================

// Exact form factor: F(k) = ∏_{i=x,y,z} sin(k_i/2) / (k_i/2)
// For k_i = 0, use limit: sin(0)/0 = 1
inline double exact_cube_form_factor(double kx, double ky, double kz) {
  auto sinc = [](double x) {
    if (std::abs(x) < 1e-4) 
      return 1.0 - x * x / 6.0 + x * x * x * x / 120.0;
    return std::sin(x) / x;
  };
  return sinc(0.5 * kx) * sinc(0.5 * ky) * sinc(0.5 * kz);
}

// Exact A(k) = i k F(k), so |A(k)|² = |k|² |F(k)|²
inline double exact_cube_Ak_squared(double kx, double ky, double kz) {
  double k2 = kx*kx + ky*ky + kz*kz;
  double F = exact_cube_form_factor(kx, ky, kz);
  return k2 * F * F;
}

// ============================================================================
// Helper: Create unit cube mesh centered at origin [-1/2, 1/2]^3
// ============================================================================
void create_unit_cube_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
  V.resize(8, 3);
  // Vertices of cube [-1/2, 1/2]^3
  V <<
    -0.5, -0.5, -0.5,   // 0
     0.5, -0.5, -0.5,   // 1
     0.5,  0.5, -0.5,   // 2
    -0.5,  0.5, -0.5,   // 3
    -0.5, -0.5,  0.5,   // 4
     0.5, -0.5,  0.5,   // 5
     0.5,  0.5,  0.5,   // 6
    -0.5,  0.5,  0.5;   // 7
  
  F.resize(12, 3);
  // Faces (two triangles per face), outward orientation
  F <<
    0, 2, 1,  0, 3, 2,   // z = -0.5 (bottom)
    4, 5, 6,  4, 6, 7,   // z =  0.5 (top)
    0, 1, 5,  0, 5, 4,   // y = -0.5 (front)
    2, 7, 6,  2, 3, 7,   // y =  0.5 (back)
    0, 4, 7,  0, 7, 3,   // x = -0.5 (left)
    1, 6, 5,  1, 2, 6;   // x =  0.5 (right)
}

// ============================================================================
// CPU implementation of A(k) computation from mesh (for testing)
// ============================================================================

// Compute A(k) = ∫ e^{ik·x} n(x) dS for a mesh
std::pair<double, double> compute_scalar_Ak_from_mesh(const std::vector<TriPacked>& tris, const std::array<double, 3>& kdir, double kmag) {
  std::pair<double, double> A = {0.0, 0.0};
  double kx = kmag * kdir[0];
  double ky = kmag * kdir[1];
  double kz = kmag * kdir[2];
  
  for (const auto& t : tris) {
    // Convert float3 to double for computation
    double ax = t.a.x, ay = t.a.y, az = t.a.z;
    double e1x = t.e1.x, e1y = t.e1.y, e1z = t.e1.z;
    double e2x = t.e2.x, e2y = t.e2.y, e2z = t.e2.z;
    double Sx = t.S.x, Sy = t.S.y, Sz = t.S.z;
    
    // alpha = k · e1, beta = k · e2
    double alpha = kx * e1x + ky * e1y + kz * e1z;
    double beta = kx * e2x + ky * e2y + kz * e2z;
    double gamma = kdir[0] * Sx + kdir[1] * Sy + kdir[2] * Sz;
    
    // Phi(alpha, beta) = 2i [E(beta) - E(alpha)]/(beta - alpha)
    // E(z) = (sin z + i(1-cos z)) / z
    auto E_func = [](double z) -> std::pair<double, double> {
      if (std::abs(z) < 1e-4) {
        double z2 = z * z, z4 = z2 * z2;
        double real = 1.0 - z2 / 6.0 + z4 / 120.0;
        double imag = z * 0.5 - z * z2 / 24.0 + z4 * z / 720.0;
        return {real, imag};  // E(0) = 1
      }
      double s = std::sin(z);
      double c = std::cos(z);
      return {s / z, (1.0 - c) / z};
    };

    auto E_prime = [](double z) -> std::pair<double, double> {
      double z2 = z * z;
      if (std::abs(z) < 1e-4) {
        double z3 = z * z2, z4 = z2 * z2;
        double real = -z / 3.0 + z3 / 30.0;
        double imag = 0.5 - z2 / 8.0 + z4 / 120.0;
        return { real, imag };
      }
      double s = std::sin(z);
      double c = std::cos(z);
      return { (z * c - s) / z2, (z * s - (1.0f - c)) / z2 };
      };
    
    std::pair<double, double> phi;
    double d = beta - alpha;
    if (std::abs(d) < 1e-4) {
      // Use derivative: E'(z) = (z cos z - sin z + i(z sin z - (1-cos z))) / z²
      auto Ep = E_prime(0.5 * (alpha + beta));
      // 2i * (Ep.re + i Ep.im) = 2i*Ep.re - 2*Ep.im
      phi = { 2.0 * Ep.second, -2.0f * Ep.first };
    } else {
      auto Ea = E_func(alpha);
      auto Eb = E_func(beta);
      double num_re = Eb.first - Ea.first;
      double num_im = Eb.second - Ea.second;
      // 2i * (num_re + i*num_im) / d = (-2*num_im, 2*num_re) / d
      phi = {2.0 * num_im / d, -2.0 * num_re / d};
    }
    
    // Phase: k · a
    double phase = kx * ax + ky * ay + kz * az;
    double cos_phase = std::cos(phase);
    double sin_phase = std::sin(phase);
    
    // e^{ik·a} * phi
    double scalar_re = cos_phase * phi.first - sin_phase * phi.second;
    double scalar_im = sin_phase * phi.first + cos_phase * phi.second;
    
    // Accumulate: scalar * S
    A.first += scalar_re * gamma;
    A.second += scalar_im * gamma;
  }
  
  return A;
}

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
    
    // Compute A(k) from mesh
    std::pair<double, double> A_mesh = compute_scalar_Ak_from_mesh(geom.tris, k_test.first, k_test.second);
    double Ak2_mesh = A_mesh.first * A_mesh.first + A_mesh.second * A_mesh.second;
    
    // Exact |A(k)|²
    double Ak2_exact = exact_cube_Ak_squared(kx, ky, kz);
    
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
    double F = exact_cube_form_factor(kx, ky, kz);
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
  double volume_computed = compute_intersection_volume_cuda(geom, geom, KG, 256);
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
  
  std::cout << "\n=== Summary ===" << std::endl;
  if (all_passed) {
    std::cout << "All tests PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "Some tests FAILED!" << std::endl;
    return 1;
  }
}

