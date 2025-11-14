#include "form_factor_helpers.hpp"
#include "element_functions.hpp"
#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

// The imaginary unit
static const std::complex<double> I(0.0, 1.0);

// ============================================================================
// Form Factor Field Computation Helpers
// ============================================================================

// Compute A_parallel(k) = (k·A(k))/|k| for triangles
std::complex<double> compute_A_triangle(
  const TriPacked& tri,
  const Eigen::Vector3d& k)
{
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  Eigen::Vector3d e1(tri.e1.x, tri.e1.y, tri.e1.z);
  Eigen::Vector3d e2(tri.e2.x, tri.e2.y, tri.e2.z);
  Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
  Eigen::Vector3d S(tri.S.x, tri.S.y, tri.S.z);
  
  double alpha = k.dot(e1);
  double beta = k.dot(e2);
  double gamma = k.dot(S) / kmag;  // (k·S)/|k|
  
  std::complex<double> phi = Triangle_Phi_ab(alpha, beta);  
  std::complex<double> phase = std::exp(k.dot(a) * I);
  return phase * phi * gamma;
}

// Compute A_parallel(k) for disks
std::complex<double> compute_A_disk(
  const DiskPacked& disk,
  const Eigen::Vector3d& k)
{
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  Eigen::Vector3d n(disk.n.x, disk.n.y, disk.n.z);
  Eigen::Vector3d c(disk.c.x, disk.c.y, disk.c.z);
  
  double kdotn = k.dot(n);
  double r2 = std::max(0.0, kmag * kmag - kdotn * kdotn);
  double r = std::sqrt(r2);
  
  double rho_r = disk.rho * r;
  double S_magnitude = (rho_r < 1e-10) ? disk.area : 2.0 * disk.area * Disk_J1_over_x(rho_r);
  double A_parallel_mag = S_magnitude * kdotn / kmag;
  
  std::complex<double> phase = std::exp(k.dot(c) * I);
  return phase * A_parallel_mag;
}

// Compute A_parallel(k) for Gaussian splats
std::complex<double> compute_A_gaussian(
  const GaussianPacked& gauss,
  const Eigen::Vector3d& k)
{
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  Eigen::Vector3d n(gauss.n.x, gauss.n.y, gauss.n.z);
  Eigen::Vector3d c(gauss.c.x, gauss.c.y, gauss.c.z);
  
  double kdotn = k.dot(n);
  double k_perp_sq = std::max(0.0, kmag * kmag - kdotn * kdotn);
  
  double exp_arg = -0.5 * gauss.sigma * gauss.sigma * k_perp_sq;
  double S_magnitude = gauss.w * std::exp(exp_arg);
  double A_parallel_mag = S_magnitude * kdotn / kmag;
  
  std::complex<double> phase = std::exp(k.dot(c) * I);
  return phase * A_parallel_mag;
}

// Compute A_parallel(k) for entire geometry
std::complex<double> compute_A_geometry(
  const Geometry& geom,
  const Eigen::Vector3d& k)
{
  std::complex<double> A_total(0.0, 0.0);
  
  if (geom.type == GEOM_TRIANGLE) {
    for (const auto& tri : geom.tris) {
      A_total += compute_A_triangle(tri, k);
    }
  } else if (geom.type == GEOM_DISK) {
    for (const auto& disk : geom.disks) {
      A_total += compute_A_disk(disk, k);
    }
  } else if (geom.type == GEOM_GAUSSIAN) {
    for (const auto& gauss : geom.gaussians) {
      A_total += compute_A_gaussian(gauss, k);
    }
  }
  
  return A_total;
}

// ============================================================================
// Exact Form Factor Formulas for Simple Shapes
// ============================================================================

// Exact form factor for unit cube centered at origin: F(k) = ∏_{i=x,y,z} sin(k_i/2) / (k_i/2)
double exact_cube_form_factor(double kx, double ky, double kz) {
  auto sinc = [](double x) {
    if (std::abs(x) < 1e-4) 
      return 1.0 - x * x / 6.0 + x * x * x * x / 120.0;
    return std::sin(x) / x;
  };
  return sinc(0.5 * kx) * sinc(0.5 * ky) * sinc(0.5 * kz);
}

// Exact |A(k)|² for unit cube: |A(k)|² = |k|² |F(k)|²
double exact_cube_Ak_squared(double kx, double ky, double kz) {
  double k2 = kx*kx + ky*ky + kz*kz;
  double F = exact_cube_form_factor(kx, ky, kz);
  return k2 * F * F;
}

// Exact form factor for solid sphere of radius R: F_ball(k) = 4π (sin(kR) - kR cos(kR))/k³
double exact_sphere_form_factor(double k, double R) {
  if (std::abs(k) < 1e-4) {
    // Limit as k->0: F_ball(0) = 4πR³/3
    double R3 = R * R * R;
    return 4.0 * M_PI * R3 / 3.0 * (1.0 - k * k * R * R / 10.0 + k * k * k * k * R * R * R * R / 280.0);
  }
  double kR = k * R;
  double k3 = k * k * k;
  double sin_kR = std::sin(kR);
  double cos_kR = std::cos(kR);
  return 4.0 * M_PI * (sin_kR - kR * cos_kR) / k3;
}

// Exact |A(k)|² for sphere: |A(k)|² = |k|² |F(k)|²
double exact_sphere_Ak_squared(double k, double R) {
  double F = exact_sphere_form_factor(k, R);
  return k * k * F * F;
}

} // namespace PoIntInt

