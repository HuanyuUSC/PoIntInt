#include "form_factor_helpers.hpp"
#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

// ============================================================================
// Form Factor Field Computation Helpers
// ============================================================================

// Compute A_parallel(k) = (k·A(k))/|k| for triangles
std::complex<double> compute_A_triangle(
  const TriPacked& tri,
  const Eigen::Vector3d& k)
{
  double kx = k.x(), ky = k.y(), kz = k.z();
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  Eigen::Vector3d e1(tri.e1.x, tri.e1.y, tri.e1.z);
  Eigen::Vector3d e2(tri.e2.x, tri.e2.y, tri.e2.z);
  Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
  Eigen::Vector3d S(tri.S.x, tri.S.y, tri.S.z);
  
  double alpha = k.dot(e1);
  double beta = k.dot(e2);
  double gamma = k.dot(S) / kmag;  // (k·S)/|k|
  
  // Compute Phi_ab(alpha, beta) = 2i [E(β) - E(α)]/(β-α)
  // where E(z) = (e^(iz) - 1)/(iz) = sinc(z)
  auto E_func = [](double z) -> std::complex<double> {
    double az = std::abs(z);
    if (az < 1e-4) {
      double z2 = z * z, z4 = z2 * z2;
      double real = 1.0 - z2 / 6.0 + z4 / 120.0;
      double imag = z * 0.5 - z * z2 / 24.0 + z4 * z / 720.0;
      return std::complex<double>(real, imag);
    }
    double s = std::sin(z);
    double c = std::cos(z);
    return std::complex<double>(s / z, (1.0 - c) / z);
  };
  
  auto E_prime = [](double z) -> std::complex<double> {
    double az = std::abs(z);
    if (az < 1e-4) {
      double z2 = z * z, z3 = z2 * z, z4 = z2 * z2;
      double real = -z / 3.0 + z3 / 30.0;
      double imag = 0.5 - z2 / 8.0 + z4 / 120.0;
      return std::complex<double>(real, imag);
    }
    double s = std::sin(z);
    double c = std::cos(z);
    double z2 = z * z;
    double re = (z * c - s) / z2;
    double im = (z * s - (1.0 - c)) / z2;
    return std::complex<double>(re, im);
  };
  
  std::complex<double> phi;
  double d = beta - alpha;
  if (std::abs(d) < 1e-5) {
    // Use derivative when alpha ≈ beta: Phi = 2i * E'((alpha+beta)/2)
    std::complex<double> Ep = E_prime(0.5 * (alpha + beta));
    // Match CUDA: make_float2(2.0f*Ep.y, -2.0f*Ep.x)
    // where Ep.y = Ep.im, Ep.x = Ep.re
    phi = std::complex<double>(2.0 * Ep.imag(), -2.0 * Ep.real());
  } else {
    std::complex<double> Ea = E_func(alpha);
    std::complex<double> Eb = E_func(beta);
    std::complex<double> diff = Eb - Ea;
    // Match CUDA: make_float2(2.0f*q.y, -2.0f*q.x) where q = diff/d
    // q.y = diff.im/d, q.x = diff.re/d
    phi = std::complex<double>(2.0 * diff.imag() / d, -2.0 * diff.real() / d);
  }
  
  std::complex<double> phase = std::exp(std::complex<double>(0.0, k.dot(a)));
  return phase * phi * gamma;
}

// Compute A_parallel(k) for disks
std::complex<double> compute_A_disk(
  const DiskPacked& disk,
  const Eigen::Vector3d& k)
{
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  Eigen::Vector3d kvec(k.x(), k.y(), k.z());
  Eigen::Vector3d n(disk.n.x, disk.n.y, disk.n.z);
  Eigen::Vector3d c(disk.c.x, disk.c.y, disk.c.z);
  
  double kdir_dot_n = kvec.dot(n) / kmag;
  double kdotn = kmag * kdir_dot_n;
  double r2 = std::max(0.0, kmag * kmag - kdotn * kdotn);
  double r = std::sqrt(r2);
  
  // J1_over_x approximation
  auto J1_over_x = [](double x) -> double {
    double ax = std::abs(x);
    if (ax < 1e-3) {
      double x2 = x * x;
      return 0.5 - x2/16.0 + x2*x2/384.0;
    }
    if (ax <= 12.0) {
      double q = 0.25 * x * x;
      double term = 0.5;
      double sum = term;
      for (int m = 0; m < 20; ++m) {
        double denom = (double)(m + 1) * (double)(m + 2);
        term *= -q / denom;
        sum += term;
        if (std::abs(term) < 1e-7 * std::abs(sum)) break;
      }
      return sum;
    }
    // Large x: Hankel asymptotics
    double invx = 1.0 / ax;
    double invx2 = invx * invx;
    double invx3 = invx2 * invx;
    double chi = ax - 0.75 * M_PI;
    double amp = std::sqrt(2.0 / (M_PI * ax));
    double cosp = (1.0 - 15.0 / 128.0 * invx2) * std::cos(chi);
    double sinp = (3.0 / 8.0 * invx - 315.0 / 3072.0 * invx3) * std::sin(chi);
    double J1 = amp * (cosp - sinp);
    return J1 * invx;
  };
  
  double rho_r = disk.rho * r;
  double S_magnitude = (rho_r < 1e-10) ? disk.area : 2.0 * disk.area * J1_over_x(rho_r);
  double A_parallel_mag = kdir_dot_n * S_magnitude;
  
  std::complex<double> phase = std::exp(std::complex<double>(0.0, kvec.dot(c)));
  return phase * std::complex<double>(A_parallel_mag, 0.0);
}

// Compute A_parallel(k) for Gaussian splats
std::complex<double> compute_A_gaussian(
  const GaussianPacked& gauss,
  const Eigen::Vector3d& k)
{
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  Eigen::Vector3d kvec(k.x(), k.y(), k.z());
  Eigen::Vector3d n(gauss.n.x, gauss.n.y, gauss.n.z);
  Eigen::Vector3d c(gauss.c.x, gauss.c.y, gauss.c.z);
  
  double kdir_dot_n = kvec.dot(n) / kmag;
  double kdotn = kmag * kdir_dot_n;
  double k_perp_sq = std::max(0.0, kmag * kmag - kdotn * kdotn);
  
  double exp_arg = -0.5 * gauss.sigma * gauss.sigma * k_perp_sq;
  double S_magnitude = gauss.w * std::exp(exp_arg);
  double A_parallel_mag = kdir_dot_n * S_magnitude;
  
  std::complex<double> phase = std::exp(std::complex<double>(0.0, kvec.dot(c)));
  return phase * std::complex<double>(A_parallel_mag, 0.0);
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

