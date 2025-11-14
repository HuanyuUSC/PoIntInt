#include "dof/dof_helpers.hpp"
#include "geometry_packing.hpp"
#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

// ============================================================================
// Helper: Compute A(k) for a geometry on CPU (for gradient computation)
// ============================================================================

// Compute A_parallel(k) = (k·A(k))/|k| for triangles
std::complex<double> compute_A_triangle_cpu(
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
  
  // Compute Phi_ab(alpha, beta) = J_triangle
  auto sinc = [](double x) -> std::complex<double> {
    if (std::abs(x) < 1e-4) {
      return std::complex<double>(1.0 - x*x/6.0, -x/2.0);
    }
    return (std::exp(std::complex<double>(0.0, x)) - 1.0) / std::complex<double>(0.0, x);
  };
  
  std::complex<double> phi;
  if (std::abs(alpha - beta) < 1e-6 && std::abs(alpha) < 1e-6) {
    // Degenerate case
    phi = std::complex<double>(0.5, 0.0);
  } else {
    std::complex<double> sinc_alpha = sinc(alpha);
    std::complex<double> sinc_beta = sinc(beta);
    std::complex<double> sinc_diff = sinc(alpha - beta);
    phi = (sinc_beta * sinc_diff - sinc_alpha) / std::complex<double>(0.0, beta);
  }
  
  std::complex<double> phase = std::exp(std::complex<double>(0.0, k.dot(a)));
  return phase * phi * gamma;
}

// Compute A_parallel(k) for disks
std::complex<double> compute_A_disk_cpu(
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
std::complex<double> compute_A_gaussian_cpu(
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
std::complex<double> compute_A_geometry_cpu(
  const Geometry& geom,
  const Eigen::Vector3d& k)
{
  std::complex<double> A_total(0.0, 0.0);
  
  if (geom.type == GEOM_TRIANGLE) {
    for (const auto& tri : geom.tris) {
      A_total += compute_A_triangle_cpu(tri, k);
    }
  } else if (geom.type == GEOM_DISK) {
    for (const auto& disk : geom.disks) {
      A_total += compute_A_disk_cpu(disk, k);
    }
  } else if (geom.type == GEOM_GAUSSIAN) {
    for (const auto& gauss : geom.gaussians) {
      A_total += compute_A_gaussian_cpu(gauss, k);
    }
  }
  
  return A_total;
}

} // namespace PoIntInt

