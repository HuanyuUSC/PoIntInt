#include "dof_parameterization.hpp"
#include "geometry_packing.hpp"
#include <cmath>
#include <complex>
#include <algorithm>
#include <cuda_runtime.h>

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

// ============================================================================
// AffineDoF Implementation (Simple: translation + matrix)
// ============================================================================

std::pair<Eigen::Matrix3d, Eigen::Vector3d> AffineDoF::build_affine_transform(
  const Eigen::VectorXd& dofs) const
{
  assert(dofs.size() == 12);
  
  // Translation: dofs[0:3]
  Eigen::Vector3d t = dofs.segment<3>(0);
  
  // Matrix: dofs[3:12] (row-major)
  Eigen::Matrix3d A = Eigen::Map<const Eigen::Matrix3d>(dofs.data() + 3).transpose();
  
  return std::make_pair(A, t);
}

Geometry AffineDoF::apply(const Geometry& geom, const Eigen::VectorXd& dofs) const {
  auto [A, t] = build_affine_transform(dofs);
  double det_A = A.determinant();
  
  Geometry transformed = geom;
  
  if (geom.type == GEOM_TRIANGLE) {
    // Transform triangles: vertices -> A*vertex + t
    for (auto& tri : transformed.tris) {
      Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
      Eigen::Vector3d b = a + Eigen::Vector3d(tri.e1.x, tri.e1.y, tri.e1.z);
      Eigen::Vector3d c = a + Eigen::Vector3d(tri.e2.x, tri.e2.y, tri.e2.z);
      
      a = A * a + t;
      b = A * b + t;
      c = A * c + t;
      
      Eigen::Vector3d e1 = b - a;
      Eigen::Vector3d e2 = c - a;
      Eigen::Vector3d S = 0.5 * e1.cross(e2);
      
      tri.a = make_float3((float)a.x(), (float)a.y(), (float)a.z());
      tri.e1 = make_float3((float)e1.x(), (float)e1.y(), (float)e1.z());
      tri.e2 = make_float3((float)e2.x(), (float)e2.y(), (float)e2.z());
      tri.S = make_float3((float)S.x(), (float)S.y(), (float)S.z());
    }
  } else if (geom.type == GEOM_DISK) {
    // Transform disks: center -> A*center + t, normal -> A*normal (normalized)
    for (auto& disk : transformed.disks) {
      Eigen::Vector3d c(disk.c.x, disk.c.y, disk.c.z);
      Eigen::Vector3d n(disk.n.x, disk.n.y, disk.n.z);
      
      c = A * c + t;
      n = A * n;
      n.normalize();
      
      // Scale radius by average scale factor (geometric mean of scale factors)
      double scale_factor = std::pow(std::abs(det_A), 1.0/3.0);
      disk.rho *= (float)scale_factor;
      disk.area *= (float)(std::abs(det_A) / scale_factor);  // Adjust area
      
      disk.c = make_float3((float)c.x(), (float)c.y(), (float)c.z());
      disk.n = make_float3((float)n.x(), (float)n.y(), (float)n.z());
    }
  } else if (geom.type == GEOM_GAUSSIAN) {
    // Transform Gaussian splats: center -> A*center + t, normal -> A*normal (normalized)
    for (auto& gauss : transformed.gaussians) {
      Eigen::Vector3d c(gauss.c.x, gauss.c.y, gauss.c.z);
      Eigen::Vector3d n(gauss.n.x, gauss.n.y, gauss.n.z);
      
      c = A * c + t;
      n = A * n;
      n.normalize();
      
      // Scale sigma by average scale factor
      double scale_factor = std::pow(std::abs(det_A), 1.0/3.0);
      gauss.sigma *= (float)scale_factor;
      gauss.w *= (float)(std::abs(det_A) / scale_factor);  // Adjust weight
      
      gauss.c = make_float3((float)c.x(), (float)c.y(), (float)c.z());
      gauss.n = make_float3((float)n.x(), (float)n.y(), (float)n.z());
    }
  }
  
  return transformed;
}

Eigen::VectorXcd AffineDoF::compute_A_gradient(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  // Compute gradient using finite differencing by directly transforming geometry
  // Use central differences for better accuracy
  Eigen::VectorXcd grad(12);
  double eps = 1e-6;
  
  for (int i = 0; i < 12; ++i) {
    // Forward difference
    Eigen::VectorXd dofs_plus = dofs;
    dofs_plus(i) += eps;
    Geometry geom_plus = apply(geom, dofs_plus);
    std::complex<double> A_plus = compute_A_geometry_cpu(geom_plus, k);
    
    // Backward difference
    Eigen::VectorXd dofs_minus = dofs;
    dofs_minus(i) -= eps;
    Geometry geom_minus = apply(geom, dofs_minus);
    std::complex<double> A_minus = compute_A_geometry_cpu(geom_minus, k);
    
    // Central difference: (f(x+h) - f(x-h)) / (2h)
    grad(i) = (A_plus - A_minus) / (2.0 * eps);
  }
  
  return grad;
}

// ============================================================================
// AffineDoF_Decomposed Implementation (Legacy: rotation + scale + shear)
// ============================================================================

Eigen::Matrix3d AffineDoF_Decomposed::rotation_matrix(const Eigen::Vector3d& axis_angle) const {
  double angle = axis_angle.norm();
  if (angle < 1e-10) {
    return Eigen::Matrix3d::Identity();
  }
  
  Eigen::Vector3d axis = axis_angle / angle;
  double c = std::cos(angle);
  double s = std::sin(angle);
  double t = 1.0 - c;
  
  Eigen::Matrix3d R;
  R(0, 0) = t * axis.x() * axis.x() + c;
  R(0, 1) = t * axis.x() * axis.y() - s * axis.z();
  R(0, 2) = t * axis.x() * axis.z() + s * axis.y();
  R(1, 0) = t * axis.x() * axis.y() + s * axis.z();
  R(1, 1) = t * axis.y() * axis.y() + c;
  R(1, 2) = t * axis.y() * axis.z() - s * axis.x();
  R(2, 0) = t * axis.x() * axis.z() - s * axis.y();
  R(2, 1) = t * axis.y() * axis.z() + s * axis.x();
  R(2, 2) = t * axis.z() * axis.z() + c;
  
  return R;
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> AffineDoF_Decomposed::build_affine_transform(
  const Eigen::VectorXd& dofs) const
{
  assert(dofs.size() == 12);
  
  Eigen::Vector3d t = dofs.segment<3>(0);  // Translation
  Eigen::Vector3d r = dofs.segment<3>(3);  // Rotation (axis-angle)
  Eigen::Vector3d s = dofs.segment<3>(6);  // Scale
  Eigen::Vector3d h = dofs.segment<3>(9);  // Shear [hxy, hxz, hyz]
  
  // If scale is zero or negative, use identity (scale = 1)
  if (s.x() <= 0.0) s.x() = 1.0;
  if (s.y() <= 0.0) s.y() = 1.0;
  if (s.z() <= 0.0) s.z() = 1.0;
  
  // Build transformation matrix: A = R * S * H
  Eigen::Matrix3d R = rotation_matrix(r);
  
  // Scale matrix
  Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
  S(0, 0) = s.x();
  S(1, 1) = s.y();
  S(2, 2) = s.z();
  
  // Shear matrix (upper triangular)
  Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
  H(0, 1) = h.x();  // hxy
  H(0, 2) = h.y();  // hxz
  H(1, 2) = h.z();  // hyz
  
  Eigen::Matrix3d A = R * S * H;
  
  return std::make_pair(A, t);
}

Geometry AffineDoF_Decomposed::apply(const Geometry& geom, const Eigen::VectorXd& dofs) const {
  auto [A, t] = build_affine_transform(dofs);
  double det_A = A.determinant();
  
  Geometry transformed = geom;
  
  if (geom.type == GEOM_TRIANGLE) {
    // Transform triangles: vertices -> A*vertex + t
    for (auto& tri : transformed.tris) {
      Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
      Eigen::Vector3d b = a + Eigen::Vector3d(tri.e1.x, tri.e1.y, tri.e1.z);
      Eigen::Vector3d c = a + Eigen::Vector3d(tri.e2.x, tri.e2.y, tri.e2.z);
      
      a = A * a + t;
      b = A * b + t;
      c = A * c + t;
      
      Eigen::Vector3d e1 = b - a;
      Eigen::Vector3d e2 = c - a;
      Eigen::Vector3d S = 0.5 * e1.cross(e2);
      
      tri.a = make_float3((float)a.x(), (float)a.y(), (float)a.z());
      tri.e1 = make_float3((float)e1.x(), (float)e1.y(), (float)e1.z());
      tri.e2 = make_float3((float)e2.x(), (float)e2.y(), (float)e2.z());
      tri.S = make_float3((float)S.x(), (float)S.y(), (float)S.z());
    }
  } else if (geom.type == GEOM_DISK) {
    // Transform disks: center -> A*center + t, normal -> A*normal (normalized)
    for (auto& disk : transformed.disks) {
      Eigen::Vector3d c(disk.c.x, disk.c.y, disk.c.z);
      Eigen::Vector3d n(disk.n.x, disk.n.y, disk.n.z);
      
      c = A * c + t;
      n = A * n;
      n.normalize();
      
      // Scale radius by average scale factor (geometric mean of scale factors)
      double scale_factor = std::pow(std::abs(det_A), 1.0/3.0);
      disk.rho *= (float)scale_factor;
      disk.area *= (float)(std::abs(det_A) / scale_factor);  // Adjust area
      
      disk.c = make_float3((float)c.x(), (float)c.y(), (float)c.z());
      disk.n = make_float3((float)n.x(), (float)n.y(), (float)n.z());
    }
  } else if (geom.type == GEOM_GAUSSIAN) {
    // Transform Gaussian splats: center -> A*center + t, normal -> A*normal (normalized)
    for (auto& gauss : transformed.gaussians) {
      Eigen::Vector3d c(gauss.c.x, gauss.c.y, gauss.c.z);
      Eigen::Vector3d n(gauss.n.x, gauss.n.y, gauss.n.z);
      
      c = A * c + t;
      n = A * n;
      n.normalize();
      
      // Scale sigma by average scale factor
      double scale_factor = std::pow(std::abs(det_A), 1.0/3.0);
      gauss.sigma *= (float)scale_factor;
      gauss.w *= (float)(std::abs(det_A) / scale_factor);  // Adjust weight
      
      gauss.c = make_float3((float)c.x(), (float)c.y(), (float)c.z());
      gauss.n = make_float3((float)n.x(), (float)n.y(), (float)n.z());
    }
  }
  
  return transformed;
}

Eigen::VectorXcd AffineDoF_Decomposed::compute_A_gradient(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  // Compute gradient using finite differencing by directly transforming geometry
  // Use central differences for better accuracy
  Eigen::VectorXcd grad(12);
  double eps = 1e-6;
  
  for (int i = 0; i < 12; ++i) {
    // Forward difference
    Eigen::VectorXd dofs_plus = dofs;
    dofs_plus(i) += eps;
    Geometry geom_plus = apply(geom, dofs_plus);
    std::complex<double> A_plus = compute_A_geometry_cpu(geom_plus, k);
    
    // Backward difference
    Eigen::VectorXd dofs_minus = dofs;
    dofs_minus(i) -= eps;
    Geometry geom_minus = apply(geom, dofs_minus);
    std::complex<double> A_minus = compute_A_geometry_cpu(geom_minus, k);
    
    // Central difference: (f(x+h) - f(x-h)) / (2h)
    grad(i) = (A_plus - A_minus) / (2.0 * eps);
  }
  
  return grad;
}

} // namespace PoIntInt

