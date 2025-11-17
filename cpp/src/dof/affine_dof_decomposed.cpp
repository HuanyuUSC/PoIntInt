#include "dof/affine_dof_decomposed.hpp"
#include "form_factor_helpers.hpp"
#include "element_functions.hpp"
#include "geometry/packing.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

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

std::complex<double> AffineDoF_Decomposed::compute_A(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  auto [A, t] = build_affine_transform(dofs);

  std::complex<double> Ak(0.0, 0.0);

  double kmag = k.norm();
  if (kmag < 1e-10) return Ak; // A(0) = 0

  const Eigen::Vector3d khat = k / kmag;
  
  // the imaginary unit
  static const std::complex<double> I(0.0, 1.0);

  if (geom.type == GEOM_TRIANGLE) {
    // For each triangle, compute A(k) contribution
    for (size_t i = 0; i < geom.tris.size(); ++i) {
      const auto& tri = geom.tris[i];
      Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
      Eigen::Vector3d e1(tri.e1.x, tri.e1.y, tri.e1.z);
      Eigen::Vector3d e2(tri.e2.x, tri.e2.y, tri.e2.z);

      // Transformed quantities
      Eigen::Vector3d a_prime = A * a + t;
      Eigen::Vector3d e1_prime = A * e1;
      Eigen::Vector3d e2_prime = A * e2;
      Eigen::Vector3d S_prime = 0.5 * e1_prime.cross(e2_prime);

      // Compute A(k) for transformed triangle
      double alpha_prime = k.dot(e1_prime);
      double beta_prime = k.dot(e2_prime);
      double gamma_prime = khat.dot(S_prime);

      std::complex<double> phi_prime = Triangle_Phi_ab(alpha_prime, beta_prime);
      std::complex<double> phase_prime = std::exp(I * k.dot(a_prime));
      Ak += phase_prime * phi_prime * gamma_prime;
    }
  }
  else if (geom.type == GEOM_DISK) {
    // For each disk, compute A(k) contribution
    for (size_t i = 0; i < geom.disks.size(); ++i) {
      const auto& disk = geom.disks[i];
      Eigen::Vector3d c(disk.c.x, disk.c.y, disk.c.z);
      Eigen::Vector3d n(disk.n.x, disk.n.y, disk.n.z);

      // Transformed quantities
      Eigen::Vector3d c_prime = A * c + t;
      Eigen::Vector3d k_prime = A.transpose() * k;

      double kdotn = k_prime.dot(n);
      double r2 = std::max(0.0, k_prime.squaredNorm() - kdotn * kdotn);
      double r = std::sqrt(r2);

      double rho_r = disk.rho * r;
      double S_magnitude = (rho_r < 1e-10) ? disk.area : 2.0 * disk.area * Disk_J1_over_x(rho_r);
      double A_parallel_mag = S_magnitude * kdotn / kmag;

      std::complex<double> phase = std::exp(k.dot(c_prime) * I);

      Ak += phase * A_parallel_mag;
    }
  }
  else if (geom.type == GEOM_GAUSSIAN) {
    // For each Gaussian surfel, compute A(k) contribution
    for (size_t i = 0; i < geom.gaussians.size(); ++i) {
      const auto& gauss = geom.gaussians[i];
      Eigen::Vector3d c(gauss.c.x, gauss.c.y, gauss.c.z);
      Eigen::Vector3d n(gauss.n.x, gauss.n.y, gauss.n.z);

      // Transformed quantities
      Eigen::Vector3d c_prime = A * c + t;
      Eigen::Vector3d k_prime = A.transpose() * k;

      double kdotn = k_prime.dot(n);
      double r2 = k_prime.squaredNorm() - kdotn * kdotn;

      double exp_arg = -0.5 * gauss.sigma * gauss.sigma * r2;
      double S_magnitude = gauss.w * std::exp(exp_arg);
      double A_parallel_mag = S_magnitude * kdotn / kmag;

      std::complex<double> phase = std::exp(k.dot(c_prime) * I);

      Ak += phase * A_parallel_mag;
    }
  }

  return Ak;
}

Eigen::VectorXcd AffineDoF_Decomposed::compute_A_gradient(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  // Compute gradient using finite differencing by calling compute_A directly
  // Use central differences for better accuracy
  Eigen::VectorXcd grad(12);
  double eps = 1e-6;
  
  for (int i = 0; i < 12; ++i) {
    // Forward difference
    Eigen::VectorXd dofs_plus = dofs;
    dofs_plus(i) += eps;
    std::complex<double> A_plus = compute_A(geom, k, dofs_plus);
    
    // Backward difference
    Eigen::VectorXd dofs_minus = dofs;
    dofs_minus(i) -= eps;
    std::complex<double> A_minus = compute_A(geom, k, dofs_minus);
    
    // Central difference: (f(x+h) - f(x-h)) / (2h)
    grad(i) = (A_plus - A_minus) / (2.0 * eps);
  }
  
  return grad;
}

} // namespace PoIntInt

