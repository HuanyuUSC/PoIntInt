#include "dof/affine_dof_decomposed.hpp"
#include "form_factor_helpers.hpp"
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
    std::complex<double> A_plus = compute_A_geometry(geom_plus, k);
    
    // Backward difference
    Eigen::VectorXd dofs_minus = dofs;
    dofs_minus(i) -= eps;
    Geometry geom_minus = apply(geom, dofs_minus);
    std::complex<double> A_minus = compute_A_geometry(geom_minus, k);
    
    // Central difference: (f(x+h) - f(x-h)) / (2h)
    grad(i) = (A_plus - A_minus) / (2.0 * eps);
  }
  
  return grad;
}

} // namespace PoIntInt

