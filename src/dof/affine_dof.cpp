#include "dof/affine_dof.hpp"
#include "form_factor_helpers.hpp"
#include "geometry/packing.hpp"
#include <cuda_runtime.h>
#include <cassert>

namespace PoIntInt {

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
    std::complex<double> A_plus = compute_A_geometry(geom_plus, k);
    
    // Backward difference
    Eigen::VectorXd dofs_minus = dofs;
    dofs_minus(i) -= eps;
    Geometry geom_minus = apply(geom, dofs_minus);
    std::complex<double> A_minus = compute_A_geometry(geom_minus, k);
    
    // Central difference: (f(x+h) - f(x-h)) / (2h)
    grad(i) = (A_plus - A_minus) / (2.0 * eps);
  }
  }
  
  return grad;
}

Eigen::VectorXd AffineDoF::compute_volume_gradient(
  const Geometry& geom,
  const Eigen::VectorXd& dofs) const
{
  auto [A, t] = build_affine_transform(dofs);
  double det_A = A.determinant();
  double sign_det_A = (det_A >= 0.0) ? 1.0 : -1.0;
  
  // Compute base volume (volume of original geometry)
  // For triangle meshes: V = (1/3) * Σ_triangles (a · S)
  double V_base = 0.0;
  if (geom.type == GEOM_TRIANGLE) {
    for (const auto& tri : geom.tris) {
      Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
      Eigen::Vector3d S(tri.S.x, tri.S.y, tri.S.z);
      V_base += a.dot(S);
    }
    V_base /= 3.0;
  } else if (geom.type == GEOM_DISK) {
    for (const auto& disk : geom.disks) {
      Eigen::Vector3d c(disk.c.x, disk.c.y, disk.c.z);
      Eigen::Vector3d n(disk.n.x, disk.n.y, disk.n.z);
      V_base += c.dot(n) * disk.area;
    }
    V_base /= 3.0;
  } else if (geom.type == GEOM_GAUSSIAN) {
    for (const auto& gauss : geom.gaussians) {
      Eigen::Vector3d c(gauss.c.x, gauss.c.y, gauss.c.z);
      Eigen::Vector3d n(gauss.n.x, gauss.n.y, gauss.n.z);
      V_base += c.dot(n) * gauss.w;
    }
    V_base /= 3.0;
  }
  
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(12);
  
  // Gradient w.r.t. translation t (DoFs 0-2) = 0  
  // Gradient w.r.t. matrix A (DoFs 3-11)
  // V' = |det(A)| * V_base
  // dV'/dA_ij = sign(det(A)) * cof(A) * V_base
  
  // Compute cofactor matrix
  Eigen::Matrix3d cofactor_A;
  cofactor_A(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
  cofactor_A(0, 1) = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
  cofactor_A(0, 2) = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);
  cofactor_A(1, 0) = A(2, 1) * A(0, 2) - A(2, 2) * A(0, 1);
  cofactor_A(1, 1) = A(2, 2) * A(0, 0) - A(2, 0) * A(0, 2);
  cofactor_A(1, 2) = A(2, 0) * A(0, 1) - A(2, 1) * A(0, 0);
  cofactor_A(2, 0) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
  cofactor_A(2, 1) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
  cofactor_A(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      int dof_idx = 3 + row * 3 + col;
      grad(dof_idx) = sign_det_A * cofactor_A(row, col) * V_base;
    }
  }
  
  return grad;
}

} // namespace PoIntInt

