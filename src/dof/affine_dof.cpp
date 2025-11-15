#include "dof/affine_dof.hpp"
#include "element_functions.hpp"
#include "form_factor_helpers.hpp"
#include "geometry/packing.hpp"
#include <cuda_runtime.h>
#include <cassert>

namespace PoIntInt {

// the imaginary unit
static const std::complex<double> I(0.0, 1.0);

std::pair<Eigen::Matrix3d, Eigen::Vector3d> AffineDoF::build_affine_transform(
  const Eigen::VectorXd& dofs)
{
  assert(dofs.size() == 12);
  
  // Translation: dofs[0:3]
  Eigen::Vector3d t = dofs.segment<3>(0);
  
  // Matrix: dofs[3:12] (row-major)
  Eigen::Matrix3d A = Eigen::Map<const Eigen::Matrix3d>(dofs.data() + 3).transpose();
  
  return std::make_pair(A, t);
}

Eigen::Matrix3d AffineDoF::cofactor_matrix(const Eigen::Matrix3d& A)
{
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

  return cofactor_A;
}

Eigen::Matrix3d AffineDoF::cross_product_matrix(const Eigen::Vector3d& v) {
  Eigen::Matrix3d cp;
  cp << 0, -v.z(), v.y(),
    v.z(), 0, -v.x(),
    -v.y(), v.x(), 0;
  return cp;
}

Geometry AffineDoF::apply(const Geometry& geom, const Eigen::VectorXd& dofs) const {
  auto [A, t] = build_affine_transform(dofs);
  double det_A = A.determinant();
  
  Geometry transformed = geom;
  
  if (geom.type == GEOM_TRIANGLE) {
    // Transform triangles: vertices -> A*vertex + t
    for (auto& tri : transformed.tris) {
      Eigen::Vector3d a = A * Eigen::Vector3d(tri.a.x, tri.a.y, tri.a.z) + t;
      Eigen::Vector3d e1 = A * Eigen::Vector3d(tri.e1.x, tri.e1.y, tri.e1.z);
      Eigen::Vector3d e2 = A * Eigen::Vector3d(tri.e2.x, tri.e2.y, tri.e2.z);
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

std::complex<double> AffineDoF::compute_A(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  auto [A, t] = build_affine_transform(dofs);

  std::complex<double> Ak(0.0, 0.0);

  double kmag = k.norm();
  if (kmag < 1e-10) return Ak; // A(0) = 0

  const Eigen::Vector3d khat = k / kmag;

  if (geom.type == GEOM_TRIANGLE) {
    // For each triangle, compute A(k) contribution
    for (const auto& tri : geom.tris) {
      Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
      Eigen::Vector3d e1(tri.e1.x, tri.e1.y, tri.e1.z);
      Eigen::Vector3d e2(tri.e2.x, tri.e2.y, tri.e2.z);
      Eigen::Vector3d S(tri.S.x, tri.S.y, tri.S.z);

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
    for (const auto& disk : geom.disks) {
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
    for (const auto& gauss : geom.gaussians) {
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

Eigen::VectorXcd AffineDoF::compute_A_gradient(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  auto [A, t] = build_affine_transform(dofs);

  Eigen::VectorXcd grad = Eigen::VectorXcd::Zero(12);

  double kmag = k.norm();
  if (kmag < 1e-10) return grad; // gradient = 0 at k=0

  const Eigen::Vector3d khat = k / kmag;
  
  if (geom.type == GEOM_TRIANGLE) {
    // For each triangle, compute gradient contribution
    for (const auto& tri : geom.tris) {
      Eigen::Vector3d a(tri.a.x, tri.a.y, tri.a.z);
      Eigen::Vector3d e1(tri.e1.x, tri.e1.y, tri.e1.z);
      Eigen::Vector3d e2(tri.e2.x, tri.e2.y, tri.e2.z);
      Eigen::Vector3d S(tri.S.x, tri.S.y, tri.S.z);
      
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
      std::complex<double> A_tri = phase_prime * phi_prime * gamma_prime;
      
      // Compute gradients of Phi w.r.t. alpha and beta
      auto [dPhi_dalpha, dPhi_dbeta] = Triangle_Phi_ab_gradient(alpha_prime, beta_prime);
      
      // Gradient w.r.t. translation t (DoFs 0-2)
      // dA/dt = i*k * A (since phase = exp(i*k·(A*a+t)), and t only affects phase)
      grad.head(3) += I * A_tri * k;

      Eigen::Matrix3cd gA = A_tri * I * k * a.transpose();
      gA += gamma_prime * phase_prime * k * (dPhi_dalpha * e1 + dPhi_dbeta * e2).transpose();      
      gA -= cross_product_matrix(khat) * A * cross_product_matrix(S) * phase_prime * phi_prime;
      Eigen::Map<Eigen::Matrix3cd>(grad.data() + 3) += gA.transpose();
    }
  } else if (geom.type == GEOM_DISK) {
    // For each disk, compute gradient contribution
    for (const auto& disk : geom.disks) {
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
      std::complex<double> A_disk = phase * A_parallel_mag;

      // Gradient w.r.t. translation t (DoFs 0-2)
      // dA/dt = i*k * A (since phase = exp(i*k·(A*c+t)), and t only affects phase)
      grad.head(3) += I * A_disk * k;

      Eigen::Matrix3cd gA = A_disk * I * k * c.transpose();
      gA += khat * n.transpose() * phase * S_magnitude;
      double dS_magnitude = (rho_r < 1e-10) ? -0.25 * disk.rho : 2.0 * Disk_J1_over_x_prime(rho_r) / r;
      dS_magnitude *= disk.rho * disk.area;
      gA += phase * kdotn / kmag * dS_magnitude * k * (k_prime - kdotn * n).transpose();
      Eigen::Map<Eigen::Matrix3cd>(grad.data() + 3) += gA.transpose();
    }
  } else if (geom.type == GEOM_GAUSSIAN) {
    // For each Gaussian surfel, compute A(k) contribution
    for (const auto& gauss : geom.gaussians) {
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
      std::complex<double> A_gauss = phase * A_parallel_mag;

      // Gradient w.r.t. translation t (DoFs 0-2)
      // dA/dt = i*k * A (since phase = exp(i*k·(A*c+t)), and t only affects phase)
      grad.head(3) += I * A_gauss * k;

      Eigen::Matrix3cd gA = A_gauss * I * k * c.transpose();
      gA += khat * n.transpose() * phase * S_magnitude;
      double dS_magnitude = -gauss.sigma * gauss.sigma * S_magnitude;
      gA += phase * kdotn / kmag * dS_magnitude * k * (k_prime - kdotn * n).transpose();
      Eigen::Map<Eigen::Matrix3cd>(grad.data() + 3) += gA.transpose();
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
  Eigen::Matrix3d cof_A = cofactor_matrix(A);
  
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      int dof_idx = 3 + row * 3 + col;
      grad(dof_idx) = sign_det_A * cof_A(row, col) * V_base;
    }
  }
  
  return grad;
}

} // namespace PoIntInt

