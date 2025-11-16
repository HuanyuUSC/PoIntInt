#include "dof/triangle_mesh_dof.hpp"
#include "element_functions.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <functional>

namespace PoIntInt {

TriangleMeshDoF::TriangleMeshDoF(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
  : num_vertices_(V.rows()), num_dofs_(3 * V.rows()), F_(F)
{
  assert(V.cols() == 3 && F.cols() == 3);
  assert(F.maxCoeff() < V.rows());  // Face indices must be valid
}

std::complex<double> TriangleMeshDoF::compute_A(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  assert(geom.type == GEOM_TRIANGLE);       // geometry type is triangle mesh
  assert(geom.tris.size() == F_.rows());    // number of triangles matches
  assert(dofs.size() == num_dofs_);         // correct number of DoFs

  std::complex<double> A(0.0, 0.0);

  const double knorm = k.norm();
  if (knorm < 1e-10) return A; // A(0) = 0

  const Eigen::Vector3d khat = k / knorm;

  // the imaginary unit
  static const std::complex<double> I(0.0, 1.0);

  // Per-face loop
  A = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, F_.rows()),
    std::complex<double>(0.0, 0.0),
    [&](const tbb::blocked_range<int>& r, std::complex<double> local_sum) -> std::complex<double> {
      for (int f = r.begin(); f < r.end(); ++f) {
        const Eigen::Vector3i& vid = F_.row(f);
        Eigen::Vector3d v[3] = { dofs.segment<3>(3 * vid[0]),
          dofs.segment<3>(3 * vid[1]), dofs.segment<3>(3 * vid[2]) };

        // edges e1=v1-v0, e2=v2-v0
        Eigen::Vector3d e1 = v[1] - v[0];
        Eigen::Vector3d e2 = v[2] - v[0];

        // Area-vector σn = 0.5 * (v1-v0) x (v2-v0)
        Eigen::Vector3d sigma_n = 0.5 * e1.cross(e2);

        // κ = k̂ · σn (real)
        const double kappa = khat.dot(sigma_n);

        // T = exp(i k·v0)
        std::complex<double> T = std::exp(I * k.dot(v[0]));

        // ψ = Φ(k·e1, k·e2)
        std::complex<double> psi = Triangle_Phi_ab(k.dot(e1), k.dot(e2));

        local_sum += kappa * T * psi;
      }
      return local_sum;
    },
    [](const std::complex<double>& a, const std::complex<double>& b) -> std::complex<double> {
      return a + b;
    });

  return A;
}

Eigen::VectorXcd TriangleMeshDoF::compute_A_gradient(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const Eigen::VectorXd& dofs) const
{
  assert(geom.type == GEOM_TRIANGLE);       // geometry type is triangle mesh
  assert(geom.tris.size() == F_.rows());    // number of triangles matches
  assert(dofs.size() == num_dofs_);         // correct number of DoFs
  
  Eigen::VectorXcd grad = Eigen::VectorXcd::Zero(num_dofs_);

  const double knorm = k.norm();
  if (knorm < 1e-10) return grad; // gradient = 0 at k=0

  const Eigen::Vector3d khat = k / knorm;

  // the imaginary unit
  static const std::complex<double> I(0.0, 1.0);

  // Per-face loop
  Eigen::VectorXcd init_grad = Eigen::VectorXcd::Zero(num_dofs_);
  Eigen::VectorXcd grad_sum = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, F_.rows()),
    init_grad,
    [&](const tbb::blocked_range<int>& r, Eigen::VectorXcd local_grad) -> Eigen::VectorXcd {
      for (int f = r.begin(); f < r.end(); ++f) {
        const Eigen::Vector3i& vid = F_.row(f);
        Eigen::Vector3d v[3] = { dofs.segment<3>(3 * vid[0]),
          dofs.segment<3>(3 * vid[1]), dofs.segment<3>(3 * vid[2])};

        // edges e1=v1-v0, e2=v2-v0
        Eigen::Vector3d e1 = v[1] - v[0];
        Eigen::Vector3d e2 = v[2] - v[0];

        // Area-vector σn = 0.5 * (v1-v0) x (v2-v0)
        Eigen::Vector3d sigma_n = 0.5 * e1.cross(e2);

        // κ = k̂ · σn (real)
        const double kappa = khat.dot(sigma_n);

        // ∇_v_j κ = 0.5 * k̂ × (v_{j+2} - v_{j+1})
        Eigen::Vector3d dkap[3] = { 0.5 * khat.cross(e2 - e1), -0.5 * khat.cross(e2), 0.5 * khat.cross(e1) };

        // T = exp(i k·v0)
        std::complex<double> T = std::exp(I * k.dot(v[0]));

        // ψ = Φ(k·e1, k·e2)
        double k_dot_e1 = k.dot(e1);
        double k_dot_e2 = k.dot(e2);    
        std::complex<double> psi = Triangle_Phi_ab(k_dot_e1, k_dot_e2);
        // and needed partials:
        auto [dPa, dPb] = Triangle_Phi_ab_gradient(k_dot_e1, k_dot_e2);

        local_grad.segment<3>(3 * vid[0]) += T * (psi * dkap[0] + kappa * k * (psi * I - dPa - dPb));
        local_grad.segment<3>(3 * vid[1]) += T * (psi * dkap[1] + kappa * k * dPa);
        local_grad.segment<3>(3 * vid[2]) += T * (psi * dkap[2] + kappa * k * dPb);
      }
      return local_grad;
    },
    [](const Eigen::VectorXcd& a, const Eigen::VectorXcd& b) -> Eigen::VectorXcd {
      return a + b;
    });
  grad += grad_sum;

  return grad;
}

Eigen::VectorXd TriangleMeshDoF::compute_volume_gradient(
  const Geometry& geom,
  const Eigen::VectorXd& dofs) const
{
  assert(geom.type == GEOM_TRIANGLE);       // geometry type is triangle mesh
  assert(geom.tris.size() == F_.rows());    // number of triangles matches
  assert(dofs.size() == num_dofs_);         // correct number of DoFs
  
  // Initialize gradient
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_dofs_);
  
  // For each triangle, compute its contribution to the volume gradient
  // Volume: V = (1/6) * Σ_triangles |v_i, v_j, v_k|
  Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(num_dofs_);
  Eigen::VectorXd grad_sum = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, F_.rows()),
    init_grad,
    [&](const tbb::blocked_range<int>& r, Eigen::VectorXd local_grad) -> Eigen::VectorXd {
      for (int f = r.begin(); f < r.end(); ++f) {
        int i = F_(f, 0);  // First vertex index
        int j = F_(f, 1);  // Second vertex index
        int k_idx = F_(f, 2);  // Third vertex index
        
        // Triangle vertices
        const Eigen::Vector3d& v_i = dofs.segment<3>(3 * i);
        const Eigen::Vector3d& v_j = dofs.segment<3>(3 * j);
        const Eigen::Vector3d& v_k = dofs.segment<3>(3 * k_idx);

        local_grad.segment<3>(3 * i) += v_j.cross(v_k);
        local_grad.segment<3>(3 * j) += v_k.cross(v_i);
        local_grad.segment<3>(3 * k_idx) += v_i.cross(v_j);
      }
      return local_grad;
    },
    [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) -> Eigen::VectorXd {
      return a + b;
    });
  grad = grad_sum / 6.0;
  
  return grad;
}

} // namespace PoIntInt

