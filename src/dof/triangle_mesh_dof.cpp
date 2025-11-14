#include "dof/triangle_mesh_dof.hpp"
#include "element_functions.hpp"
#include <Eigen/Dense>
#include <cassert>

namespace PoIntInt {

TriangleMeshDoF::TriangleMeshDoF(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
  : num_vertices_(V.rows()), num_dofs_(3 * V.rows()), F_(F)
{
  assert(V.cols() == 3 && F.cols() == 3);
  assert(F.maxCoeff() < V.rows());  // Face indices must be valid
}

Geometry TriangleMeshDoF::apply(const Geometry& geom, const Eigen::VectorXd& dofs) const {
  assert(geom.type == GEOM_TRIANGLE);       // geometry type is triangle mesh
  assert(geom.tris.size() == F_.rows());    // number of triangles matches
  assert(dofs.size() == num_dofs_);         // correct number of DoFs
  
  // Reconstruct triangles from new vertex positions
  // Use the stored face connectivity F_
  Geometry transformed = geom;
  for (int i = 0; i < F_.rows(); i++)
  {
    Eigen::Vector3d a = dofs.segment<3>(3 * F_(i, 0));
    Eigen::Vector3d e1 = dofs.segment<3>(3 * F_(i, 1)) - a;
    Eigen::Vector3d e2 = dofs.segment<3>(3 * F_(i, 2)) - a;
    Eigen::Vector3d S = 0.5 * e1.cross(e2);
    transformed.tris[i].a = make_float3((float)a.x(), (float)a.y(), (float)a.z());
    transformed.tris[i].e1 = make_float3((float)e1.x(), (float)e1.y(), (float)e1.z());
    transformed.tris[i].e2 = make_float3((float)e2.x(), (float)e2.y(), (float)e2.z());
    transformed.tris[i].S = make_float3((float)S.x(), (float)S.y(), (float)S.z());
  }
  
  return transformed;
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

  // Per-face loop
  for (int f = 0; f < F_.rows(); ++f) {
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

    // the imaginary unit
    static const std::complex<double> I(0.0, 1.0);

    // T = exp(i k·v0)
    std::complex<double> T = std::exp(I * k.dot(v[0]));

    // ψ = Φ(k·e1, k·e2)
    std::complex<double> psi = Triangle_Phi_ab(k.dot(e1), k.dot(e2));

    A += kappa * T * psi;
  }

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

  // Per-face loop
  for (int f = 0; f < F_.rows(); ++f) {
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

    // the imaginary unit
    static const std::complex<double> I(0.0, 1.0);

    // T = exp(i k·v0)
    std::complex<double> T = std::exp(I * k.dot(v[0]));

    // ψ = Φ(k·e1, k·e2)
    double k_dot_e1 = k.dot(e1);
    double k_dot_e2 = k.dot(e2);    
    std::complex<double> psi = Triangle_Phi_ab(k_dot_e1, k_dot_e2);
    // and needed partials:
    auto [dPa, dPb] = Triangle_Phi_ab_gradient(k_dot_e1, k_dot_e2);

    grad.segment<3>(3 * vid[0]) += T * (psi * dkap[0] + kappa * k * (psi * I - dPa - dPb));
    grad.segment<3>(3 * vid[1]) += T * (psi * dkap[1] + kappa * k * dPa);
    grad.segment<3>(3 * vid[2]) += T * (psi * dkap[2] + kappa * k * dPb);
  }

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
  for (int f = 0; f < F_.rows(); ++f) {
    int i = F_(f, 0);  // First vertex index
    int j = F_(f, 1);  // Second vertex index
    int k = F_(f, 2);  // Third vertex index
    
    // Triangle vertices
    const Eigen::Vector3d& v_i = dofs.segment<3>(3 * i);
    const Eigen::Vector3d& v_j = dofs.segment<3>(3 * j);
    const Eigen::Vector3d& v_k = dofs.segment<3>(3 * k);

    grad.segment<3>(3 * i) += v_j.cross(v_k);
    grad.segment<3>(3 * j) += v_k.cross(v_i);
    grad.segment<3>(3 * k) += v_i.cross(v_j);
  }
  grad /= 6.0;
  
  return grad;
}

} // namespace PoIntInt

