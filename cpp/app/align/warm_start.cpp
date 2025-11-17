#include "warm_start.hpp"
#include "compute_volume.hpp"
#include <Eigen/Eigenvalues>
#include <iostream>

namespace PoIntInt {

Moments compute_moments(
  const Geometry& geom,
  const std::shared_ptr<AffineDoF>& affine_dof,
  const Eigen::VectorXd& dofs)
{
  Moments moments;
  
  // Translation: dofs[0:3]
  Eigen::Vector3d t = dofs.segment<3>(0);

  // Matrix: dofs[3:12] (row-major)
  Eigen::Matrix3d A = Eigen::Map<const Eigen::Matrix3d>(dofs.data() + 3).transpose();
  
  if (geom.type == GEOM_TRIANGLE) {
    // For triangle meshes, compute moments directly from transformed vertices
    double V = 0.0;
    Eigen::Vector3d first_moment = Eigen::Vector3d::Zero();
    Eigen::Matrix3d second_moment = Eigen::Matrix3d::Zero();
    
    for (const auto& tri : geom.tris) {
      // Get triangle vertices in reference space
      Eigen::Vector3d a_ref(tri.a.x, tri.a.y, tri.a.z);
      Eigen::Vector3d e1_ref(tri.e1.x, tri.e1.y, tri.e1.z);
      Eigen::Vector3d e2_ref(tri.e2.x, tri.e2.y, tri.e2.z);
      
      // Transform vertices
      Eigen::Vector3d a = A * a_ref + t;
      Eigen::Vector3d b = A * (a_ref + e1_ref) + t;
      Eigen::Vector3d c = A * (a_ref + e2_ref) + t;
      
      // Signed volume of tetrahedron (0, a, b, c)
      double vol_tet = (1.0/6.0) * a.dot((b - a).cross(c - a));
      V += vol_tet;
      
      // Centroid of tetrahedron
      Eigen::Vector3d centroid_tet = (a + b + c) / 4.0;
      first_moment += vol_tet * centroid_tet;
      
      // Second moment contribution (simplified - using centroid approximation)
      // For exact computation, we'd need to integrate x*x^T over the tetrahedron
      Eigen::Matrix3d outer = centroid_tet * centroid_tet.transpose();
      second_moment += vol_tet * outer;
    }
    
    if (std::abs(V) > 1e-10) {
      moments.volume = V;
      moments.centroid = first_moment / V;
      
      // Covariance = (1/V) * ∫ (x-c)(x-c)^T dV = (1/V) * ∫ xx^T dV - c*c^T
      moments.covariance = (second_moment / V) - moments.centroid * moments.centroid.transpose();
    } else {
      moments.volume = 0.0;
      moments.centroid = Eigen::Vector3d::Zero();
      moments.covariance = Eigen::Matrix3d::Zero();
    }
  } else {
    // For other geometry types, use divergence theorem via volume computation
    // This is approximate - for exact moments we'd need to extend the divergence theorem
    // For now, use a simplified approach: compute volume and approximate centroid
    
    // Compute volume
    double V = affine_dof->compute_volume(geom, dofs);
    moments.volume = V;
    
    // For non-triangle geometries, approximate centroid using element centers
    // This is not exact but provides a reasonable initialization
    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
    double total_weight = 0.0;
    
    if (geom.type == GEOM_DISK) {
      for (const auto& disk : geom.disks) {
        Eigen::Vector3d c_ref(disk.c.x, disk.c.y, disk.c.z);
        Eigen::Vector3d c = A * c_ref + t;
        double weight = disk.area;
        weighted_sum += weight * c;
        total_weight += weight;
      }
    } else if (geom.type == GEOM_GAUSSIAN) {
      for (const auto& gauss : geom.gaussians) {
        Eigen::Vector3d c_ref(gauss.c.x, gauss.c.y, gauss.c.z);
        Eigen::Vector3d c = A * c_ref + t;
        double weight = gauss.w;
        weighted_sum += weight * c;
        total_weight += weight;
      }
    }
    
    if (total_weight > 1e-10) {
      moments.centroid = weighted_sum / total_weight;
    } else {
      moments.centroid = Eigen::Vector3d::Zero();
    }
    
    // Approximate covariance using element distribution
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    if (geom.type == GEOM_DISK) {
      for (const auto& disk : geom.disks) {
        Eigen::Vector3d c_ref(disk.c.x, disk.c.y, disk.c.z);
        Eigen::Vector3d c = A * c_ref + t;
        Eigen::Vector3d diff = c - moments.centroid;
        cov += disk.area * (diff * diff.transpose());
      }
      if (total_weight > 1e-10) {
        cov /= total_weight;
      }
    } else if (geom.type == GEOM_GAUSSIAN) {
      for (const auto& gauss : geom.gaussians) {
        Eigen::Vector3d c_ref(gauss.c.x, gauss.c.y, gauss.c.z);
        Eigen::Vector3d c = A * c_ref + t;
        Eigen::Vector3d diff = c - moments.centroid;
        // Add both position variance and Gaussian variance
        double sigma_sq = gauss.sigma * gauss.sigma;
        cov += gauss.w * (diff * diff.transpose() + sigma_sq * Eigen::Matrix3d::Identity());
      }
      if (total_weight > 1e-10) {
        cov /= total_weight;
      }
    }
    moments.covariance = cov;
  }
  
  return moments;
}

Eigen::VectorXd compute_warm_start(
  const Geometry& geom1,
  const Geometry& geom2,
  const std::shared_ptr<AffineDoF>& affine_dof)
{
  // Create identity DoF for geom1 (fixed)
  Eigen::VectorXd identity_dofs(12);
  identity_dofs << 0.0, 0.0, 0.0,  // translation
                   1.0, 0.0, 0.0,  // first row of identity matrix
                   0.0, 1.0, 0.0,  // second row of identity matrix
                   0.0, 0.0, 1.0;  // third row of identity matrix
  
  // Compute moments for both geometries
  Moments moments1 = compute_moments(geom1, affine_dof, identity_dofs);
  Moments moments2 = compute_moments(geom2, affine_dof, identity_dofs);
  
  std::cout << "Warm start: Computing affine transform to match moments" << std::endl;
  std::cout << "  Geometry 1: V=" << moments1.volume << ", c=[" 
            << moments1.centroid.transpose() << "]" << std::endl;
  std::cout << "  Geometry 2: V=" << moments2.volume << ", c=[" 
            << moments2.centroid.transpose() << "]" << std::endl;
  
  // Find similarity transform (uniform scaling + rotation) A = s*R, t such that:
  //   1. Translation aligns centroids: t = c1 - A*c2
  //   2. Rotation aligns principal axes (using Procrustes)
  //   3. Uniform scale from volume ratio: s = (V1/V2)^(1/3)
  
  Eigen::Matrix3d R;  // Rotation matrix
  double s;           // Uniform scale
  Eigen::Matrix3d A; // Final transform: A = s*R
  
  // Compute principal axes (eigenvectors) of covariance matrices
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver1(moments1.covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver2(moments2.covariance);
  
  if (solver1.info() != Eigen::Success || solver2.info() != Eigen::Success) {
    std::cout << "Warning: Covariance eigendecomposition failed, using identity transform" << std::endl;
    R = Eigen::Matrix3d::Identity();
    s = 1.0;
  } else {
    // Get eigenvectors (principal axes) - sorted by eigenvalue (largest first)
    Eigen::Matrix3d P1 = solver1.eigenvectors();  // Principal axes of geometry 1
    Eigen::Matrix3d P2 = solver2.eigenvectors();  // Principal axes of geometry 2
    
    // Ensure right-handed coordinate systems (det = 1)
    if (P1.determinant() < 0) {
      P1.col(2) *= -1;  // Flip last axis
    }
    if (P2.determinant() < 0) {
      P2.col(2) *= -1;  // Flip last axis
    }
    
    // Find rotation that best aligns principal axes using SVD (Procrustes)
    // We want R such that R*P2 ≈ P1, i.e., minimize ||P1 - R*P2||_F
    // Solution: R = U*V^T where P1*P2^T = U*S*V^T (SVD)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(P1 * P2.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    
    // Ensure R is a proper rotation (det = 1, not a reflection)
    if (R.determinant() < 0) {
      // If reflection, flip one column of U
      Eigen::Matrix3d U_corrected = svd.matrixU();
      U_corrected.col(2) *= -1;
      R = U_corrected * svd.matrixV().transpose();
    }
    
    // Compute uniform scale from volume ratio
    // For a similarity transform, volume scales as s^3
    if (std::abs(moments2.volume) > 1e-10) {
      s = std::cbrt(moments1.volume / moments2.volume);
    } else {
      // Fallback: use average eigenvalue ratio
      Eigen::Vector3d evals1 = solver1.eigenvalues();
      Eigen::Vector3d evals2 = solver2.eigenvalues();
      double eps = 1e-6;
      double mean_ratio = 0.0;
      int count = 0;
      for (int i = 0; i < 3; ++i) {
        if (evals2(i) > eps) {
          mean_ratio += std::sqrt(evals1(i) / evals2(i));
          count++;
        }
      }
      s = (count > 0) ? (mean_ratio / count) : 1.0;
    }
    
    // Clamp scale to reasonable range
    s = std::max(0.1, std::min(10.0, s));
  }
  
  // Final transform: A = s*R
  A = s * R;
  
  // Compute translation to align centroids
  // t = c1 - A*c2 = c1 - s*R*c2
  Eigen::Vector3d t = moments1.centroid - A * moments2.centroid;
  
  std::cout << "  Similarity transform: scale = " << s << std::endl;
  std::cout << "  Rotation determinant: " << R.determinant() << std::endl;
  
  // Pack into DoF vector (translation first, then matrix row-major)
  Eigen::VectorXd warm_start_dofs(12);
  warm_start_dofs.segment<3>(0) = t;
  warm_start_dofs.segment<3>(3) = A.row(0);
  warm_start_dofs.segment<3>(6) = A.row(1);
  warm_start_dofs.segment<3>(9) = A.row(2);
  
  std::cout << "  Warm start transform computed" << std::endl;
  
  return warm_start_dofs;
}

} // namespace PoIntInt

