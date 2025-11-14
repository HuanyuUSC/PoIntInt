#pragma once
#include "geometry/types.hpp"
#include "geometry/geometry.hpp"
#include "geometry/packing.hpp"
#include "quadrature/kgrid.hpp"
#include <Eigen/Dense>
#include <vector>

namespace PoIntInt {

// Result structure for multi-object computation
struct IntersectionVolumeMatrixResult {
  Eigen::MatrixXd volume_matrix;  // nObj × nObj symmetric matrix
  
  // Metadata
  int num_objects() const { return (int)volume_matrix.rows(); }
};

// Main multi-object interface (no gradients, Phase 1)
// Computes pairwise intersection volume matrix V where V[i,j] = intersection volume between objects i and j
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false
);

} // namespace PoIntInt

