#pragma once
#include "geometry_types.hpp"
#include "geometry_packing.hpp"

namespace PoIntInt {

// Unified intersection volume computation
// type1/type2: 0 = triangle mesh, 1 = point cloud (disks)
double compute_intersection_volume_cuda(
  const std::vector<TriPacked>& tris1,
  const std::vector<DiskPacked>& disks1,
  GeometryType type1,
  const std::vector<TriPacked>& tris2,
  const std::vector<DiskPacked>& disks2,
  GeometryType type2,
  const KGrid& KG,
  int blockSize = 256);

// Legacy function for triangle-triangle intersection
double compute_intersection_volume_cuda(
  const std::vector<TriPacked>& tris1,
  const std::vector<TriPacked>& tris2,
  const KGrid& KG,
  int blockSize = 256);

} // namespace PoIntInt

