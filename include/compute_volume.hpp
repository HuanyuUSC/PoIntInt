#pragma once
#include "geometry/types.hpp"
#include "geometry/geometry.hpp"
#include "geometry/packing.hpp"

namespace PoIntInt {

// Main interface using Geometry struct
double compute_intersection_volume_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& KG,
  int blockSize = 256,
  bool enable_profiling = false);

} // namespace PoIntInt

