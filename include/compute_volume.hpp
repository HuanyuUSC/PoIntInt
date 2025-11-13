#pragma once
#include "geometry_types.hpp"
#include "geometry.hpp"
#include "geometry_packing.hpp"

namespace PoIntInt {

// Main interface using Geometry struct
double compute_intersection_volume_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& KG,
  int blockSize = 256);

} // namespace PoIntInt

