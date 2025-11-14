#pragma once
#include "geometry/geometry.hpp"

namespace PoIntInt {

// Compute volume of a single geometry using divergence theorem:
// V = (1/3) ∫_S (x, y, z) · n dS
// where S is the boundary surface and n is the outward normal
double compute_volume_cuda(
  const Geometry& geom,
  int blockSize = 256,
  bool enable_profiling = false);

} // namespace PoIntInt

