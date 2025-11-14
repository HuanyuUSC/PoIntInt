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

// CPU version of volume computation (for testing/debugging)
// Uses the same formula as the CUDA version but computed on CPU
double compute_volume_cpu(const Geometry& geom);

} // namespace PoIntInt

