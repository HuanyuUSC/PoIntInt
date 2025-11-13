#pragma once
#include "geometry_types.hpp"
#include <vector>

namespace PoIntInt {

// Unified geometry container supporting multiple geometry types
struct Geometry {
  GeometryType type;
  std::vector<TriPacked> tris;
  std::vector<DiskPacked> disks;
  // Future: std::vector<GaussianPacked> gaussians;
  
  // Constructors
  Geometry() : type(GEOM_TRIANGLE) {}
  explicit Geometry(GeometryType t) : type(t) {}
  
  // Metadata
  int num_elements() const {
    switch (type) {
      case GEOM_TRIANGLE: return (int)tris.size();
      case GEOM_DISK: return (int)disks.size();
      default: return 0;
    }
  }
  
  bool is_empty() const {
    return num_elements() == 0;
  }
  
  // Clear all geometry data
  void clear() {
    tris.clear();
    disks.clear();
  }
};

} // namespace PoIntInt

