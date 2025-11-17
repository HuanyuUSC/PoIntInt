#pragma once

namespace PoIntInt {

// ============================================================================
// Computation Flags
// ============================================================================
// Shared flags for unified volume computation interfaces (intersection volume,
// intersection volume matrix, self volume, etc.)
// ============================================================================

enum class ComputationFlags {
  VOLUME_ONLY = 0x01,
  GRADIENT = 0x02,
  HESSIAN = 0x04,
  ALL = 0x07  // VOLUME_ONLY | GRADIENT | HESSIAN
};

// Bitwise OR operator for ComputationFlags
inline ComputationFlags operator|(ComputationFlags a, ComputationFlags b) {
  return static_cast<ComputationFlags>(static_cast<int>(a) | static_cast<int>(b));
}

// Bitwise AND operator for ComputationFlags
inline ComputationFlags operator&(ComputationFlags a, ComputationFlags b) {
  return static_cast<ComputationFlags>(static_cast<int>(a) & static_cast<int>(b));
}

// Helper functions for flag checking
inline bool needs_volume(ComputationFlags flags) {
  return (static_cast<int>(flags) & static_cast<int>(ComputationFlags::VOLUME_ONLY)) != 0;
}

inline bool needs_gradient(ComputationFlags flags) {
  return (static_cast<int>(flags) & static_cast<int>(ComputationFlags::GRADIENT)) != 0;
}

inline bool needs_hessian(ComputationFlags flags) {
  return (static_cast<int>(flags) & static_cast<int>(ComputationFlags::HESSIAN)) != 0;
}

} // namespace PoIntInt
