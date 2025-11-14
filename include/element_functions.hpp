#pragma once
#include <complex>

namespace PoIntInt {

// ============================================================================
// Auxilliary Functions for Triangle Elements
// ============================================================================

// Compute E(z) = (sin z + i(1-cos z))/z
std::complex<double> Triangle_E_func(double z);

// Compute E'(z) = d/dz E(z)
std::complex<double> Triangle_E_prime(double z);

// Compute E''(z) = d^2/dz^2 E(z)
std::complex<double> Triangle_E_double_prime(double z);

// Compute Phi_ab(alpha, beta) = -2i [E(β) - E(α)]/(β-α)
std::complex<double> Triangle_Phi_ab(double alpha, double beta);

// Compute partial derivatives of Phi_ab w.r.t. alpha and beta
// Returns: (dPhi/dalpha, dPhi/dbeta)
std::pair<std::complex<double>, std::complex<double>> 
  Triangle_Phi_ab_gradient(double alpha, double beta);


// ============================================================================
// Auxilliary Functions for Disk Elements
// ============================================================================

// Compute J1(z)/z
double Disk_J1_over_x(double z);

} // namespace PoIntInt

