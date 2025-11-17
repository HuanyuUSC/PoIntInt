#include "element_functions.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

// The imaginary unit
static const std::complex<double> I(0.0, 1.0);

// ============================================================================
// Auxilliary Functions for Triangle Elements
// ============================================================================

std::complex<double> Triangle_E_func(double z) {
  double az = std::abs(z);
  if (az < 1e-4) {
    double z2 = z * z, z4 = z2 * z2;
    double real = 1.0 - z2 / 6.0 + z4 / 120.0;
    double imag = z * 0.5 - z * z2 / 24.0 + z4 * z / 720.0;
    return std::complex<double>(real, imag);
  }
  double s = std::sin(z);
  double c = std::cos(z);
  return std::complex<double>(s / z, (1.0 - c) / z);
}

std::complex<double> Triangle_E_prime(double z) {
  double az = std::abs(z);
  if (az < 1e-4) {
    double z2 = z * z, z3 = z2 * z, z4 = z2 * z2;
    double real = -z / 3.0 + z3 / 30.0;
    double imag = 0.5 - z2 / 8.0 + z4 / 144.0;
    return std::complex<double>(real, imag);
  }
  double s = std::sin(z);
  double c = std::cos(z);
  double z2 = z * z;
  double re = (z * c - s) / z2;
  double im = (z * s - (1.0 - c)) / z2;
  return std::complex<double>(re, im);
}

std::complex<double> Triangle_E_double_prime(double z) {
  double z2 = z * z;
  if (std::abs(z) < 1e-4) {
    // E''(z) approximation for small z
    return std::complex<double>(-1.0 / 3.0 + z2 / 10.0, -z / 4.0 + z2 * z / 36.0);
  }
  else {
    double s = std::sin(z);
    double c = std::cos(z);
    double z3 = z * z * z;
    double re = (2.0 * s - 2.0 * z * c - z2 * s) / z3;
    double im = (2.0 * (1.0 - c) - 2.0 * z * s + z2 * c) / z3;
    return std::complex<double>(re, im);
  }
}

// Phi(alpha, beta) = -2i [E(beta) - E(alpha)] / (beta - alpha)
std::complex<double> Triangle_Phi_ab(double alpha, double beta) {
  double d = beta - alpha;
  if (std::abs(d) < 1e-3) {
    return -2.0 * I * Triangle_E_prime(0.5 * (alpha + beta));
  } else {
    return -2.0 * I * (Triangle_E_func(beta) - Triangle_E_func(alpha)) / d;
  }
}

// compute dPhi/dalpha and dPhi/dbeta
std::pair<std::complex<double>, std::complex<double>> 
Triangle_Phi_ab_gradient(double alpha, double beta) {
  double d = beta - alpha;  
  if (std::abs(d) < 1e-3) {
    // When alpha ≈ beta, use second derivative
    // Phi ≈ -2i * E'((alpha+beta)/2)
    // dPhi/dalpha = dPhi/dbeta = -i * E''((alpha+beta)/2)
    double z = 0.5 * (alpha + beta);
    std::complex<double> dPhi_dz = -I * Triangle_E_double_prime(z);
    return std::make_pair(dPhi_dz, dPhi_dz);
  } else {
    // General case: Phi = -2i * (Eb - Ea) / d
    // dPhi/dalpha = -2i * [(Eb - Ea)/d^2 - Ea'/d]
    // dPhi/dbeta = -2i * [-(Eb - Ea)/d^2 + Eb'/d] = 2i * [(Eb - Ea)/d^2 - Eb'/d]
    double d2 = d * d;
    std::complex<double> term1 = (Triangle_E_func(beta) - Triangle_E_func(alpha)) / d2;
    std::complex<double> dPhi_dalpha = -2.0 * I * (term1 - Triangle_E_prime(alpha) / d);
    std::complex<double> dPhi_dbeta = 2.0 * I * (term1 - Triangle_E_prime(beta) / d);
    return std::make_pair(dPhi_dalpha, dPhi_dbeta);
  }
}


// ============================================================================
// Auxilliary Functions for Disk Elements
// ============================================================================

double Disk_J1_over_x(double x)
{
  double ax = std::abs(x);
  if (ax < 1e-5) {
    // From series: J1(x)/x ≈ 1/2 - x^2/16 + x^4/384
    double x2 = x * x;
    return 0.5 - x2 / 16.0 + x2 * x2 / 384.0;
  }
  return std::cyl_bessel_j(1, x) / x;
}

double Disk_J1_over_x_prime(double x)
{
  const double ax = std::abs(x);
  if (ax < 1e-5) {
    // From series: (J1(x)/x)' ≈ -x/8 + x^3/96
    return -x / 8.0 + (x * x * x) / 96.0;
  }
  // -J2 / x
  return -std::cyl_bessel_j(2, x) / x;
}

} // namespace PoIntInt

