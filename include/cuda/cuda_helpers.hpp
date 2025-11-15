#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <math_constants.h>

namespace PoIntInt {

// ===================== 3x3 Matrix Helper Functions =====================

// Simple 3x3 matrix structure (row-major storage)
struct Mat3x3 {
  double3 row0, row1, row2;
  
  __device__ __forceinline__ Mat3x3() {}
  
  __device__ __forceinline__ Mat3x3(double3 r0, double3 r1, double3 r2) 
    : row0(r0), row1(r1), row2(r2) {}
  
  // Access element at (row, col)
  __device__ __forceinline__ double& operator()(int row, int col) {
    if (row == 0) {
      if (col == 0) return row0.x; else if (col == 1) return row0.y; else return row0.z;
    } else if (row == 1) {
      if (col == 0) return row1.x; else if (col == 1) return row1.y; else return row1.z;
    } else {
      if (col == 0) return row2.x; else if (col == 1) return row2.y; else return row2.z;
    }
  }
  
  __device__ __forceinline__ double operator()(int row, int col) const {
    if (row == 0) {
      if (col == 0) return row0.x; else if (col == 1) return row0.y; else return row0.z;
    } else if (row == 1) {
      if (col == 0) return row1.x; else if (col == 1) return row1.y; else return row1.z;
    } else {
      if (col == 0) return row2.x; else if (col == 1) return row2.y; else return row2.z;
    }
  }
};

// Matrix-vector multiplication: result = M * v
__device__ __forceinline__ double3 mat3x3_mul_vec(const Mat3x3& M, const double3& v) {
  return make_double3(
    M.row0.x*v.x + M.row0.y*v.y + M.row0.z*v.z,
    M.row1.x*v.x + M.row1.y*v.y + M.row1.z*v.z,
    M.row2.x*v.x + M.row2.y*v.y + M.row2.z*v.z
  );
}

// Matrix-matrix multiplication: result = A * B
__device__ __forceinline__ Mat3x3 mat3x3_mul_mat(const Mat3x3& A, const Mat3x3& B) {
  // Compute each row of result
  double3 r0 = make_double3(
    A.row0.x*B.row0.x + A.row0.y*B.row1.x + A.row0.z*B.row2.x,
    A.row0.x*B.row0.y + A.row0.y*B.row1.y + A.row0.z*B.row2.y,
    A.row0.x*B.row0.z + A.row0.y*B.row1.z + A.row0.z*B.row2.z
  );
  double3 r1 = make_double3(
    A.row1.x*B.row0.x + A.row1.y*B.row1.x + A.row1.z*B.row2.x,
    A.row1.x*B.row0.y + A.row1.y*B.row1.y + A.row1.z*B.row2.y,
    A.row1.x*B.row0.z + A.row1.y*B.row1.z + A.row1.z*B.row2.z
  );
  double3 r2 = make_double3(
    A.row2.x*B.row0.x + A.row2.y*B.row1.x + A.row2.z*B.row2.x,
    A.row2.x*B.row0.y + A.row2.y*B.row1.y + A.row2.z*B.row2.y,
    A.row2.x*B.row0.z + A.row2.y*B.row1.z + A.row2.z*B.row2.z
  );
  return Mat3x3(r0, r1, r2);
}

// Cross product matrix [v]_×
__device__ __forceinline__ Mat3x3 cross_product_matrix(const double3& v) {
  return Mat3x3(
    make_double3(0.0, -v.z, v.y),
    make_double3(v.z, 0.0, -v.x),
    make_double3(-v.y, v.x, 0.0)
  );
}

// ===================== Complex Number Utilities =====================

__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
  return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ double2 cexp_i(double phase) {
  double s, c;
  sincos(phase, &s, &c);
  return make_double2(c, s);
}

__device__ __forceinline__ double2 cadd(double2 a, double2 b) {
  return make_double2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ double2 cscale(double2 a, double s) {
  return make_double2(a.x * s, a.y * s);
}

// ===================== Element Functions =====================

// J1_over_x(x) := J1(x)/x  (accurate ~1e-6 in double across full range)
__device__ __forceinline__ double J1_over_x(double x) {
  double ax = fabs(x);

  // Tiny x: even Taylor up to x^8
  if (ax < 1e-3) {
    double x2 = x * x;
    double t = 0.5;                      // 1/2
    t += (-1.0 / 16.0) * x2;              // - x^2/16
    double x4 = x2 * x2;
    t += (1.0 / 384.0) * x4;              // + x^4/384
    double x6 = x4 * x2;
    t += (-1.0 / 18432.0) * x6;           // - x^6/18432
    return t;
  }

  // Small–moderate x: stable series with term recurrence
  if (ax <= 12.0) {
    double q = 0.25 * x * x;            // x^2/4
    double term = 0.5;                    // m=0
    double sum = term;
#pragma unroll
    for (int m = 0; m < 20; ++m) {
      double denom = (double)(m + 1) * (double)(m + 2);
      term *= -q / denom;              // term_{m+1}
      sum += term;
      if (fabs(term) < 1e-7 * fabs(sum)) break;
    }
    return sum;
  }

  // Large x: Hankel asymptotics (3 correction terms), then divide by x
  double invx = 1.0 / ax;
  double invx2 = invx * invx;
  double invx3 = invx2 * invx;

  double chi = ax - 0.75 * CUDART_PI;
  double s, c;
  sincos(chi, &s, &c);

  double amp = sqrt(2.0 / (CUDART_PI * ax));
  double cosp = (1.0 - 15.0 / 128.0 * invx2) * c;
  double sinp = (3.0 / 8.0 * invx - 315.0 / 3072.0 * invx3) * s;
  double J1 = amp * (cosp - sinp);
  return J1 * invx;                          // J1(x)/x
}

// E(z) = (sin z + i (1 - cos z)) / z, stable small-z
__device__ __forceinline__ double2 E_func(double z) {
  double az = fabs(z);
  double threshold = 1e-4;
  if (az < threshold) {
    // series: sin z / z ≈ 1 - z^2/6 + z^4/120
    //         (1 - cos z)/z ≈ z/2 - z^3/24 + z^5/720
    double z2 = z*z, z4 = z2*z2;
    double real = 1.0 - z2*(1.0/6.0) + z4*(1.0/120.0);
    double imag = z*(0.5) - z*z2*(1.0/24.0) + z4*z*(1.0/720.0);
    return make_double2(real, imag);
  } else {
    double s, c;
    sincos(z, &s, &c);
    return make_double2(s/z, (1.0 - c)/z);
  }
}

// E'(z) = d/dz E(z)  (stable)
__device__ __forceinline__ double2 E_prime(double z) {
  double az = fabs(z);
  double threshold = 1e-4;
  if (az < threshold) {
    // from series:
    // Re ≈ -(1/3) z + (1/30) z^3
    // Im ≈  1/2  - (1/8) z^2 + (1/144) z^4
    double z2 = z*z, z3 = z2*z, z4 = z2*z2;
    double real = -(1.0/3.0)*z + (1.0/30.0)*z3;
    double imag =  0.5 - (1.0/8.0)*z2 + (1.0/144.0)*z4;
    return make_double2(real, imag);
  } else {
    double s, c;
    sincos(z, &s, &c);
    // Re: (z cos z - sin z)/z^2
    // Im: (z sin z - (1 - cos z))/z^2
    double z2 = z*z;
    double re = (z*c - s)/z2;
    double im = (z*s - (1.0 - c))/z2;
    return make_double2(re, im);
  }
}

// E''(z) = d^2/dz^2 E(z)
__device__ __forceinline__ double2 E_double_prime(double z) {
  double z2 = z*z;
  if (fabs(z) < 1e-4) {
    // E''(z) approximation for small z
    return make_double2(-1.0/3.0 + z2/10.0, -z/4.0 + z2*z/36.0);
  } else {
    double s = sin(z);
    double c = cos(z);
    double z3 = z2*z;
    double re = (2.0*s - 2.0*z*c - z2*s)/z3;
    double im = (2.0*(1.0 - c) - 2.0*z*s + z2*c)/z3;
    return make_double2(re, im);
  }
}

// Phi_ab(alpha, beta) = -2i * (E(beta) - E(alpha)) / (beta - alpha)
__device__ __forceinline__ double2 Phi_ab(double alpha, double beta) {
  double d = beta - alpha;
  double threshold = 1e-3;  // Match CPU: 1e-3
  if (fabs(d) < threshold) {
    double2 Ep = E_prime(0.5*(alpha+beta));
    return make_double2(2.0*Ep.y, -2.0*Ep.x);
  } else {
    double2 Ea = E_func(alpha);
    double2 Eb = E_func(beta);
    double2 num = make_double2(Eb.x - Ea.x, Eb.y - Ea.y);
    double invd = 1.0/d;
    double2 q = make_double2(num.x*invd, num.y*invd);
    return make_double2(2.0*q.y, -2.0*q.x);
  }
}

// Compute gradient of Phi_ab w.r.t. alpha and beta
// Returns: (dPhi/dalpha, dPhi/dbeta) as double2 pairs
__device__ __forceinline__ void Phi_ab_gradient(double alpha, double beta, double2* dPhi_dalpha, double2* dPhi_dbeta) {
  double d = beta - alpha;
  double threshold = 1e-3;  // Match CPU: 1e-3
  
  if (fabs(d) < threshold) {
    // When alpha ≈ beta, use E''(alpha)
    // Phi ≈ -2i * E'((alpha+beta)/2)
    // dPhi/dalpha = dPhi/dbeta = -i * E''((alpha+beta)/2)
    double z = 0.5 * (alpha + beta);
    double2 Epp = E_double_prime(z);
    // -i * (re, im) = (im, -re)
    *dPhi_dalpha = make_double2(Epp.y, -Epp.x);
    *dPhi_dbeta = make_double2(Epp.y, -Epp.x);
  } else {
    // General case: Phi = -2i * (Eb - Ea) / d
    // dPhi/dalpha = -2i * [(Eb - Ea)/d^2 - Ea'/d]
    // dPhi/dbeta = 2i * [(Eb - Ea)/d^2 - Eb'/d]
    // Match CPU implementation exactly
    double2 Ea = E_func(alpha);
    double2 Eb = E_func(beta);
    double2 Ea_prime = E_prime(alpha);
    double2 Eb_prime = E_prime(beta);
    
    double invd = 1.0 / d;
    double invd2 = invd * invd;
    
    // term1 = (Eb - Ea) / d^2
    double2 Eb_minus_Ea = make_double2(Eb.x - Ea.x, Eb.y - Ea.y);
    double2 term1 = make_double2(Eb_minus_Ea.x * invd2, Eb_minus_Ea.y * invd2);
    
    // Ea'/d and Eb'/d
    double2 Ea_prime_over_d = make_double2(Ea_prime.x * invd, Ea_prime.y * invd);
    double2 Eb_prime_over_d = make_double2(Eb_prime.x * invd, Eb_prime.y * invd);
    
    // dPhi/dalpha = -2i * (term1 - Ea_prime_over_d)
    // -2i * (re, im) = (2*im, -2*re)
    double2 diff_alpha = make_double2(term1.x - Ea_prime_over_d.x, term1.y - Ea_prime_over_d.y);
    *dPhi_dalpha = make_double2(2.0 * diff_alpha.y, -2.0 * diff_alpha.x);
    
    // dPhi/dbeta = 2i * (term1 - Eb_prime_over_d)
    // 2i * (re, im) = (-2*im, 2*re)
    double2 diff_beta = make_double2(term1.x - Eb_prime_over_d.x, term1.y - Eb_prime_over_d.y);
    *dPhi_dbeta = make_double2(-2.0 * diff_beta.y, 2.0 * diff_beta.x);
  }
}

} // namespace PoIntInt

