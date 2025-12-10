"""
Mathematical primitive functions for form factor computation.

These are the core building blocks used in surface integral evaluation:
- E_func: Auxiliary function for triangle integrals
- Phi_ab: Edge integral function for triangles
- J1_over_x: Bessel function ratio for disk/Gaussian elements

All functions are implemented as Warp device functions (@wp.func)
for use inside GPU kernels.

Ref: cpp/include/cuda/cuda_helpers.hpp, cpp/src/element_functions.cpp
"""

import warp as wp


# =============================================================================
# Complex number utilities (represent complex as wp.vec2d: [real, imag])
# =============================================================================


@wp.func
def cmul(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    """
    Complex multiplication.

    Math:
        (a.x + i·a.y) × (b.x + i·b.y) = (a.x·b.x - a.y·b.y) + i(a.x·b.y + a.y·b.x)
    """
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def cexp_i(phase: wp.float64) -> wp.vec2d:
    """
    Complex exponential exp(i·phase).

    Math:
        exp(i·φ) = cos(φ) + i·sin(φ)
    """
    return wp.vec2d(wp.cos(phase), wp.sin(phase))


@wp.func
def cscale(a: wp.vec2d, s: wp.float64) -> wp.vec2d:
    """Scale complex number by real scalar."""
    return wp.vec2d(a[0] * s, a[1] * s)


# =============================================================================
# Triangle element functions
# =============================================================================


@wp.func
def E_func(z: wp.float64) -> wp.vec2d:
    """
    Auxiliary function for triangle form factor integration.

    Math:
        E(z) = (sin z + i(1 - cos z)) / z

    For small |z| < 1e-4, uses Taylor series to avoid division by zero:
        Re(E) ≈ 1 - z²/6 + z⁴/120
        Im(E) ≈ z/2 - z³/24 + z⁵/720

    Args:
        z: Real argument

    Returns:
        Complex result as vec2d [real, imag]

    Ref: README §2, cpp/include/cuda/cuda_helpers.hpp:200-216
    """
    az = wp.abs(z)

    if az < wp.float64(1e-4):
        # Taylor series for numerical stability
        z2 = z * z
        z4 = z2 * z2
        real = wp.float64(1.0) - z2 / wp.float64(6.0) + z4 / wp.float64(120.0)
        imag = z * wp.float64(0.5) - z * z2 / wp.float64(24.0) + z4 * z / wp.float64(720.0)
        return wp.vec2d(real, imag)
    else:
        s = wp.sin(z)
        c = wp.cos(z)
        return wp.vec2d(s / z, (wp.float64(1.0) - c) / z)


@wp.func
def E_prime(z: wp.float64) -> wp.vec2d:
    """
    Derivative of E(z) with respect to z.

    Math:
        E'(z) = d/dz E(z)
        Re(E') = (z·cos(z) - sin(z)) / z²
        Im(E') = (z·sin(z) - (1 - cos(z))) / z²

    For small |z| < 1e-4:
        Re(E') ≈ -z/3 + z³/30
        Im(E') ≈ 1/2 - z²/8 + z⁴/144

    Ref: cpp/include/cuda/cuda_helpers.hpp:219-240
    """
    az = wp.abs(z)

    if az < wp.float64(1e-4):
        z2 = z * z
        z3 = z2 * z
        z4 = z2 * z2
        real = -z / wp.float64(3.0) + z3 / wp.float64(30.0)
        imag = wp.float64(0.5) - z2 / wp.float64(8.0) + z4 / wp.float64(144.0)
        return wp.vec2d(real, imag)
    else:
        s = wp.sin(z)
        c = wp.cos(z)
        z2 = z * z
        real = (z * c - s) / z2
        imag = (z * s - (wp.float64(1.0) - c)) / z2
        return wp.vec2d(real, imag)


@wp.func
def Phi_ab(alpha: wp.float64, beta: wp.float64) -> wp.vec2d:
    """
    Edge integral function for triangle form factor.

    Math:
        Φ(α, β) = -2i · [E(β) - E(α)] / (β - α)

    When α ≈ β (|β - α| < 1e-3), uses derivative form:
        Φ(α, β) ≈ -2i · E'((α + β) / 2)

    Note: -2i × (re, im) = (2·im, -2·re)

    Args:
        alpha: k · e₁ (edge 1 projection)
        beta: k · e₂ (edge 2 projection)

    Returns:
        Complex result as vec2d

    Ref: README §2, cpp/include/cuda/cuda_helpers.hpp:258-273
    """
    d = beta - alpha

    if wp.abs(d) < wp.float64(1e-3):
        # Use derivative form when α ≈ β
        Ep = E_prime(wp.float64(0.5) * (alpha + beta))
        # -2i × (re, im) = (2·im, -2·re)
        return wp.vec2d(wp.float64(2.0) * Ep[1], wp.float64(-2.0) * Ep[0])
    else:
        Ea = E_func(alpha)
        Eb = E_func(beta)
        # (Eb - Ea) / d
        diff_re = (Eb[0] - Ea[0]) / d
        diff_im = (Eb[1] - Ea[1]) / d
        # -2i × (re, im) = (2·im, -2·re)
        return wp.vec2d(wp.float64(2.0) * diff_im, wp.float64(-2.0) * diff_re)


# =============================================================================
# Disk/Gaussian element functions
# =============================================================================


@wp.func
def J1_over_x(x: wp.float64) -> wp.float64:
    """
    Bessel function ratio J₁(x) / x.

    Used for disk and Gaussian splat form factors. The function
    appears in the Fourier transform of a circular disk.

    Math:
        J₁(x) / x where J₁ is the Bessel function of first kind

    For small |x| < 1e-5, uses Taylor series:
        J₁(x)/x ≈ 1/2 - x²/16 + x⁴/384 - x⁶/18432

    For moderate |x| ≤ 12, uses power series with term recurrence.
    For large |x| > 12, uses Hankel asymptotic expansion.

    Args:
        x: Real argument

    Returns:
        J₁(x) / x

    Ref: cpp/include/cuda/cuda_helpers.hpp:105-154
    """
    ax = wp.abs(x)
    PI = wp.float64(3.14159265358979323846)

    # Tiny x: Taylor expansion
    if ax < wp.float64(1e-5):
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        return wp.float64(0.5) - x2 / wp.float64(16.0) + x4 / wp.float64(384.0) - x6 / wp.float64(18432.0)

    # Small to moderate x: power series
    if ax <= wp.float64(12.0):
        q = wp.float64(0.25) * x * x  # x²/4
        # Declare as dynamic variables (not constants)
        term = wp.float64(0.5)  # m=0 term
        total = wp.float64(0.5)

        # Series: Σ (-1)^m (x/2)^(2m) / (m! (m+1)!)
        # Unroll loop to avoid dynamic loop mutation issues in Warp
        for m in range(20):
            denom = wp.float64(m + 1) * wp.float64(m + 2)
            term = term * (-q) / denom
            total = total + term

        return total

    # Large x: Hankel asymptotic expansion
    invx = wp.float64(1.0) / ax
    invx2 = invx * invx
    invx3 = invx2 * invx

    chi = ax - wp.float64(0.75) * PI  # x - 3π/4
    s = wp.sin(chi)
    c = wp.cos(chi)

    amp = wp.sqrt(wp.float64(2.0) / (PI * ax))
    cosp = (wp.float64(1.0) + wp.float64(15.0) / wp.float64(128.0) * invx2) * c
    sinp = (wp.float64(3.0) / wp.float64(8.0) * invx - wp.float64(105.0) / wp.float64(1024.0) * invx3) * s
    J1 = amp * (cosp - sinp)

    return J1 * invx  # J₁(x) / x
