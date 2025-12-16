"""
PoIntInt: Point-based Intersection Volume via Double Surface Integral.

Computes intersection volume using the regularized double surface integral:

    V̂_ε = (1/4π) Σᵢ Σⱼ (n₁ᵢ · n₂ⱼ) · K_ε(‖xᵢ - yⱼ‖) · wᵢ · vⱼ

where K_ε is the ball-averaged Coulomb kernel that regularizes the 1/r
singularity while preserving far-field behavior.

Implementation uses 2D parallelization over all (i,j) pairs with atomic
accumulation distributed across N₁ locations to minimize contention.

Ref: Form factor intersection theory (docs/form_factor_intersection_theory.md)
"""

import math
import numpy as np
import warp as wp


# =============================================================================
# Regularized Kernel (Ball-averaged Coulomb)
# =============================================================================


@wp.func
def K_eps(r: float, eps: float) -> float:
    """
    Ball-averaged Coulomb kernel.

    Replaces each surface sample by a uniform ball of radius a = ε/2,
    and averages the Coulomb interaction over these balls.

    K_ε(r) = 1/r                                  if r ≥ ε
           = (192 - 80t² + 30t³ - t⁵) / (160a)   if r < ε

    where a = ε/2, t = r/a.

    Properties:
        - C² continuous on [0, ∞)
        - Matches 1/r at r = ε up to second derivative
        - K_ε(0) = 6/(5a) = 12/(5ε) < ∞
    """
    if r >= eps:
        return 1.0 / r

    a = eps * 0.5
    t = r / a
    t2 = t * t
    t3 = t2 * t
    t5 = t3 * t2
    return (192.0 - 80.0 * t2 + 30.0 * t3 - t5) / (160.0 * a)


# =============================================================================
# Warp Kernels
# =============================================================================


@wp.kernel
def _pointint_kernel(
    x: wp.array(dtype=wp.vec3),       # (N1,) positions on surface 1
    n1: wp.array(dtype=wp.vec3),      # (N1,) normals on surface 1
    w: wp.array(dtype=float),         # (N1,) area weights on surface 1
    y: wp.array(dtype=wp.vec3),       # (N2,) positions on surface 2
    n2: wp.array(dtype=wp.vec3),      # (N2,) normals on surface 2
    v: wp.array(dtype=float),         # (N2,) area weights on surface 2
    eps: float,                       # regularization parameter ε
    per_point: wp.array(dtype=float), # (N1,) per-point accumulator
):
    """
    Compute pairwise contribution for point pair (i, j).

    2D kernel launch: dim=(N1, N2)
    Each thread computes one (i,j) pair and atomically adds to per_point[i].
    This distributes atomic contention across N1 locations instead of 1.
    """
    i, j = wp.tid()

    r = wp.length(x[i] - y[j])
    dot_nn = wp.dot(n1[i], n2[j])
    contrib = dot_nn * w[i] * v[j] * K_eps(r, eps)

    wp.atomic_add(per_point, i, contrib)


@wp.kernel
def _reduce_sum_kernel(
    per_point: wp.array(dtype=float),
    out: wp.array(dtype=float),
):
    """Reduce per-point contributions to single scalar."""
    i = wp.tid()
    wp.atomic_add(out, 0, per_point[i])


# =============================================================================
# Core API
# =============================================================================


def intersection_volume_direct(
    x: np.ndarray,
    n1: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    n2: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-3,
    device: str = None,
) -> float:
    """
    Compute intersection volume between two point-sampled surfaces.

    Uses the regularized PoIntInt formula:
        V = (1/4π) Σᵢ Σⱼ (n₁ᵢ · n₂ⱼ) · K_ε(‖xᵢ - yⱼ‖) · wᵢ · vⱼ

    where K_ε is the ball-averaged Coulomb kernel.

    Args:
        x: Positions on surface 1, shape (N1, 3)
        n1: Unit normals on surface 1, shape (N1, 3)
        w: Area weights on surface 1, shape (N1,)
        y: Positions on surface 2, shape (N2, 3)
        n2: Unit normals on surface 2, shape (N2, 3)
        v: Area weights on surface 2, shape (N2,)
        eps: Regularization parameter ε (default 1e-3, typically ~sample spacing)
        device: Warp device (default: None for current device)

    Returns:
        Intersection volume as scalar float

    Example:
        >>> vol = pointint_volume(x1, n1, w1, x2, n2, w2)
    """
    N1 = len(x)
    N2 = len(y)

    # Convert to warp arrays
    x_wp = wp.array(x.astype(np.float32), dtype=wp.vec3, device=device)
    n1_wp = wp.array(n1.astype(np.float32), dtype=wp.vec3, device=device)
    w_wp = wp.array(w.astype(np.float32), dtype=float, device=device)
    y_wp = wp.array(y.astype(np.float32), dtype=wp.vec3, device=device)
    n2_wp = wp.array(n2.astype(np.float32), dtype=wp.vec3, device=device)
    v_wp = wp.array(v.astype(np.float32), dtype=float, device=device)

    # Per-point accumulator (N1,) - distributes atomic contention
    per_point = wp.zeros(N1, dtype=float, device=device)

    # 2D kernel launch: full parallelization over N1 × N2 pairs
    wp.launch(
        kernel=_pointint_kernel,
        dim=(N1, N2),
        inputs=[x_wp, n1_wp, w_wp, y_wp, n2_wp, v_wp, eps, per_point],
        device=device,
    )

    # Reduce to scalar
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        kernel=_reduce_sum_kernel,
        dim=N1,
        inputs=[per_point, out],
        device=device,
    )

    # Scale by 1/(4π)
    return float(out.numpy()[0]) / (4.0 * math.pi)

