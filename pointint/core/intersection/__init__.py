"""
Intersection Volume Computation using NVIDIA Warp.

This module computes intersection volumes between 3D solids using a
Fourier (k-space) method. The core algorithm:

1. Express intersection volume as a double surface integral over boundaries
2. Rewrite the 1/||x-y|| kernel via its Fourier representation
3. Discretize k-space (Gauss-Legendre × Lebedev) and precompute form factors
4. Recover pairwise overlap volumes from weighted inner products

Math (README §1.1):
    Vol(Ω₁ ∩ Ω₂) = (1/4π) ∫∫ [n₁(x)·n₂(y) / ||x-y||] dS_x dS_y

Fourier formulation (README §2):
    Vol(Ω₁ ∩ Ω₂) = (1/8π³) ∫ [A₁(k)·conj(A₂(k)) / ||k||²] dk

where A(k) = ∫_∂Ω exp(ik·x) n(x) dS_x is the form factor.

Ref: README.md, cpp/src/compute_intersection_volume.cu
"""

from pointint.core.intersection.primitives import E_func, Phi_ab, J1_over_x
from pointint.core.intersection.kgrid import build_kgrid, load_lebedev
from pointint.core.intersection.form_factors import compute_A_triangles
from pointint.core.intersection.volume import intersection_volume

__all__ = [
    "E_func",
    "Phi_ab",
    "J1_over_x",
    "build_kgrid",
    "load_lebedev",
    "compute_A_triangles",
    "intersection_volume",
]
