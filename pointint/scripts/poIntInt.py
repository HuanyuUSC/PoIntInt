import numpy as np, math
from fmm3dpy import lfmm3d  # Laplace FMM wrapper  (1/r kernel)  # pip install fmm3dpy
# If import fails, see: https://fmm3d.readthedocs.io/en/latest/python.html

# -------- helpers: sampling, exact sphere intersection, and estimator glue --------

def fibonacci_sphere(n, R=1.0, center=(0,0,0)):
    i = np.arange(n, dtype=float) + 0.5
    phi = (1 + 5**0.5) / 2
    theta = 2*np.pi * i / phi
    z = 1 - 2*i/n
    r = np.sqrt(np.clip(1 - z*z, 0.0, 1.0))
    x = r*np.cos(theta); y = r*np.sin(theta)
    pts_unit = np.stack([x,y,z], axis=1)
    pts = np.asarray(center)[None,:] + R*pts_unit
    normals = pts_unit               # outward
    w = np.full(n, 4*np.pi*R*R/n)    # equal area weights on sphere
    return pts, normals, w

def exact_sphere_intersection(R, r, d):
    if d >= R + r: return 0.0
    if d <= abs(R - r): return 4.0/3.0 * math.pi * min(R, r)**3
    term1 = (R + r - d)**2
    term2 = (d**2 + 2*d*(R + r) - 3*(R - r)**2)
    return math.pi * term1 * term2 / (12.0 * d)

def pointint_direct(x1, n1, w1, x2, n2, w2, eps=0.0):
    x1e = x1[:, None, :]
    x2e = x2[None, :, :]
    diff = x1e - x2e
    r = np.linalg.norm(diff, axis=2)
    if eps > 0:
        r = np.sqrt(r*r + eps*eps)
    ndot = (n1[:, None, :] * n2[None, :, :]).sum(axis=2)
    W = (w1[:, None] * w2[None, :])
    integral = (W * ndot / r).sum()
    return integral / (4*math.pi)

def pointint_via_fmm3d(x1, n1, w1, x2, n2, w2, eps=1e-6):
    """
    Compute (1/4π) * sum_i w1_i n1_i · sum_j (w2_j n2_j)/||x1_i - x2_j||   using FMM.
    Implemented by 3-channel Laplace potentials with charges c_j^(k)=w2_j*n2_jk.
    """
    # FMM3D expects shape (3, n) arrays for sources/targets.
    src = x2.T.copy()                          # (3, n2)
    trg = x1.T.copy()                          # (3, n1)

    # charges: shape can be (nd, nsources). We'll use nd=3 for x,y,z components.
    charges = (w2[:,None] * n2).T              # shape (3, n2)

    # Run Laplace FMM: request potentials at targets (pgt=1); not asking for gradients.
    out = lfmm3d(eps=eps, sources=src, charges=charges, targets=trg, pgt=1, nd=3)
    # out.pottarg has shape (nd, nt) => rows are Ux, Uy, Uz at the targets.
    U = out.pottarg.T                          # (n1, 3)

    # Assemble the scalar: (1/4π) sum_i w1_i * (n1_i · U_i)
    return float((w1 * np.sum(n1 * U, axis=1)).sum() / (4*np.pi))

# -------- demo: two unit spheres, vary d and n, compare to exact --------

def demo(R=1.0, d=1.6, n=200, eps=1e-6, seed=0):
    c1 = np.array([0.0,0.0,0.0])
    c2 = np.array([d,  0.0,0.0])

    x1, n1, w1 = fibonacci_sphere(n, R, c1)
    x2, n2, w2 = fibonacci_sphere(n, R, c2)

    V_exact = exact_sphere_intersection(R, R, d)
    V_fmm   = pointint_via_fmm3d(x1, n1, w1, x2, n2, w2, eps=eps)
    V_direct = pointint_direct(x1, n1, w1, x2, n2, w2, eps=eps)

    abs_err_fmm = abs(V_fmm - V_exact)
    rel_err_fmm = abs_err_fmm / V_exact if V_exact > 0 else (0.0 if abs(V_fmm) < 1e-14 else np.inf)

    abs_err_direct = abs(V_direct - V_exact)
    rel_err_direct = abs_err_direct / V_exact if V_exact > 0 else (0.0 if abs(V_direct) < 1e-14 else np.inf)

    print(f"d={d}, n={n}, eps={eps}")-
    print(f"  V_exact = {V_exact:.10f}")
    print(f"  V_fmm   = {V_fmm:.10f}")
    print(f"  V_direct   = {V_direct:.10f}")
    print(f"  abs_err_fmm = {abs_err_fmm:.3e}  rel_err_fmm = {rel_err_fmm:.3e}")
    print(f"  abs_err_direct = {abs_err_direct:.3e}  rel_err_direct = {rel_err_direct:.3e}")

if __name__ == "__main__":
    # Try a few resolutions/separations
    for d in [0.8, 1.2, 1.6, 1.9, 2.1]:
        for n in [200, 400, 800]:
            demo(R=1.0, d=d, n=n, eps=1e-6)
