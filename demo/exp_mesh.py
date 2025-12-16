"""
Mesh-Mesh Intersection Volume Experiment (Fox Model)

This script benchmarks intersection volume computation between two meshes
using four different backends:
- Warp Fourier (GPU)
- Warp Direct (GPU)
- C++ CUDA (GPU)
- C++ CPU

Since there is no analytical ground truth for mesh intersection,
C++ CUDA is used as the reference baseline for comparison.

Usage:
    python exp_mesh.py [mesh_path]

    If no mesh_path is provided, uses the default fox.obj mesh.

Output:
    exp_mesh_results.csv - Detailed results for all experiments
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import trimesh

# Import C++ bindings
import pointint_cpp as pi

# Warp setup
import warp as wp
wp.init()

# Import Warp-based intersection volume functions
from pointint.core.intersection import (
    build_kgrid as build_kgrid_warp,
    load_lebedev as load_lebedev_warp,
)
from pointint.core.intersection.volume import intersection_volume_mesh
from pointint.core.intersection.volume_direct import intersection_volume_direct


def load_mesh(path: str) -> tuple:
    """Load mesh from file and return vertices (V) and faces (F)."""
    mesh = trimesh.load(path, force='mesh')
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int32)
    return V, F


def normalize_mesh(V: np.ndarray) -> np.ndarray:
    """Center and scale mesh to fit in [-0.5, 0.5]^3."""
    center = (V.max(axis=0) + V.min(axis=0)) / 2
    V_centered = V - center
    scale = V_centered.max() - V_centered.min()
    if scale > 0:
        V_centered = V_centered / scale
    return V_centered


def main():
    print("=" * 60)
    print("Mesh-Mesh Intersection Volume Experiment")
    print("=" * 60)

    # =========================================================================
    # Setup
    # =========================================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get mesh path from command line or use default
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
    else:
        mesh_path = os.path.join(script_dir, "../data/assets/fox.obj")

    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        sys.exit(1)

    # Load and normalize mesh
    print(f"Loading mesh from: {mesh_path}")
    V_orig, F = load_mesh(mesh_path)
    V = normalize_mesh(V_orig)
    print(f"Mesh loaded: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Compute mesh volume
    geom_base = pi.make_triangle_mesh(V, F)
    mesh_volume = pi.compute_volume_cpu(geom_base)
    print(f"Mesh volume: {mesh_volume:.6f}")

    # Load Lebedev quadrature grid
    lebedev_path = os.path.join(script_dir, "../data/lebedev/lebedev_029.txt")

    if not os.path.exists(lebedev_path):
        print(f"Error: Lebedev file not found at {lebedev_path}")
        sys.exit(1)

    print(f"Loading Lebedev grid from: {lebedev_path}")

    # C++ backend setup
    dirs_cpp, weights_cpp = pi.load_lebedev(lebedev_path)

    # Warp backend setup
    leb_dirs_warp, leb_w_warp = load_lebedev_warp(lebedev_path)

    # =========================================================================
    # Experiment Parameters
    # =========================================================================

    distances = np.arange(0, 2.1, 0.1)  # 0, 0.1, 0.2, ..., 2.0
    methods = ["Warp Fourier", "Warp Direct", "C++ CUDA", "C++ CPU"]
    n_runs = 5
    n_radial = 32

    print(f"\nExperiment parameters:")
    print(f"  Mesh: {os.path.basename(mesh_path)}")
    print(f"  Distances: {len(distances)} points from 0 to 2")
    print(f"  Methods: {methods}")
    print(f"  Runs per configuration: {n_runs}")
    print(f"  N_radial: {n_radial}")
    print(f"  Total experiments: {len(methods) * len(distances) * n_runs}")

    # Build kgrids
    kgrid_cpp = pi.build_kgrid(dirs_cpp, weights_cpp, n_radial)
    kgrid_warp = build_kgrid_warp(leb_dirs_warp, leb_w_warp, n_radial)
    print(f"KGrid created with {len(dirs_cpp) * n_radial} nodes")

    # =========================================================================
    # Pre-compute Reference Values (C++ CUDA)
    # =========================================================================

    print("\nPre-computing reference values (C++ CUDA)...")
    ref_volumes = {}
    for d in distances:
        V2 = V + np.array([d, 0.0, 0.0])
        geom1 = pi.make_triangle_mesh(V, F)
        geom2 = pi.make_triangle_mesh(V2, F)
        ref_volumes[round(d, 1)] = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid_cpp)
    print("Reference values computed.")

    # =========================================================================
    # Run Experiments
    # =========================================================================

    results = []
    total_exp = len(methods) * len(distances) * n_runs
    exp_count = 0

    print("\nRunning experiments...")

    for method in methods:
        print(f"\n  Method: {method}")

        for d in distances:
            # Translate second mesh
            V2 = V + np.array([d, 0.0, 0.0])
            ref_vol = ref_volumes[round(d, 1)]

            for run in range(1, n_runs + 1):
                exp_count += 1

                # Warmup for first run
                if run == 1 and d == 0:
                    if method == "Warp Fourier":
                        _ = intersection_volume_mesh(V, F, V2, F, kgrid_warp)
                    elif method == "Warp Direct":
                        _ = intersection_volume_direct(V, F, V2, F)
                    elif method == "C++ CUDA":
                        geom1 = pi.make_triangle_mesh(V, F)
                        geom2 = pi.make_triangle_mesh(V2, F)
                        _ = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid_cpp)
                    else:  # C++ CPU
                        geom1 = pi.make_triangle_mesh(V, F)
                        geom2 = pi.make_triangle_mesh(V2, F)
                        _ = pi.compute_intersection_volume_cpu(geom1, geom2, kgrid_cpp)

                # Timed computation
                t0 = time.time()

                if method == "Warp Fourier":
                    vol = intersection_volume_mesh(V, F, V2, F, kgrid_warp)
                elif method == "Warp Direct":
                    vol = intersection_volume_direct(V, F, V2, F)
                elif method == "C++ CUDA":
                    geom1 = pi.make_triangle_mesh(V, F)
                    geom2 = pi.make_triangle_mesh(V2, F)
                    vol = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid_cpp)
                else:  # C++ CPU
                    geom1 = pi.make_triangle_mesh(V, F)
                    geom2 = pi.make_triangle_mesh(V2, F)
                    vol = pi.compute_intersection_volume_cpu(geom1, geom2, kgrid_cpp)

                elapsed = time.time() - t0

                # Compute percentages
                intersection_pct = (vol / mesh_volume * 100) if mesh_volume > 1e-10 else 0.0
                if ref_vol > 1e-10:
                    diff_vs_ref = abs(vol - ref_vol) / ref_vol * 100
                else:
                    diff_vs_ref = 0.0 if abs(vol) < 1e-10 else float('inf')

                results.append({
                    'method': method,
                    'run': run,
                    'distance': round(d, 1),
                    'time_ms': elapsed * 1000,
                    'volume': vol,
                    'mesh_volume': mesh_volume,
                    'intersection_pct': intersection_pct,
                    'ref_volume': ref_vol,
                    'diff_vs_ref_pct': diff_vs_ref
                })

        # Print progress
        print(f"    Completed {len(distances) * n_runs} experiments")

    # =========================================================================
    # Save Results
    # =========================================================================

    df = pd.DataFrame(results)

    # Save to CSV
    output_path = os.path.join(script_dir, "exp_mesh_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # =========================================================================
    # Summary Statistics
    # =========================================================================

    print("\n" + "=" * 60)
    print("Summary Statistics (averaged over all runs and distances)")
    print("=" * 60)

    summary = df.groupby('method').agg({
        'time_ms': ['mean', 'std'],
        'diff_vs_ref_pct': ['mean', 'std', 'max']
    }).round(4)

    print(summary)

    # Per-distance summary for one run
    print("\n" + "=" * 60)
    print("Volume by Distance (first run only)")
    print("=" * 60)

    first_run = df[df['run'] == 1].pivot(
        index='distance',
        columns='method',
        values='volume'
    ).round(6)

    print(first_run)

    # Intersection percentage
    print("\n" + "=" * 60)
    print("Intersection % by Distance (first run only)")
    print("=" * 60)

    first_run_pct = df[df['run'] == 1].pivot(
        index='distance',
        columns='method',
        values='intersection_pct'
    ).round(2)

    print(first_run_pct)

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
