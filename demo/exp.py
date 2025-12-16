"""
Sphere-Sphere Intersection Volume Experiment

This script benchmarks intersection volume computation between two spheres
using four different backends:
- Warp Fourier (GPU)
- Warp Direct (GPU)
- C++ CUDA (GPU)
- C++ CPU

Usage:
    python exp.py

Output:
    exp_results.csv - Detailed results for all experiments
"""

import os
import sys
import time
import numpy as np
import pandas as pd

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
from pointint.core.intersection.volume import intersection_volume_pointcloud
from pointint.core.intersection.volume_direct import intersection_volume_direct_pointcloud


def main():
    print("=" * 60)
    print("Sphere-Sphere Intersection Volume Experiment")
    print("=" * 60)

    # =========================================================================
    # Setup
    # =========================================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    lebedev_path = os.path.join(script_dir, "../data/lebedev/lebedev_029.txt")

    if not os.path.exists(lebedev_path):
        print(f"Error: Lebedev file not found at {lebedev_path}")
        sys.exit(1)

    print(f"Loading Lebedev grid from: {lebedev_path}")

    # C++ backend setup
    dirs_cpp, weights_cpp = pi.load_lebedev(lebedev_path)

    # Warp backend setup
    leb_dirs_warp, leb_w_warp = load_lebedev_warp(lebedev_path)

    # Create sphere point cloud (radius = 1)
    n_points = 1000
    P_sphere, N_sphere, R_sphere = pi.create_sphere_pointcloud(n_points)
    print(f"Created sphere point cloud with {n_points} points")

    # =========================================================================
    # Experiment Parameters
    # =========================================================================

    distances = np.arange(0, 2.1, 0.1)  # 0, 0.1, 0.2, ..., 2.0
    methods = ["Warp Fourier", "Warp Direct", "C++ CUDA", "C++ CPU"]
    n_runs = 5
    n_radial = 32

    print(f"\nExperiment parameters:")
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
    # Run Experiments
    # =========================================================================

    results = []
    total_exp = len(methods) * len(distances) * n_runs
    exp_count = 0

    print("\nRunning experiments...")

    for method in methods:
        print(f"\n  Method: {method}")

        for d in distances:
            # Translate second sphere
            P2 = P_sphere + np.array([d, 0.0, 0.0])

            # Ground truth (analytical formula)
            vol_gt = pi.sphere_sphere_intersection_volume(
                np.array([0.0, 0.0, 0.0]), 1.0,
                np.array([d, 0.0, 0.0]), 1.0
            )

            for run in range(1, n_runs + 1):
                exp_count += 1

                # Warmup for first run
                if run == 1 and d == 0:
                    # Do a warmup computation
                    if method == "Warp Fourier":
                        _ = intersection_volume_pointcloud(
                            P_sphere, N_sphere, R_sphere,
                            P2, N_sphere, R_sphere,
                            kgrid_warp
                        )
                    elif method == "Warp Direct":
                        _ = intersection_volume_direct_pointcloud(
                            P_sphere, N_sphere, R_sphere,
                            P2, N_sphere, R_sphere
                        )
                    elif method == "C++ CUDA":
                        geom1 = pi.make_point_cloud(P_sphere, N_sphere, R_sphere)
                        geom2 = pi.make_point_cloud(P2, N_sphere, R_sphere)
                        _ = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid_cpp)
                    else:  # C++ CPU
                        geom1 = pi.make_point_cloud(P_sphere, N_sphere, R_sphere)
                        geom2 = pi.make_point_cloud(P2, N_sphere, R_sphere)
                        _ = pi.compute_intersection_volume_cpu(geom1, geom2, kgrid_cpp)

                # Timed computation
                t0 = time.time()

                if method == "Warp Fourier":
                    vol = intersection_volume_pointcloud(
                        P_sphere, N_sphere, R_sphere,
                        P2, N_sphere, R_sphere,
                        kgrid_warp
                    )
                elif method == "Warp Direct":
                    vol = intersection_volume_direct_pointcloud(
                        P_sphere, N_sphere, R_sphere,
                        P2, N_sphere, R_sphere
                    )
                elif method == "C++ CUDA":
                    geom1 = pi.make_point_cloud(P_sphere, N_sphere, R_sphere)
                    geom2 = pi.make_point_cloud(P2, N_sphere, R_sphere)
                    vol = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid_cpp)
                else:  # C++ CPU
                    geom1 = pi.make_point_cloud(P_sphere, N_sphere, R_sphere)
                    geom2 = pi.make_point_cloud(P2, N_sphere, R_sphere)
                    vol = pi.compute_intersection_volume_cpu(geom1, geom2, kgrid_cpp)

                elapsed = time.time() - t0

                # Compute relative error
                if vol_gt > 1e-10:
                    rel_err = abs(vol - vol_gt) / vol_gt * 100
                else:
                    rel_err = 0.0 if abs(vol) < 1e-10 else float('inf')

                results.append({
                    'method': method,
                    'run': run,
                    'distance': round(d, 1),
                    'time_ms': elapsed * 1000,
                    'volume': vol,
                    'gt_volume': vol_gt,
                    'rel_error_pct': rel_err
                })

        # Print progress
        print(f"    Completed {len(distances) * n_runs} experiments")

    # =========================================================================
    # Save Results
    # =========================================================================

    df = pd.DataFrame(results)

    # Save to CSV
    output_path = os.path.join(script_dir, "exp_results.csv")
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
        'rel_error_pct': ['mean', 'std', 'max']
    }).round(4)

    print(summary)

    # Per-distance summary for one run
    print("\n" + "=" * 60)
    print("Error by Distance (first run only)")
    print("=" * 60)

    first_run = df[df['run'] == 1].pivot(
        index='distance',
        columns='method',
        values='rel_error_pct'
    ).round(2)

    print(first_run)

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
