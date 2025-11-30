import sys
import os
import numpy as np

# Add the build directory to the Python path
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cpp', 'build', 'Release'))
sys.path.append(build_path)

import pointint_core_python as pic

def main():
    print("\n--- PoIntInt Pybind11 Test Script ---")

    # 1. Load Lebedev grid and build K-Grid
    leb_file = os.path.join(os.path.dirname(__file__), '../data', 'lebedev', 'lebedev_047.txt')
    print(f"Loading Lebedev grid from: {leb_file}")
    leb = pic.load_lebedev_txt(leb_file)
    kgrid = pic.build_kgrid(leb.dirs, leb.weights, 32)
    print("Successfully built K-Grid.")

    # 2. Create Geometries (two unit cubes, one translated)
    V1 = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ], dtype=np.float64)
    F = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 7, 6], [2, 3, 7],
        [0, 4, 7], [0, 7, 3], [1, 6, 5], [1, 2, 6]
    ], dtype=np.int32)

    # Translate one cube for intersection
    V2 = V1 + np.array([0.5, 0.0, 0.0])

    geom1 = pic.make_triangle_mesh(V1, F)
    geom2 = pic.make_triangle_mesh(V2, F)
    print("Successfully created two triangle meshes.")

    # 3. Compute Intersection Volume
    volume = pic.compute_intersection_volume_cuda(geom1, geom2, kgrid)
    print(f"\nComputed intersection volume: {volume:.6f}")
    # For two unit cubes shifted by 0.5, the analytical volume is 0.5
    print(f"Analytical volume: 0.5")
    print(f"Absolute error: {abs(volume - 0.5):.6f}")

    # 4. Example with DoFs
    print("\n--- Testing with DoFs ---")
    affine_dof = pic.AffineDoF()
    
    # Identity DoFs for geom1
    dofs1 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    
    # DoFs for geom2 that translate it by (0.5, 0, 0)
    dofs2 = np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    # Use original (untranslated) vertex data for both geometries
    geom_orig = pic.make_triangle_mesh(V1, F)

    result = pic.compute_intersection_volume_cuda(
        geom_orig, geom_orig, affine_dof, affine_dof, dofs1, dofs2, kgrid, pic.ComputationFlags.VOLUME_ONLY
    )
    print(f"Intersection volume with DoFs: {result.volume:.6f}")
    print(f"Absolute error: {abs(result.volume - 0.5):.6f}")

if __name__ == "__main__":
    main()