"""Two spheres collision demo using viser visualization."""
import time
import torch
import kaolin as kal
import numpy as np
import pyvista as pv
import viser
import warp as wp

import pointint

# Config
RADIUS, DIST, HEIGHT = 0.1, 0.5, 0.8
NUM_SAMPLES, HANDLES, STEPS = 50_000, 5, 5000
SPEED = 1.0


def create_sphere(offset):
    """Create sphere mesh and sample interior points."""
    sphere = pv.Icosphere(radius=RADIUS, nsub=3)
    verts = torch.tensor(sphere.points, dtype=torch.float32, device="cuda")
    faces = torch.tensor(sphere.faces.reshape(-1, 4)[:, 1:4], dtype=torch.int64, device="cuda")
    verts[:, 0] += offset
    verts[:, 2] += HEIGHT

    # Sample interior
    bbox_min, bbox_max = verts.min(0).values, verts.max(0).values
    pts = torch.rand(NUM_SAMPLES, 3, device="cuda") * (bbox_max - bbox_min) + bbox_min
    inside = kal.ops.mesh.check_sign(verts.unsqueeze(0), faces, pts.unsqueeze(0), hash_resolution=512).squeeze()
    return kal.rep.SurfaceMesh(verts, faces), pts[inside]


def train_object(pts):
    """Train Simplicits object."""
    n = pts.shape[0]
    return pointint.simplicits.easy_api.SimplicitsObject.create_trained(
        pts,
        torch.full((n,), 1e5, device="cuda"),
        torch.full((n,), 0.45, device="cuda"),
        torch.full((n,), 500.0, device="cuda"),
        0.5, num_handles=HANDLES, training_num_steps=STEPS,
        training_lr_start=1e-3, training_lr_end=1e-3,
        training_le_coeff=1e-1, training_lo_coeff=1e6,
        training_log_every=STEPS // 10, normalize_for_training=True,
    )


def main():
    # Create spheres
    meshes, sim_objs, orig_verts = [], [], []
    for offset in [-DIST / 2, DIST / 2]:
        mesh, pts = create_sphere(offset)
        orig_verts.append(mesh.vertices.clone())
        sim_objs.append(train_object(pts))
        meshes.append(mesh)

    # Setup scene
    scene = pointint.simplicits.easy_api.SimplicitsScene()
    scene.max_newton_steps, scene.timestep, scene.direct_solve = 5, 0.03, True
    obj_ids = [scene.add_object(obj) for obj in sim_objs]
    scene.set_scene_gravity(acc_gravity=torch.zeros(3))
    scene.set_scene_floor(floor_height=0.0, floor_axis=2, floor_penalty=1000.0)

    # Set initial velocities (balls moving towards each other)
    z_dot = wp.to_torch(scene.sim_z_dot).clone()
    z_dot[wp.to_torch(scene.object_to_z_map[obj_ids[0]])[-9]] = SPEED
    z_dot[wp.to_torch(scene.object_to_z_map[obj_ids[1]])[-9]] = -SPEED
    wp.copy(src=wp.from_torch(z_dot, dtype=wp.float32), dest=scene.sim_z_dot)

    scene.enable_collisions(collision_particle_radius=0.1, collision_penalty=5000.0)

    # Viser visualization
    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0, 0, 0))

    colors = [(1.0, 0.5, 0.0), (0.0, 0.8, 0.8)]
    handles = []
    for i, mesh in enumerate(meshes):
        h = server.scene.add_mesh_simple(
            f"/sphere_{i}",
            vertices=mesh.vertices.cpu().numpy().astype(np.float32),
            faces=mesh.faces.cpu().numpy().astype(np.uint32),
            color=colors[i],
        )
        handles.append(h)

    print("Open http://localhost:8080 in browser")

    # Simulation loop
    for _ in range(200):
        scene.run_sim_step()
        for obj_id, handle, rest in zip(obj_ids, handles, orig_verts):
            handle.vertices = scene.get_object_deformed_pts(obj_id, rest).detach().cpu().numpy().astype(np.float32)
        time.sleep(0.033)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
