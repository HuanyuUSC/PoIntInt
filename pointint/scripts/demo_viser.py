"""Two spheres collision demo using viser visualization."""
import time
from pathlib import Path
import torch
import kaolin as kal
import numpy as np
import trimesh
import viser
import warp as wp

from pointint.core.simulator.simplicits.easy_api import SimplicitsObject, SimplicitsScene

CACHE_FILE = Path(__file__).parent / "cache" / "spheres.pt"

# Config
RADIUS, DIST, HEIGHT = 0.1, 0.5, 0.8
NUM_SAMPLES, HANDLES, STEPS = 50_000, 5, 5000
SPEED = 1.0


def create_sphere(offset):
    """Create sphere mesh and sample interior points."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=RADIUS)
    verts = torch.tensor(sphere.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(sphere.faces, dtype=torch.int64, device="cuda")
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
    return SimplicitsObject.create_trained(
        pts,
        torch.full((n,), 1e5, device="cuda"),
        torch.full((n,), 0.45, device="cuda"),
        torch.full((n,), 500.0, device="cuda"),
        0.5, num_handles=HANDLES, training_num_steps=STEPS,
        training_lr_start=1e-3, training_lr_end=1e-3,
        training_le_coeff=1e-1, training_lo_coeff=1e6,
        training_log_every=STEPS // 10, normalize_for_training=True,
    )


def save_trained_data(meshes, sim_objs, orig_verts):
    """Save trained data to cache."""
    CACHE_FILE.parent.mkdir(exist_ok=True)
    data = {
        "meshes": [(m.vertices.cpu(), m.faces.cpu()) for m in meshes],
        "orig_verts": [v.cpu() for v in orig_verts],
        "sim_objs": [{
            "pts": o.pts.cpu(), "yms": o.yms.cpu(), "prs": o.prs.cpu(),
            "rhos": o.rhos.cpu(), "appx_vol": o.appx_vol.cpu(),
            "model_state": o.skinning_weight_function.state_dict(),
        } for o in sim_objs],
    }
    torch.save(data, CACHE_FILE)
    print(f"Saved to {CACHE_FILE}")


def load_trained_data():
    """Load trained data from cache."""
    from pointint.core.simulator.simplicits.easy_api import NormalizedSkinningWeightsFcn
    from pointint.core.simulator.simplicits.network import SimplicitsMLP

    data = torch.load(CACHE_FILE, weights_only=False)
    meshes, orig_verts, sim_objs = [], [], []
    for (v, f), ov, od in zip(data["meshes"], data["orig_verts"], data["sim_objs"]):
        meshes.append(kal.rep.SurfaceMesh(v.cuda(), f.cuda()))
        orig_verts.append(ov.cuda())
        # Reconstruct skinning weight function
        model = SimplicitsMLP(3, 64, HANDLES, 6).cuda()
        swf = NormalizedSkinningWeightsFcn(model, od["pts"].min(0).values.cuda(), od["pts"].max(0).values.cuda())
        swf.load_state_dict(od["model_state"])
        sim_objs.append(SimplicitsObject(
            od["pts"].cuda(), od["yms"].cuda(), od["prs"].cuda(),
            od["rhos"].cuda(), od["appx_vol"].cuda(), swf
        ))
    print(f"Loaded from {CACHE_FILE}")
    return meshes, sim_objs, orig_verts


def main():
    # Create or load spheres
    if CACHE_FILE.exists():
        meshes, sim_objs, orig_verts = load_trained_data()
    else:
        print("Training spheres (first run)...")
        meshes, sim_objs, orig_verts = [], [], []
        for offset in [-DIST / 2, DIST / 2]:
            mesh, pts = create_sphere(offset)
            orig_verts.append(mesh.vertices.clone())
            sim_objs.append(train_object(pts))
            meshes.append(mesh)
        save_trained_data(meshes, sim_objs, orig_verts)

    # Setup scene
    print("Setting up scene...")
    scene = SimplicitsScene()
    scene.max_newton_steps, scene.timestep, scene.direct_solve = 5, 0.03, True
    obj_ids = [scene.add_object(obj) for obj in sim_objs]
    scene.set_scene_gravity(acc_gravity=torch.zeros(3))
    scene.set_scene_floor(floor_height=0.0, floor_axis=2, floor_penalty=1000.0)

    # Set initial velocities (balls moving towards each other)
    z_dot = wp.to_torch(scene.sim_z_dot).clone()
    z_dot[wp.to_torch(scene.object_to_z_map[obj_ids[0]])[-9]] = SPEED
    z_dot[wp.to_torch(scene.object_to_z_map[obj_ids[1]])[-9]] = -SPEED
    wp.copy(src=wp.from_torch(z_dot, dtype=wp.float32), dest=scene.sim_z_dot)

    scene.enable_collisions(collision_particle_radius=0.01, collision_penalty=5000.0)

    # Viser visualization
    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0, 0, 0))

    colors = [(1.0, 0.5, 0.0), (0.0, 0.8, 0.8)]
    mesh_handles = []
    for i, mesh in enumerate(meshes):
        h = server.scene.add_mesh_simple(
            f"/sphere_{i}",
            vertices=mesh.vertices.cpu().numpy().astype(np.float32),
            faces=mesh.faces.cpu().numpy().astype(np.uint32),
            color=colors[i],
        )
        mesh_handles.append(h)

    # Simulation state
    sim_running = False

    # GUI controls
    with server.gui.add_folder("Simulation"):
        start_button = server.gui.add_button("Start Collision")
        stop_button = server.gui.add_button("Stop")
        reset_button = server.gui.add_button("Reset")

    @start_button.on_click
    def _(_):
        nonlocal sim_running
        sim_running = True
        print("Simulation started")

    @stop_button.on_click
    def _(_):
        nonlocal sim_running
        sim_running = False
        print("Simulation stopped")

    @reset_button.on_click
    def _(_):
        nonlocal sim_running
        sim_running = False
        # Reset scene state
        scene.reset()
        # Reset velocities
        z_dot = wp.to_torch(scene.sim_z_dot).clone()
        z_dot[wp.to_torch(scene.object_to_z_map[obj_ids[0]])[-9]] = SPEED
        z_dot[wp.to_torch(scene.object_to_z_map[obj_ids[1]])[-9]] = -SPEED
        wp.copy(src=wp.from_torch(z_dot, dtype=wp.float32), dest=scene.sim_z_dot)
        # Update visualization
        for obj_id, handle, rest in zip(obj_ids, mesh_handles, orig_verts):
            handle.vertices = scene.get_object_deformed_pts(obj_id, rest).detach().cpu().numpy().astype(np.float32)
        print("Simulation reset")

    print("Open http://localhost:8080 in browser")
    print("Click 'Start Collision' button to begin simulation")

    # Main loop
    while True:
        if sim_running:
            scene.run_sim_step()
            for obj_id, handle, rest in zip(obj_ids, mesh_handles, orig_verts):
                handle.vertices = scene.get_object_deformed_pts(obj_id, rest).detach().cpu().numpy().astype(np.float32)
        time.sleep(0.033)


if __name__ == "__main__":
    main()
