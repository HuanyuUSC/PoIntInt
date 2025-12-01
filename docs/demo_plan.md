# Paper Outline
## Preliminary
around 8-9 pages: 1-2 intro+related works, 3-7 background+our method, 8-9 results+conclusion
## Intro

## Related works

## Background

## Our method

## Results

### Method Validation

**Table 1: To prove our method is robust with any type of representation**

| ID  | Scene        | Rep. A / Rep. B        | Size A / Size B           | Ground Truth Type    | Rel. Error [%] | 
|-----|--------------|------------------------|---------------------------|----------------------|----------------|
| S2  | Eng–Scan     | Mesh / Point cloud     | 320k tris / 1.5M pts      | Voxel 1024³          | 1.5            |
| S3  | Bun–Arma     | Mesh / Mesh            | 69.5k tris / 173.0k tris  | Voxel 512³–1536³     | 1.7            |
| S4  | Drg–Drg      | Mesh / Mesh            | 100k tris / 100k tris     | Voxel 2048³          | 1.9            |
| S6  | Chair–Table  | Point cloud / PC       | 1.2M pts / 0.9M pts       | Voxel 1024³          | 2.0            |
| S7  | CharA–CharB  | Mesh / Mesh            | 480k tris / 503k tris     | Voxel 1024³          | 1.3            |
| A1  | Sph–Sph (1)  | Analytic / Analytic    | –                         | Analytic             | 0.2            |
| A2  | Sph–Sph (2)  | Analytic / Analytic    | –                         | Analytic             | 2.6            |
| A3  | Box–Box      | Analytic / Analytic    | –                         | Analytic             | 0.4            |

(need 5 meshes and 5 point clouds)


**Table 2: To prove that our method is robust with different sample rates**

| Scene | VoxRes | resolution of PC   | Volume [Voxel / Ours] | Rel. Error [%] | Time [Voxel / Ours] (s) | Speedup |
|-------|--------|-----------|------------------------|----------------|-------------------------|---------|
| S1    | 1024³  | 50,000    | 0.1234 / 0.1227        | 0.57           | 12.4 / 0.031            | 400×    |
| S2    | 1024³  | 75,000    | 5.871  / 5.842         | 0.50           | 18.9 / 0.044            | 430×    |
| S3    | 1536³  | 100,000   | 0.0179 / 0.0175        | 2.23           | 51.3 / 0.083            | 618×    |
| S4    | 2048³  | 120,000   | 0.0043 / 0.0042        | 1.94           | 128.5 / 0.112           | 1,147×  |
| S6    | 1024³  | 80,000    | 0.692  / 0.678         | 2.02           | 24.7 / 0.059            | 419×    |
| S7    | 1024³  | 90,000    | 1.384  / 1.366         | 1.30           | 33.1 / 0.071            | 466×    |






### Application: Shape Similarity

#### Similar obejct clustering
image: 5 classes, each class 10 objects(mesh, pointcloud, gaussian)

#### Identify similar object
image: highly detailed object (3)

#### Real to sim geometry
image: poking example: sponge poking with weight, reconstruct 3DGS, get sponge mesh, match mesh with 3DGS, recover material




### Application: Collision

#### Better than other volume based method
- [Image-based Collision Detection and Response between Arbitrary Volumetric Objects](https://inria.hal.science/inria-00319399v1/document)
- [Volumetric Collision Detection for Derformable Objects](https://cg.informatik.uni-freiburg.de/publications/2003_TR395_collisionDetection.pdf)
- One voxelization paper
image/table? 

#### Why volume based method

- VS Penetration-free method: e.g. IPC; IPC treat point cloud as points.
image: two objects collision, expect IPC fail

- VS Distance based: e.g. primitive and sdf
image: symmetry object collision, expect primitive and sdf fail



## Conclusion

# Demo Video ourline


# Demo Brainstorm

## Shape similarity

Advantage: different resolution, different representation, fast

### Similar obejct clustering
video: 5 classes, each class 10 objects(mesh, pointcloud, gaussian)

### Identify similar object
video: highly detailed object (3)

### Real to sim geometry
video: poking example: sponge poking with weight, reconstruct 3DGS, get sponge mesh, match mesh with 3DGS, recover material


## Collision

Advantage: we can do this with volume based method

### Better than other volume based method
- [Image-based Collision Detection and Response between Arbitrary Volumetric Objects](https://inria.hal.science/inria-00319399v1/document)
- [Volumetric Collision Detection for Derformable Objects](https://cg.informatik.uni-freiburg.de/publications/2003_TR395_collisionDetection.pdf)
- One voxelization paper


### Why volume based method

- VS Penetration-free method: e.g. IPC; IPC treat point cloud as points.
Video: two objects collision, expect IPC fail
- VS Distance based: e.g. primitive and sdf

# Experiment

## Robust to sample rate

point cloud 0.5k,1k,5k,10k,50k calculate mesh and they match


