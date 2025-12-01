# Paper Outline
## Preliminary
around 8-9 pages: 1-2 intro+related works, 3-7 background+our method, 8-9 results+conclusion
## Intro

## Related works

## Background

## Our method

## Results

### Method Validation

### Application: Shape Similarity

### Application: Collision



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


