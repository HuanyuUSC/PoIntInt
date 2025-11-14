#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "compute_volume.hpp"
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"

using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;
using PoIntInt::GeometryType;
using PoIntInt::GEOM_TRIANGLE;
using PoIntInt::GEOM_DISK;
using PoIntInt::GEOM_GAUSSIAN;

// ===================== Kernel =====================
// Compute volume using divergence theorem: V = (1/3) ∫_S (x, y, z) · n dS
extern "C" __global__
void compute_volume_kernel(
  const TriPacked* __restrict__ tris, int NF,
  const DiskPacked* __restrict__ disks, int ND,
  const GaussianPacked* __restrict__ gaussians, int NG,
  int geom_type,
  double* out_volume
) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  
  // Each thread accumulates contributions from its assigned elements
  double local_sum = 0.0;
  
  if (geom_type == 0) {  // GEOM_TRIANGLE
    // For triangles: V = (1/3) ∫_T (x, y, z) · n dS
    // For a flat triangle, (x, y, z) · n is constant = center · n
    // where center = (a + b + c)/3 = a + (e1 + e2)/3
    // and n = S/|S|, area = |S|, where S = 0.5*(e1 × e2)
    // So: (center · n) * area = center · S
    // = (a + (e1 + e2)/3) · S = a·S + (e1 + e2)·S/3
    // Since S is perpendicular to the triangle plane (S = 0.5*(e1 × e2)),
    // we have e1·S = e2·S = 0, so: center · S = a·S
    if (NF > 0) {
      for (int i = tid + bid * blockDim.x; i < NF; i += blockDim.x * gridDim.x) {
        TriPacked t = tris[i];
        // Center = a + (e1 + e2)/3, but e1·S = e2·S = 0, so center · S = a·S
        double contrib = (double)t.a.x * (double)t.S.x + 
                         (double)t.a.y * (double)t.S.y + 
                         (double)t.a.z * (double)t.S.z;
        local_sum += contrib;
      }
    }
  } else if (geom_type == 1) {  // GEOM_DISK
    // For disks: contribution = (center · n) * area
    if (ND > 0) {
      for (int i = tid + bid * blockDim.x; i < ND; i += blockDim.x * gridDim.x) {
        DiskPacked d = disks[i];
        if (d.rho > 0.0f && d.area >= 0.0f) {
          double contrib = ((double)d.c.x * (double)d.n.x + 
                           (double)d.c.y * (double)d.n.y + 
                           (double)d.c.z * (double)d.n.z) * (double)d.area;
          local_sum += contrib;
        }
      }
    }
  } else if (geom_type == 2) {  // GEOM_GAUSSIAN
    // For Gaussian splats: contribution = (center · n) * weight
    if (NG > 0) {
      for (int i = tid + bid * blockDim.x; i < NG; i += blockDim.x * gridDim.x) {
        GaussianPacked g = gaussians[i];
        if (g.sigma > 0.0f && g.w >= 0.0f) {
          double contrib = ((double)g.c.x * (double)g.n.x + 
                           (double)g.c.y * (double)g.n.y + 
                           (double)g.c.z * (double)g.n.z) * (double)g.w;
          local_sum += contrib;
        }
      }
    }
  }
  
  // Tree reduction in shared memory
  __shared__ double sdata[256];
  sdata[tid] = local_sum;
  __syncthreads();
  
  // Tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < blockDim.x) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Write result
  if (tid == 0) {
    atomicAdd(out_volume, sdata[0]);
  }
}

namespace PoIntInt {

double compute_volume_cuda(
  const Geometry& geom,
  int blockSize,
  bool enable_profiling)
{
  auto t_start = std::chrono::high_resolution_clock::now();
  
  int NF = (geom.type == GEOM_TRIANGLE) ? (int)geom.tris.size() : 0;
  int ND = (geom.type == GEOM_DISK) ? (int)geom.disks.size() : 0;
  int NG = (geom.type == GEOM_GAUSSIAN) ? (int)geom.gaussians.size() : 0;
  
  if (NF == 0 && ND == 0 && NG == 0) {
    return 0.0;
  }
  
  TriPacked* d_tris = nullptr;
  DiskPacked* d_disks = nullptr;
  GaussianPacked* d_gaussians = nullptr;
  double* d_volume = nullptr;
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      if (d_tris) cudaFree(d_tris); \
      if (d_disks) cudaFree(d_disks); \
      if (d_gaussians) cudaFree(d_gaussians); \
      if (d_volume) cudaFree(d_volume); \
      return 0.0; \
    } \
  } while(0)
  
  auto t_malloc_start = std::chrono::high_resolution_clock::now();
  
  // Allocate memory (always allocate at least 1 element)
  if (NF > 0) {
    CUDA_CHECK(cudaMalloc(&d_tris, NF * sizeof(TriPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_tris, sizeof(TriPacked)));
  }
  if (ND > 0) {
    CUDA_CHECK(cudaMalloc(&d_disks, ND * sizeof(DiskPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_disks, sizeof(DiskPacked)));
  }
  if (NG > 0) {
    CUDA_CHECK(cudaMalloc(&d_gaussians, NG * sizeof(GaussianPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_gaussians, sizeof(GaussianPacked)));
  }
  CUDA_CHECK(cudaMalloc(&d_volume, sizeof(double)));
  
  auto t_malloc_end = std::chrono::high_resolution_clock::now();
  auto t_memcpy_start = std::chrono::high_resolution_clock::now();
  
  // Copy data
  if (NF > 0) {
    CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), NF * sizeof(TriPacked), cudaMemcpyHostToDevice));
  }
  if (ND > 0) {
    CUDA_CHECK(cudaMemcpy(d_disks, geom.disks.data(), ND * sizeof(DiskPacked), cudaMemcpyHostToDevice));
  }
  if (NG > 0) {
    CUDA_CHECK(cudaMemcpy(d_gaussians, geom.gaussians.data(), NG * sizeof(GaussianPacked), cudaMemcpyHostToDevice));
  }
  
  double zero = 0.0;
  CUDA_CHECK(cudaMemcpy(d_volume, &zero, sizeof(double), cudaMemcpyHostToDevice));
  
  auto t_memcpy_end = std::chrono::high_resolution_clock::now();
  auto t_kernel_start = std::chrono::high_resolution_clock::now();
  
  // Launch kernel
  int num_elements = (NF > 0) ? NF : ((ND > 0) ? ND : NG);
  int num_blocks = (num_elements + blockSize - 1) / blockSize;
  dim3 grid(num_blocks);
  dim3 block(blockSize);
  
  compute_volume_kernel<<<grid, block>>>(
    d_tris, NF, d_disks, ND, d_gaussians, NG, (int)geom.type, d_volume);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  auto t_kernel_end = std::chrono::high_resolution_clock::now();
  auto t_result_start = std::chrono::high_resolution_clock::now();
  
  // Copy result back
  double volume_sum = 0.0;
  CUDA_CHECK(cudaMemcpy(&volume_sum, d_volume, sizeof(double), cudaMemcpyDeviceToHost));
  
  auto t_result_end = std::chrono::high_resolution_clock::now();
  
  // Cleanup
  cudaFree(d_tris);
  cudaFree(d_disks);
  cudaFree(d_gaussians);
  cudaFree(d_volume);
  
  #undef CUDA_CHECK
  
  // Volume = (1/3) * sum
  double volume = volume_sum / 3.0;
  
  auto t_end = std::chrono::high_resolution_clock::now();
  
  if (enable_profiling) {
    auto malloc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_malloc_end - t_malloc_start).count() / 1000.0;
    auto memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(t_memcpy_end - t_memcpy_start).count() / 1000.0;
    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_end - t_kernel_start).count() / 1000.0;
    auto result_time = std::chrono::duration_cast<std::chrono::microseconds>(t_result_end - t_result_start).count() / 1000.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CUDA Volume Computation (Divergence Theorem) ===" << std::endl;
    if (geom.type == GEOM_TRIANGLE) {
      std::cout << "Geometry: Triangle mesh (" << NF << " faces)" << std::endl;
    } else if (geom.type == GEOM_DISK) {
      std::cout << "Geometry: Point cloud (" << ND << " disks)" << std::endl;
    } else {
      std::cout << "Geometry: Gaussian splats (" << NG << " gaussians)" << std::endl;
    }
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Memory allocation: " << std::setw(8) << malloc_time << " ms" << std::endl;
    std::cout << "  Memory copy (H->D): " << std::setw(8) << memcpy_time << " ms" << std::endl;
    std::cout << "  Kernel execution:   " << std::setw(8) << kernel_time << " ms" << std::endl;
    std::cout << "  Memory copy (D->H): " << std::setw(8) << result_time << " ms" << std::endl;
    std::cout << "  Total time:         " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  return volume;
}

double compute_volume_cpu(const Geometry& geom) {
  double volume = 0.0;
  
  if (geom.type == GEOM_TRIANGLE) {
    // For triangles: V = (1/3) * Σ_triangles (a · S)
    // where a is the first vertex and S is the oriented area vector
    for (const auto& tri : geom.tris) {
      double contrib = (double)tri.a.x * (double)tri.S.x + 
                       (double)tri.a.y * (double)tri.S.y + 
                       (double)tri.a.z * (double)tri.S.z;
      volume += contrib;
    }
    volume /= 3.0;
  } else if (geom.type == GEOM_DISK) {
    // For disks: V = (1/3) * Σ_disks ((c · n) * area)
    // where c is the center, n is the normal, and area is the disk area
    for (const auto& disk : geom.disks) {
      if (disk.rho > 0.0f && disk.area >= 0.0f) {
        double contrib = ((double)disk.c.x * (double)disk.n.x + 
                         (double)disk.c.y * (double)disk.n.y + 
                         (double)disk.c.z * (double)disk.n.z) * (double)disk.area;
        volume += contrib;
      }
    }
    volume /= 3.0;
  } else if (geom.type == GEOM_GAUSSIAN) {
    // For Gaussian splats: V = (1/3) * Σ_gaussians ((c · n) * w)
    // where c is the center, n is the normal, and w is the weight
    for (const auto& gauss : geom.gaussians) {
      if (gauss.sigma > 0.0f && gauss.w >= 0.0f) {
        double contrib = ((double)gauss.c.x * (double)gauss.n.x + 
                         (double)gauss.c.y * (double)gauss.n.y + 
                         (double)gauss.c.z * (double)gauss.n.z) * (double)gauss.w;
        volume += contrib;
      }
    }
    volume /= 3.0;
  }
  
  return volume;
}

} // namespace PoIntInt

