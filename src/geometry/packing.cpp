#include "geometry/packing.hpp"
#include "geometry/geometry.hpp"
#include <cmath>
#include <cassert>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace PoIntInt {

std::vector<TriPacked> pack_tris(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F){
  assert(F.cols()==3 && V.cols()==3);
  std::vector<TriPacked> T(F.rows());
  tbb::parallel_for(tbb::blocked_range<int>(0, F.rows()),
    [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i < r.end(); ++i) {
        int ia=F(i,0), ib=F(i,1), ic=F(i,2);
        Eigen::Vector3d a = V.row(ia), b=V.row(ib), c=V.row(ic);
        Eigen::Vector3d e1=b-a, e2=c-a;
        Eigen::Vector3d S = 0.5*(e1.cross(e2));
        T[i].a  = make_float3((float)a.x(), (float)a.y(), (float)a.z());
        T[i].e1 = make_float3((float)e1.x(), (float)e1.y(), (float)e1.z());
        T[i].e2 = make_float3((float)e2.x(), (float)e2.y(), (float)e2.z());
        T[i].S  = make_float3((float)S.x(), (float)S.y(), (float)S.z());
      }
    });
  return T;
}

std::vector<DiskPacked> pack_disks(
  const Eigen::MatrixXd& P,
  const Eigen::MatrixXd& N,
  const Eigen::VectorXd& radii_or_areas,
  bool is_radius)
{
  assert(P.cols()==3 && N.cols()==3 && P.rows()==N.rows() && P.rows()==radii_or_areas.rows());
  std::vector<DiskPacked> D(P.rows());
  const double PI = 3.14159265358979323846;
  
  tbb::parallel_for(tbb::blocked_range<int>(0, P.rows()),
    [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i < r.end(); ++i) {
        Eigen::Vector3d pos = P.row(i);
        Eigen::Vector3d normal = N.row(i);
        normal.normalize();  // Ensure unit normal
        
        double rho, area;
        if (is_radius) {
          rho = radii_or_areas(i);
          area = PI * rho * rho;
        } else {
          area = radii_or_areas(i);
          rho = std::sqrt(area / PI);
        }
        
        D[i].c = make_float3((float)pos.x(), (float)pos.y(), (float)pos.z());
        D[i].n = make_float3((float)normal.x(), (float)normal.y(), (float)normal.z());
        D[i].rho = (float)rho;
        D[i].area = (float)area;
      }
    });
  return D;
}

// Factory function: create Geometry from triangle mesh
Geometry make_triangle_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
  Geometry geom(GEOM_TRIANGLE);
  geom.tris = pack_tris(V, F);
  return geom;
}

// Factory function: create Geometry from point cloud
Geometry make_point_cloud(
  const Eigen::MatrixXd& P,
  const Eigen::MatrixXd& N,
  const Eigen::VectorXd& radii_or_areas,
  bool is_radius)
{
  Geometry geom(GEOM_DISK);
  geom.disks = pack_disks(P, N, radii_or_areas, is_radius);
  return geom;
}

// Pack Gaussian splats into CUDA-friendly format
std::vector<GaussianPacked> pack_gaussians(
  const Eigen::MatrixXd& P,
  const Eigen::MatrixXd& N,
  const Eigen::VectorXd& sigmas,
  const Eigen::VectorXd& weights)
{
  assert(P.cols() == 3 && N.cols() == 3);
  assert(P.rows() == N.rows() && P.rows() == sigmas.rows() && P.rows() == weights.rows());
  
  std::vector<GaussianPacked> G(P.rows());
  
  tbb::parallel_for(tbb::blocked_range<int>(0, P.rows()),
    [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i < r.end(); ++i) {
        Eigen::Vector3d pos = P.row(i);
        Eigen::Vector3d normal = N.row(i);
        normal.normalize();  // Ensure unit normal
        
        G[i].c = make_float3((float)pos.x(), (float)pos.y(), (float)pos.z());
        G[i].n = make_float3((float)normal.x(), (float)normal.y(), (float)normal.z());
        G[i].sigma = (float)sigmas(i);
        G[i].w = (float)weights(i);
      }
    });
  return G;
}

// Factory function: create Geometry from Gaussian splats
Geometry make_gaussian_splat(
  const Eigen::MatrixXd& P,
  const Eigen::MatrixXd& N,
  const Eigen::VectorXd& sigmas,
  const Eigen::VectorXd& weights)
{
  Geometry geom(GEOM_GAUSSIAN);
  geom.gaussians = pack_gaussians(P, N, sigmas, weights);
  return geom;
}

} // namespace PoIntInt

