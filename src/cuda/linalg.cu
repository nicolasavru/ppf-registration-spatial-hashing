#include "linalg.h"

#include <Eigen/Geometry>
#include <vector_types.h>

#include "vector_ops.h"
#include "kernel.h"

__host__ float2 ht_dist(Eigen::Matrix4f a, Eigen::Matrix4f b){
  float3 a_trans = {a(0,3), a(1,3), a(2,3)};
  float3 b_trans = {b(0,3), b(1,3), b(2,3)};
  float3 trans_diff = a_trans - b_trans;

  Eigen::AngleAxisf a_rot, b_rot;
  a_rot.fromRotationMatrix(a.block<3,3>(0,0));
  b_rot.fromRotationMatrix(b.block<3,3>(0,0));
  Eigen::AngleAxisf rotation_diff_mat(a_rot.inverse() * b_rot);
  float2 result = {norm(trans_diff), fabsf(rotation_diff_mat.angle())};
  return result;
};
