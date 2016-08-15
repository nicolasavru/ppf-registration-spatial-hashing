#ifndef __KERNEL_H
#define __KERNEL_H

#include <cstdlib>

#include <vector_types.h>
#include <math_constants.h>


//Launch configuration macros
#define BLOCK_SIZE 512
#define MAX_NBLOCKS 1024
//Algorithm macros
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define N_ANGLE 30
#define D_ANGLE0 ((2.0f*float(CUDART_PI_F))/float(N_ANGLE))
#define ROT_THRESH (2*D_ANGLE0)
#define SCORE_THRESHOLD 0

__host__ __device__ unsigned int high_32(unsigned long i);
__host__ __device__ unsigned int low_32(unsigned long i);
__host__ __device__ unsigned int hash(void *f, int n, unsigned int hash=2166136261);
__host__ __device__ __forceinline__ void zeroMat4(float T[4][4]);
__host__ __device__ float dot(float3 v1, float3 v2);
__host__ __device__ float dot(float4 v1, float4 v2);
__host__ __device__ float norm(float3 v);
__host__ __device__ float norm(float4 v);
__host__ __device__ float max(float3 v);
__host__ __device__ float max(float4 v);
__host__ __device__ float min(float3 v);
__host__ __device__ float min(float4 v);
__host__ __device__ float3 cross(float3 u, float3 v);
__host__ __device__ float quant_downf(float x, float y);
__host__ __device__ float4 disc_feature(float4 f, float d_dist, float d_angle);
__host__ __device__ float3 discretize(float3 f, float d_dist);
__host__ __device__ float4 compute_ppf(float3 p1, float3 n1, float3 p2, float3 n2);
__host__ __device__ float3 trans(float T[4][4]);
__host__ __device__ float4 hrotmat2quat(float T[4][4]);
__host__ __device__ void quat2hrotmat(float4 q, float T[4][4]);
__host__ __device__ void trans(float3 v, float T[4][4]);
__host__ __device__ void rotx(float theta, float T[4][4]);
__host__ __device__ void roty(float theta, float T[4][4]);
__host__ __device__ void rotz(float theta, float T[4][4]);
__host__ __device__ void mat4f_mul(const float A[4][4],
                                   const float B[4][4],
                                   float C[4][4]);
__host__ __device__ float3 mat3f_vmul(const float A[3][3], const float3 b);
__host__ __device__ float4 mat4f_vmul(const float A[4][4], const float4 b);
__host__ __device__ float4 homogenize(float3 v);
__host__ __device__ float3 dehomogenize(float4 v);
__host__ __device__ void invht(float T[4][4], float T_inv[4][4]);

__global__ void ppf_kernel(float3 *points, float3 *norms, float4 *out, int count,
                           int ref_point_downsample_factor, float d_dist);

__global__ void ppf_encode_kernel(float4 *ppfs, unsigned long *codes, int count);

__global__ void ppf_decode_kernel(unsigned long *codes, unsigned int *key2ppfMap,
                                  unsigned int *hashKeys, int count);

__global__ void vec_decode_kernel(float4 *vecs, unsigned int *key2VecMap,
                                  float3 *vecCodes, int count);

__global__ void ppf_hash_kernel(float4 *ppfs, unsigned int *codes, int count);

__global__ void ppf_vote_count_kernel(unsigned int *sceneKeys, std::size_t *sceneIndices,
                                      unsigned int *hashKeys, std::size_t *ppfCount,
                                      unsigned long *ppf_vote_counts, int count);

__global__ void ppf_vote_kernel(unsigned int *sceneKeys, std::size_t *sceneIndices,
                                unsigned int *hashKeys, std::size_t *ppfCount,
                                std::size_t *firstPPFIndex, std::size_t *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                unsigned long *ppf_vote_indices,
                                unsigned long *votes, int count, float d_dist);

__global__ void ppf_score_kernel(unsigned int *accumulator,
                                 unsigned int *maxidx,
                                 int n_angle, int threshold,
                                 unsigned int *scores,
                                 int count);

__global__ void trans_calc_kernel(unsigned int *uniqueSceneRefPts,
                                  unsigned int *maxModelAngleCodes,
                                  float3 *model_points, float3 *model_normals,
                                  float3 *scene_points, float3 *scene_normals,
                                  float *transforms, int count);

__global__ void trans_calc_kernel2(unsigned long *votes,
                                   float3 *model_points, float3 *model_normals,
                                   float3 *scene_points, float3 *scene_normals,
                                   float *transforms, int count);

__global__ void mat2transquat_kernel(float *transformations,
                                     float3 *transformation_trans,
                                     float4 *transformation_rots,
                                     int count);

__global__ void rot_clustering_kernel(float3 *translations,
                                      float4 *quaternions,
                                      float *vote_counts,
                                      unsigned int *adjacent_trans_hash,
                                      std::size_t *transIndices,
                                      unsigned int *transKeys,  std::size_t *transCount,
                                      std::size_t *firstTransIndex, std::size_t *key2transMap,
                                      // float3 *translations_out,
                                      // float4 *quaternions_out,
                                      float *vote_counts_out,
                                      int count, float trans_thresh,
                                      bool use_l1_norm, bool use_averaged_clusters);

__global__ void trans2idx_kernel(float3 *translations,
                                 unsigned int *trans_hash,
                                 unsigned int *adjacent_trans_hash,
                                 int count, float d_dist);

__global__ void vote_weight_kernel(unsigned long *votes, unsigned int *vote_counts,
                                   float *modelPointWeights, float *weightedVoteCounts,
                                   int count);

#endif /* __KERNEL_H */
