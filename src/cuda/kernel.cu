#include "kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector_types.h>

#include "vector_ops.h"


__host__ __device__ unsigned int high_32(unsigned long i){
    return (unsigned int) (i >> 32);
}

__host__ __device__ unsigned int low_32(unsigned long i){
    return (unsigned int) (i & (-1ul >> 32));
}

// FNV-1a hash function
// http://programmers.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed
// hash defaults to 2166136261 (set in kernel.h)
__host__ __device__ unsigned int hash(void *f, int n, unsigned int hash){
    char *s = (char *) f;
    while(n--){
        hash ^= *s++;
        hash *= 16777619;
    }
    return hash;
}

__host__ __device__ __forceinline__ void zeroMat4(float T[4][4]){
    T[0][0] = 0;
    T[0][1] = 0;
    T[0][2] = 0;
    T[0][3] = 0;
    T[1][0] = 0;
    T[1][1] = 0;
    T[1][2] = 0;
    T[1][3] = 0;
    T[2][0] = 0;
    T[2][1] = 0;
    T[2][2] = 0;
    T[2][3] = 0;
    T[3][0] = 0;
    T[3][1] = 0;
    T[3][2] = 0;
    T[3][3] = 0;
}

__host__ __device__ float dot(float3 v1, float3 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__host__ __device__ float dot(float4 v1, float4 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w;
}

__host__ __device__ float norm(float3 v){
    return sqrtf(dot(v, v));
}

__host__ __device__ float norm(float4 v){
    return sqrtf(dot(v, v));
}

__host__ __device__ float max(float3 v){
    return max(v.x, max(v.y, v.z));
}

__host__ __device__ float max(float4 v){
    return max(v.x, max(v.y, max(v.z, v.w)));
}

__host__ __device__ float min(float3 v){
    return min(v.x, min(v.y, v.z));
}

__host__ __device__ float min(float4 v){
    return min(v.x, min(v.y, min(v.z, v.w)));
}

__host__ __device__  float3 cross(float3 u, float3 v){
    float3 w = {u.y*v.z - u.z*v.y,
                u.z*v.x - u.x*v.z,
                u.x*v.y - u.y*v.z};
    return w;
}

__host__ __device__ float quant_downf(float x, float y){
    return x - fmodf(x, y);
}

__host__ __device__ float4 disc_feature(float4 f, float d_dist, float d_angle){
    f.x = quant_downf(f.x, d_dist);
    f.y = quant_downf(f.y, d_angle);
    f.z = quant_downf(f.z, d_angle);
    f.w = quant_downf(f.w, d_angle);
    return f;
}

__host__ __device__ float3 discretize(float3 f, float d_dist){
    f.x = quant_downf(f.x, d_dist);
    f.y = quant_downf(f.y, d_dist);
    f.z = quant_downf(f.z, d_dist);
    return f;
}

__host__ __device__ float4 compute_ppf(float3 p1, float3 n1, float3 p2, float3 n2){
    float3 d;
    d.x = p2.x - p1.x;
    d.y = p2.y - p1.y;
    d.z = p2.z - p1.z;

    float4 f;
    f.x = norm(d);
    f.y = acosf(dot(n1,d) / (norm(n1)*norm(d)));
    f.z = acosf(dot(n2,d) / (norm(n2)*norm(d)));
    f.w = acosf(dot(n1,n2) / (norm(n1)*norm(n2)));

    return f;
}

__host__ __device__ float3 trans(float T[4][4]){
    return make_float3(T[0][3], T[1][3], T[2][3]);
}

__host__ __device__ float4 hrotmat2quat(float T[4][4]){
    // https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    float t, r;
    float4 q;
    t = T[0][0] + T[1][1] + T[2][2];
    r = sqrt(1+t);
    q.x = 0.5*r;
    q.y = copysignf(0.5*sqrtf(1+T[0][0]-T[1][1]-T[2][2]), T[2][1]-T[1][2]);
    q.z = copysignf(0.5*sqrtf(1-T[0][0]+T[1][1]-T[2][2]), T[0][2]-T[2][0]);
    q.w = copysignf(0.5*sqrtf(1-T[0][0]-T[1][1]+T[2][2]), T[1][0]-T[0][1]);
    float n = sqrt(norm(q));
    q.x /= n;
    q.y /= n;
    q.z /= n;
    q.w /= n;
    return q;
}

__host__ __device__ void quat2hrotmat(float4 q, float T[4][4]){
    // https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    float n = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    float s = n == 0 ? 0 : 2 / n;
    float wx = s * q.x * q.y;
    float wy = s * q.x * q.z;
    float wz = s * q.x * q.w;
    float xx = s * q.y * q.y;
    float xy = s * q.y * q.z;
    float xz = s * q.y * q.w;
    float yy = s * q.z * q.z;
    float yz = s * q.z * q.w;
    float zz = s * q.w * q.w;
    T[0][0] = 1 - (yy + zz);
    T[0][1] = xy - wz;
    T[0][2] = xz + wy;
    T[1][0] = xy + wz;
    T[1][1] = 1 - (xx + zz);
    T[1][2] = yz - wx;
    T[2][0] = xz - wy;
    T[2][1] = yz + wx;
    T[2][2] = 1 - (xx + yy);
}

__host__ __device__ void trans(float3 v, float T[4][4]){
    zeroMat4(T);
    T[0][0] = 1;
    T[1][1] = 1;
    T[2][2] = 1;
    T[3][3] = 1;
    T[0][3] = v.x;
    T[1][3] = v.y;
    T[2][3] = v.z;
}

__host__ __device__ void rotx(float theta, float T[4][4]){
    zeroMat4(T);
    T[0][0] = 1;
    T[1][1] = cosf(theta);
    T[2][1] = sinf(theta);
    T[1][2] = -1*T[2][1];
    T[2][2] = T[1][1];
    T[3][3] = 1;
}

__host__ __device__ void roty(float theta, float T[4][4]){
    zeroMat4(T);
    T[0][0] = cosf(theta);
    T[0][2] = sinf(theta);
    T[1][1] = 1;
    T[2][0] = -1*T[0][2];
    T[2][2] = T[0][0];
    T[3][3] = 1;
}

__host__ __device__ void rotz(float theta, float T[4][4]){
    zeroMat4(T);
    T[0][0] = cosf(theta);
    T[1][0] = sinf(theta);
    T[0][1] = -1*T[1][0];
    T[1][1] = T[0][0];
    T[2][2] = 1;
    T[3][3] = 1;
}

__host__ __device__ void mat4f_mul(const float A[4][4],
                          const float B[4][4],
                          float C[4][4]){
    zeroMat4(C);

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 4; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

__host__ __device__ float3 mat3f_vmul(const float A[3][3], const float3 b){
    float3 *Af3 = (float3 *) A;
    float3 c;
    c.x = dot(Af3[0], b);
    c.y = dot(Af3[1], b);
    c.z = dot(Af3[2], b);
    return c;
}

__host__ __device__ float4 mat4f_vmul(const float A[4][4], const float4 b){
    float4 *Af4 = (float4 *) A;
    float4 c;
    c.x = dot(Af4[0], b);
    c.y = dot(Af4[1], b);
    c.z = dot(Af4[2], b);
    c.w = dot(Af4[3], b);
    return c;
}

__host__ __device__ float4 homogenize(float3 v){
    float4 w = {v.x, v.y, v.z, 1};
    return w;
}

__host__ __device__ float3 dehomogenize(float4 v){
    float3 w = {v.x, v.y, v.z};
    return w;
}

__host__ __device__ void invht(float T[4][4], float T_inv[4][4]){
    // T = [R t;0 1]; inv(T) = [R' -R'*t;0 1]; R*R' = I
    // R'
    T_inv[0][0] = T[0][0];
    T_inv[0][1] = T[1][0];
    T_inv[0][2] = T[2][0];

    T_inv[1][0] = T[0][1];
    T_inv[1][1] = T[1][1];
    T_inv[1][2] = T[2][1];

    T_inv[2][0] = T[0][2];
    T_inv[2][1] = T[1][2];
    T_inv[2][2] = T[2][2];

    // -R'
    float neg_Rtranspose[3][3];
    neg_Rtranspose[0][0] = -T_inv[0][0];
    neg_Rtranspose[0][1] = -T_inv[0][1];
    neg_Rtranspose[0][2] = -T_inv[0][2];

    neg_Rtranspose[1][0] = -T_inv[1][0];
    neg_Rtranspose[1][1] = -T_inv[1][1];
    neg_Rtranspose[1][2] = -T_inv[1][2];

    neg_Rtranspose[2][0] = -T_inv[2][0];
    neg_Rtranspose[2][1] = -T_inv[2][1];
    neg_Rtranspose[2][2] = -T_inv[2][2];

    // t
    float3 T_tmp;
    T_tmp.x = T[0][3];
    T_tmp.y = T[1][3];
    T_tmp.z = T[2][3];

    //-R*t
    float3 tmp = mat3f_vmul(neg_Rtranspose, T_tmp);
    T_inv[0][3] = tmp.x;
    T_inv[1][3] = tmp.y;
    T_inv[2][3] = tmp.z;

    T_inv[3][0] = 0;
    T_inv[3][1] = 0;
    T_inv[3][2] = 0;
    T_inv[3][3] = 1;
}


__device__ void trans_model_scene(float3 m_r, float3 n_r_m, float3 m_i,
                                  float3 s_r, float3 n_r_s, float3 s_i,
                                  float d_dist, unsigned int &alpha_idx){
    float transm[4][4], rot_x[4][4], rot_y[4][4], rot_z[4][4], T_tmp[4][4], T_m_g[4][4], T_s_g[4][4],
    T_tmp2[4][4], T[4][4];
    float4 n_tmp;

    // m_r = discretize(m_r, d_dist);
    m_r = -1*m_r;

    trans(m_r, transm);
    roty(atan2f(n_r_m.z, n_r_m.x), rot_y);
    n_tmp = homogenize(n_r_m);
    n_tmp = mat4f_vmul(rot_y, n_tmp);
    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_m_g);

    s_r = -1*s_r;
    trans(s_r, transm);
    roty(atan2f(n_r_s.z, n_r_s.x), rot_y);
    n_tmp = homogenize(n_r_s);
    n_tmp = mat4f_vmul(rot_y, n_tmp);
    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_s_g);


    n_tmp = homogenize(m_i);
    n_tmp = mat4f_vmul(T_m_g, n_tmp);
    float3 u = dehomogenize(n_tmp);

    n_tmp = homogenize(s_i);
    n_tmp = mat4f_vmul(T_s_g, n_tmp);
    float3 v = dehomogenize(n_tmp);

    u.x = 0;
    v.x = 0;
    float alpha = atan2f(cross(u, v).x, dot(u, v));
    alpha = quant_downf(alpha + CUDART_PI_F, D_ANGLE0);
    alpha_idx = (unsigned int) (lrintf(alpha/D_ANGLE0));
    rotx(alpha, rot_x);

    invht(T_s_g, T_tmp);
    mat4f_mul(T_tmp, rot_x, T_tmp2);
    mat4f_mul(T_tmp2, T_m_g, T);
    // T is T_ms
}


__device__ void compute_rot_angles(float3 n_r_m, float3 n_r_s,
                                   float *m_roty, float *m_rotz,
                                   float *s_roty, float *s_rotz){
    float rot_y[4][4];
    float4 n_tmp;

    *m_roty = atan2f(n_r_m.z, n_r_m.x);
    roty(*m_roty, rot_y);
    n_tmp = homogenize(n_r_m);
    n_tmp = mat4f_vmul(rot_y, n_tmp);
    *m_rotz = -1*atan2f(n_tmp.y, n_tmp.x);

    *s_roty = atan2f(n_r_s.z, n_r_s.x);
    roty(*s_roty, rot_y);
    n_tmp = homogenize(n_r_s);
    n_tmp = mat4f_vmul(rot_y, n_tmp);
    *s_rotz = -1*atan2f(n_tmp.y, n_tmp.x);
}


__device__ void compute_transforms(unsigned int angle_idx, float3 m_r,
                                   float m_roty, float m_rotz,
                                   float3 s_r, float s_roty,
                                   float s_rotz, float *T){
    float transm[4][4], rot_x[4][4], rot_y[4][4], rot_z[4][4], T_tmp[4][4],
          T_tmp2[4][4], T_m_g[4][4], T_s_g[4][4];

    float (*T_arr)[4] = (float (*)[4])T;

    // m_r = discretize(m_r, D_DIST);
    m_r = -1*m_r;

    trans(m_r, transm);
    roty(m_roty, rot_y);
    rotz(m_rotz, rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_m_g);

    s_r = -1*s_r;
    trans(s_r, transm);
    roty(s_roty, rot_y);
    rotz(s_rotz, rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_s_g);

    rotx(angle_idx*D_ANGLE0 - CUDART_PI_F, rot_x);
    invht(T_s_g, T_tmp);
    mat4f_mul(T_tmp, rot_x, T_tmp2);
    mat4f_mul(T_tmp2, T_m_g, T_arr);
}


__global__ void ppf_kernel(float3 *points, float3 *norms, float4 *out, int count,
                           int ref_point_downsample_factor, float d_dist){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    int bound;

    __shared__ float3 Spoints[BLOCK_SIZE];
    __shared__ float3 Snorms[BLOCK_SIZE];

    while(idx < count) {

        float3 thisPoint = points[idx];
        float3 thisNorm  = norms[idx];
        for(int i = 0; i < count; i+=BLOCK_SIZE){

            bound = MIN(count - i, BLOCK_SIZE);

            __syncthreads();
            if(ind < bound){
                Spoints[ind] = points[i+ind];
                Snorms[ind] = norms[i+ind];
            }
            __syncthreads();

            for(int j = 0; j < bound; j++) {

                if(idx % ref_point_downsample_factor != 0){
                    out[idx*count + j + i].x = CUDART_NAN_F;
                    continue;
                };

                // handle case of identical points in pair
                if((j + i - idx) == 0){
                    out[idx*count + j + i].x = CUDART_NAN_F;
                    continue;
                };
                // Spoints and Snorms are currently unreliable due to a race condition.
                // float4 ppf = compute_ppf(thisPoint, thisNorm, Spoints[j], Snorms[j]);
                float4 ppf = compute_ppf(thisPoint, thisNorm, points[i + j], norms[i + j]);
                // if(ppf.w < D_ANGLE0){
                //     out[idx*count + j + i].x = CUDART_NAN_F;
                //     continue;
                // }
                // MATLAB model_description.m:42
                out[idx*count + j + i] = disc_feature(ppf, d_dist, D_ANGLE0);
            }
        }
        //grid stride
        __syncthreads();
        idx += blockDim.x * gridDim.x;
    }
}


__global__ void ppf_hash_kernel(float4 *ppfs, unsigned int *codes, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    while(idx < count){
        if(isnan(ppfs[idx].x)){
            codes[idx] = 0;
        }
        else{
            codes[idx] = hash(ppfs+idx, sizeof(float4));
        }

        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}


__global__ void ppf_vote_count_kernel(unsigned int *sceneKeys, std::size_t *sceneIndices,
                                      unsigned int *hashKeys, std::size_t *ppfCount,
                                      unsigned long *ppf_vote_counts, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    while(idx < count){
        unsigned int thisSceneKey = sceneKeys[idx];
        unsigned int thisSceneIndex = sceneIndices[idx];
        if(thisSceneKey == 0 ||
           thisSceneKey != hashKeys[thisSceneIndex]){
            ppf_vote_counts[idx] = 0;
        }
        else{
            ppf_vote_counts[idx] = ppfCount[thisSceneIndex];
        }

        idx += blockDim.x * gridDim.x;
    }
}


__global__ void ppf_vote_kernel(unsigned int *sceneKeys, std::size_t *sceneIndices,
                                unsigned int *hashKeys, std::size_t *ppfCount,
                                std::size_t *firstPPFIndex, std::size_t *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                unsigned long *ppf_vote_indices,
                                unsigned long *votes_old, int count, float d_dist){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    unsigned int alpha_idx;

    while(idx < count){
        unsigned int thisSceneKey = sceneKeys[idx];
        unsigned int thisSceneIndex = sceneIndices[idx];
        if (thisSceneKey == 0 ||
            thisSceneKey != hashKeys[thisSceneIndex]){
            idx += blockDim.x * gridDim.x;
            continue;
        }
        unsigned int thisPPFCount = ppfCount[thisSceneIndex];
        unsigned int thisFirstPPFIndex = firstPPFIndex[thisSceneIndex];

        unsigned int scene_r_index = idx / sceneSize;
        unsigned int scene_i_index = idx - scene_r_index*sceneSize;
        float3 scene_r_point = scenePoints[scene_r_index];
        float3 scene_r_norm  = sceneNormals[scene_r_index];
        float3 scene_i_point = scenePoints[scene_i_index];

        unsigned int modelPPFIndex, model_r_index, model_i_index;
        float3 model_r_point, model_r_norm, model_i_point;
        for(int i = 0; i < thisPPFCount; i++){
            modelPPFIndex = key2ppfMap[thisFirstPPFIndex+i];
            model_r_index = modelPPFIndex / modelSize;
            model_i_index = modelPPFIndex - model_r_index*modelSize;

            model_r_point = modelPoints[model_r_index];
            model_r_norm  = modelNormals[model_r_index];
            model_i_point = modelPoints[model_i_index];

            trans_model_scene(model_r_point, model_r_norm, model_i_point,
                              scene_r_point, scene_r_norm, scene_i_point,
                              d_dist, alpha_idx);
            votes_old[ppf_vote_indices[idx] + i] =
                (((unsigned long) scene_r_index) << 32) | (model_r_index << 6) | (alpha_idx);
        }

        idx += blockDim.x * gridDim.x;
    }
}


// for each transvec in maxidx:
//   binary seach for transvec in vecs
//   nevermind, we already know that - the ith row of the
//   accumulator (and maxidx) corresponds to vec[i]

//   goto found index in firstVecIndex and vecCounts
//   Use that to find voteIndices in vec2VoteMap
//   find possible starting indices of blocks matching Model hashKeys

// TODO: increase thread work
__global__ void trans_calc_kernel(unsigned int *uniqueSceneRefPts,
                                  unsigned int *maxModelAngleCodes,
                                  float3 *model_points, float3 *model_normals,
                                  float3 *scene_points, float3 *scene_normals,
                                  float *transforms, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    unsigned int angle_idx, model_point_idx, scene_point_idx;

    unsigned int low6 = ((unsigned int) -1) >> 26;
    float m_roty, m_rotz, s_roty, s_rotz;

    while(idx < count){
        scene_point_idx = uniqueSceneRefPts[idx];
        model_point_idx = (maxModelAngleCodes[idx] >> 6);
        angle_idx = maxModelAngleCodes[idx] & low6;
        if(scene_point_idx == 0 && model_point_idx == 0 && angle_idx == 0){
            idx += blockDim.x * gridDim.x;
            continue;
        }

        compute_rot_angles(model_normals[model_point_idx],
                           scene_normals[scene_point_idx],
                           &m_roty, &m_rotz, &s_roty, &s_rotz);

        compute_transforms(angle_idx, model_points[model_point_idx],
                           m_roty, m_rotz,
                           scene_points[scene_point_idx],
                           s_roty, s_rotz, ((float *) transforms + idx*16));

        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void trans_calc_kernel2(unsigned long *votes,
                                   float3 *model_points, float3 *model_normals,
                                   float3 *scene_points, float3 *scene_normals,
                                   float *transforms, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    unsigned int angle_idx, model_point_idx, scene_point_idx, modelanglecode;
    unsigned long vote;

    unsigned int low6 = ((unsigned int) -1) >> 26;
    // unsigned long hi32 = ((unsigned long) -1) << 32;
    // unsigned long model_point_mask = ((unsigned long) -1) ^ hi32 ^ low6;
    float m_roty, m_rotz, s_roty, s_rotz;

    while(idx < count){
        vote = votes[idx];
        scene_point_idx = high_32(vote);
        modelanglecode = low_32(vote);
        model_point_idx = (modelanglecode >> 6);
        angle_idx = modelanglecode & low6;
        if(scene_point_idx == 0 && model_point_idx == 0 && angle_idx == 0){
            idx += blockDim.x * gridDim.x;
            continue;
        }

        compute_rot_angles(model_normals[model_point_idx],
                           scene_normals[scene_point_idx],
                           &m_roty, &m_rotz, &s_roty, &s_rotz);

        compute_transforms(angle_idx, model_points[model_point_idx],
                           m_roty, m_rotz,
                           scene_points[scene_point_idx],
                           s_roty, s_rotz, ((float *) transforms + idx*16));

        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void mat2transquat_kernel(float *transformations,
                                     float3 *transformation_trans,
                                     float4 *transformation_rots,
                                     int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    while(idx < count){
        transformation_trans[idx] = trans((float (*)[4]) (transformations + idx*16));
        transformation_rots[idx] = hrotmat2quat((float (*)[4]) (transformations + idx*16));
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void trans2idx_kernel(float3 *translations,
                                 unsigned int *trans_hash,
                                 unsigned int *adjacent_trans_hash,
                                 int count, float d_dist){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    int3 norm_disc_trans;
    int3 adjacent_norm_disc_trans;

    while(idx < count){
        float3 disc_trans = discretize(translations[idx], d_dist);
        norm_disc_trans.x = (int) (disc_trans.x / d_dist);
        norm_disc_trans.y = (int) (disc_trans.y / d_dist);
        norm_disc_trans.z = (int) (disc_trans.z / d_dist);
        // Maybe replace hash with bitpacking an unsigned long.
        trans_hash[idx] = hash(&norm_disc_trans, sizeof(int3));
        for(int i = -1, c = 0; i < 2; i++){
            for(int j = -1; j < 2; j++){
                for(int k = -1; k < 2; k++, c++){
                    // THIS IS WRONG, BUT IT MAKES IT WORK
                    // if(i == j == k == 0){
                    if(i == 0 && j == 0 && k == 0){
                        adjacent_trans_hash[27*idx+c] = 0;
                        continue;
                    }
                    adjacent_norm_disc_trans.x = norm_disc_trans.x + i;
                    adjacent_norm_disc_trans.y = norm_disc_trans.y + j;
                    adjacent_norm_disc_trans.z = norm_disc_trans.z + k;
                    adjacent_trans_hash[27*idx+c] = hash(&adjacent_norm_disc_trans, sizeof(int3));
                }
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}


__global__ void rot_clustering_kernel(float3 *translations,
                                      float4 *quaternions,
                                      float *vote_counts,
                                      unsigned int *adjacent_trans_hash,
                                      std::size_t *transIndices,
                                      unsigned int *transKeys,  std::size_t *transCount,
                                      std::size_t *firstTransIndex, std::size_t *key2transMap,
                                      float *vote_counts_out,
                                      int count, float trans_thresh,
                                      bool use_l1_norm, bool use_averaged_clusters){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    float rot_thresh_sq = ROT_THRESH*ROT_THRESH;

    while(idx < count){
        float3 thisTrans = translations[idx];
        float4 thisQuat  = quaternions[idx];
        float vote_count_out = 1;
        float3 thisTrans_out = thisTrans;
        for(int adjacent_block = 0; adjacent_block < 27; adjacent_block++){
            unsigned int thisAdjacentHash = adjacent_trans_hash[27*idx+adjacent_block];
            unsigned int thisAdjacentHashIndex = transIndices[27*idx+adjacent_block];
            if(thisAdjacentHash == 0 ||
               thisAdjacentHash != transKeys[thisAdjacentHashIndex]){
                continue;
            }
            unsigned int thisTransCount = transCount[thisAdjacentHashIndex];
            unsigned int thisFirstTransIndex = firstTransIndex[thisAdjacentHashIndex];

            for(int j = 0; j < thisTransCount; j++){
                unsigned int thisTransIndex = key2transMap[thisFirstTransIndex+j];
                float thisVoteCount = vote_counts[thisTransIndex];
                float quatDiff =
                    fabsf(8*(1-dot(thisQuat, quaternions[thisTransIndex])));
                if(quatDiff < rot_thresh_sq){
                    if(!use_l1_norm){
                        float normDiffTrans =
                            norm(thisTrans - translations[thisTransIndex]);
                        if(!(normDiffTrans < trans_thresh)){
                            continue;
                        }
                    }
                    if(use_averaged_clusters){
                        thisTrans_out = vote_count_out * thisTrans_out;
                        thisTrans_out =
                            thisTrans_out + vote_counts[thisTransIndex]*translations[thisTransIndex];
                        thisTrans_out = (1/(vote_count_out + thisVoteCount)) * thisTrans_out;
                    }
                    vote_count_out += thisVoteCount;
                }
            }
        }
        vote_counts_out[idx] = vote_count_out;
        translations[idx] = thisTrans_out;

        // grid stride
        idx += blockDim.x * gridDim.x;
    }
}


__global__ void vote_weight_kernel(unsigned long *votes, unsigned int *vote_counts,
                                   float *modelPointVoteWeights, float *weightedVoteCounts,
                                   int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    while(idx < count){
        unsigned int modelanglecode = low_32(votes[idx]);
        unsigned int model_point_idx = (modelanglecode >> 6);
        weightedVoteCounts[idx] = modelPointVoteWeights[model_point_idx] * vote_counts[idx];

        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}
