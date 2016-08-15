#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <iostream>

#include <vector_types.h>

std::ostream& operator<<(std::ostream& out, const float3& obj);
std::ostream& operator<<(std::ostream& out, const float4& obj);

__host__ __device__ bool to_bool(float3 f);
__host__ __device__ bool to_bool(float4 f);

__host__ __device__ bool operator<(const float3 a, const float3 b);
__host__ __device__ bool operator<(const float4 a, const float4 b);

__host__ __device__ bool operator==(const float3 a, const float3 b);
__host__ __device__ bool operator==(const float4 a, const float4 b);

__host__ __device__ bool operator!=(const float3 a, const float3 b);
__host__ __device__ bool operator!=(const float4 a, const float4 b);

__host__ __device__ float3 operator*(float a, float3 v);
__host__ __device__ float4 operator*(float a, float4 v);

__host__ __device__ float3 operator+(float3 u, float3 v);
__host__ __device__ float4 operator+(float4 u, float4 v);

__host__ __device__ float3 operator-(float3 u, float3 v);
__host__ __device__ float4 operator-(float4 u, float4 v);

#endif /* VECTOR_OPS_H */
