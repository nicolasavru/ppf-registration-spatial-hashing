#include "vector_ops.h"

#include <iostream>


std::ostream& operator<<(std::ostream& out, const float3& obj){
    out << obj.x << ", " << obj.y << ", " << obj.z;
    return out;
}

std::ostream& operator<<(std::ostream& out, const float4& obj){
    out << obj.x << ", " << obj.y << ", " << obj.z << ", " << obj.w;
    return out;
}

__host__ __device__ bool to_bool(float3 f){
    return f.x || f.y || f.z;
}

__host__ __device__ bool to_bool(float4 f){
    return f.x || f.y || f.z || f.w;
}

__host__ __device__ bool operator<(const float4 a, const float4 b){
    // compare 4 bytes at a time instead of 2
    ulong2 ul2a = *((ulong2 *) &a);
    ulong2 ul2b = *((ulong2 *) &b);

    if((ul2a.x < ul2b.x) ||
       ((ul2a.x == ul2b.x) && (ul2b.y < ul2b.y))){
        return true;
    }
    return false;
}

__host__ __device__ bool operator<(const float3 a, const float3 b){
    // compare 4 bytes at a time instead of 2
    ulong2 ul2a = *((ulong2 *) &a);
    ulong2 ul2b = *((ulong2 *) &b);

    if((ul2a.x < ul2b.x) ||
       ((ul2a.x == ul2b.x) && (a.z < b.z))){
        return true;
    }
    return false;
}

__host__ __device__ bool operator==(const float3 a, const float3 b){
    // compare 4 bytes at a time instead of 2
    // Is allocating two variables worth saving a comparison and a bitwise and?
    ulong2 ul2a = *((ulong2 *) &a);
    ulong2 ul2b = *((ulong2 *) &b);

    if((ul2a.x == ul2b.x) && (a.z == b.z)){
        return true;
    }
    return false;
}

__host__ __device__ bool operator==(const float4 a, const float4 b){
    // compare 4 bytes at a time instead of 2
    // Is allocating two variables worth saving a comparison and a bitwise and?
    ulong2 ul2a = *((ulong2 *) &a);
    ulong2 ul2b = *((ulong2 *) &b);

    if((ul2a.x == ul2b.x) && (ul2a.y == ul2b.y)){
        return true;
    }
    return false;
}

__host__ __device__ bool operator!=(const float3 a, const float3 b){
    return !(a == b);
}

__host__ __device__ bool operator!=(const float4 a, const float4 b){
    return !(a == b);
}

__host__ __device__ float3 operator*(float a, float3 v){
    float3 w = {a*v.x, a*v.y, a*v.z};
    return w;
}

__host__ __device__ float4 operator*(float a, float4 v){
    float4 w = {a*v.x, a*v.y, a*v.z, a*v.z};
    return w;
}

__host__ __device__ float3 operator+(float3 u, float3 v){
    float3 w = {u.x+v.x, u.y+v.y, u.z+v.z};
    return w;
}

__host__ __device__ float4 operator+(float4 u, float4 v){
    float4 w = {u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w};
    return w;
}

__host__ __device__ float3 operator-(float3 u, float3 v){
    float3 w = {u.x-v.x, u.y-v.y, u.z-v.z};
    return w;
}

__host__ __device__ float4 operator-(float4 u, float4 v){
    float4 w = {u.x-v.x, u.y-v.y, u.z-v.z, u.w-v.w};
    return w;
}
