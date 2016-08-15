#include "scene.h"

#include <cstdio>

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <cuda.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/binary_search.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector_types.h>

#include "impl/util.hpp"
#include "kernel.h"


Scene::Scene(){}

// ref_point_downsample_factor defaults to 1 in scene.h
Scene::Scene(pcl::PointCloud<pcl::PointNormal> *cloud_ptr, float d_dist,
             unsigned int ref_point_downsample_factor){
    BOOST_LOG_TRIVIAL(debug) << boost::format("foo0: %d") % cloud_ptr->size();
    this->cloud_ptr = cloud_ptr;
    thrust::host_vector<float3> *points =
        new thrust::host_vector<float3>(cloud_ptr->size());
    thrust::host_vector<float3> *normals =
        new thrust::host_vector<float3>(cloud_ptr->size());

    for(int i = 0; i < cloud_ptr->size(); i++){
        (*points)[i].x = (*cloud_ptr)[i].x;
        (*points)[i].y = (*cloud_ptr)[i].y;
        (*points)[i].z = (*cloud_ptr)[i].z;
        (*normals)[i].x = (*cloud_ptr)[i].normal_x;
        (*normals)[i].y = (*cloud_ptr)[i].normal_y;
        (*normals)[i].z = (*cloud_ptr)[i].normal_z;
    }

    this->d_dist = d_dist;

    this->initPPFs(points, normals, cloud_ptr->size(), d_dist,
                   ref_point_downsample_factor);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    this->hashKeys = new thrust::device_vector<unsigned int>(this->modelPPFs->size());

    int blocks = std::min(((int)(this->modelPPFs->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_hash_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                           RAW_PTR(this->hashKeys),
                                           this->modelPPFs->size());
}

Scene::~Scene(){
    delete this->modelPoints;
    delete this->modelNormals;
    delete this->modelPPFs;
    // delete this->hashKeys;
}

void Scene::initPPFs(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n,
                     float d_dist, unsigned int ref_point_downsample_factor){
    this->n = n;
    // check if these are used later or can be discarded after this function
    this->modelPoints = new thrust::device_vector<float3>(*points);
    this->modelNormals = new thrust::device_vector<float3>(*normals);
    this->modelPPFs = new thrust::device_vector<float4>(n*n);

#ifdef DEBUG
    BOOST_LOG_TRIVIAL(debug) << boost::format("n: %d") % n;
    // start cuda timer
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
#endif

    // This will crash if n = 0;
    int blocks = std::min(((int)(this->n + BLOCK_SIZE) - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // ppf_kernel computes ppfs and descritizes them, but does *not* hash them
    // hashing is done by ppf_hash_kernel, called only for model, not scene (model.cu:46)
    ppf_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPoints),
                                      RAW_PTR(this->modelNormals),
                                      RAW_PTR(this->modelPPFs),
                                      n, ref_point_downsample_factor, this->d_dist);

#ifdef DEBUG
    // end cuda timer
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    BOOST_LOG_TRIVIAL(info) <<
        boost::format("Time to generate PPFs: %3.1f ms") % elapsedTime;
#endif
}

int Scene::numPoints(){
    return this->n;
}

thrust::device_vector<float3> *Scene::getModelPoints(){
    return this->modelPoints;
}

thrust::device_vector<float3> *Scene::getModelNormals(){
    return this->modelNormals;
}
thrust::device_vector<float4> *Scene::getModelPPFs(){
    return this->modelPPFs;
}

thrust::device_vector<unsigned int>* Scene::getHashKeys(){
    return this->hashKeys;
}
