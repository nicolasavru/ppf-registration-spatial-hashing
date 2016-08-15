#ifndef __SCENE_H
#define __SCENE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


class Scene {

    public:

        Scene();
        Scene(pcl::PointCloud<pcl::PointNormal> *cloud_ptr, float d_dist,
              unsigned int ref_point_downsample_factor=1);

        ~Scene();

        int numPoints();
        thrust::device_vector<float3> *getModelPoints();
        thrust::device_vector<float3> *getModelNormals();
        thrust::device_vector<float4> *getModelPPFs();
        thrust::device_vector<unsigned int> *getHashKeys();

        pcl::PointCloud<pcl::PointNormal> *cloud_ptr;

    protected:

        // Number of PPF in the mode. I.e., number of elements in each of
        // the following arrays;
        unsigned long n;

        // Vector of model points
        thrust::device_vector<float3> *modelPoints;

        // Vector of model normals
        thrust::device_vector<float3> *modelNormals;

        // Vector of model point pair features
        thrust::device_vector<float4> *modelPPFs;

        // For a scene, hashKeys stores the hashes of all point pair features.
        // For a model, hashKeys is an array of all UNIQUE hashKeys. A binary search
        //   should be used to find the index of desired hash key.
        thrust::device_vector<unsigned int> *hashKeys;

    float d_dist;

    void initPPFs(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n,
                  float d_dist, unsigned int ref_point_downsample_factor=1);
};


#endif /* __SCENE_H */
