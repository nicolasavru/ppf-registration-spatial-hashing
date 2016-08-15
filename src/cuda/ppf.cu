#include "ppf.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <cuda.h>
#include <Eigen/Core>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector_types.h>

#include "impl/util.hpp"
#include "kernel.h"
#include "model.h"


std::vector<std::vector<Eigen::Matrix4f>> ppf_registration(
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> scene_clouds,
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> model_clouds,
    std::vector<float> model_d_dists, unsigned int ref_point_downsample_factor,
    float vote_count_threshold, bool cpu_clustering,
    bool use_l1_norm, bool use_averaged_clusters,
    int devUse, float *model_weights){

    int numDevices;
    HANDLE_ERROR(cudaGetDeviceCount(&numDevices));
    BOOST_LOG_TRIVIAL(info) << boost::format("Found %d CUDA devices:") % numDevices;
    cudaDeviceProp prop;
    for(int i = 0; i < numDevices; i++){
        cudaGetDeviceProperties(&prop, i);
        BOOST_LOG_TRIVIAL(info) << boost::format("%d) name: %s") % i % prop.name;
    }
    HANDLE_ERROR(cudaSetDevice(std::min(numDevices-1, devUse)));
    int devNum;
    HANDLE_ERROR(cudaGetDevice(&devNum));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, devNum));
    BOOST_LOG_TRIVIAL(info) << boost::format("Using device %d, %s") % devNum % prop.name;

    // cuda setup
    int blocks = prop.multiProcessorCount;
    BOOST_LOG_TRIVIAL(debug) << boost::format("multiProcessorCount: %d") % blocks;

    std::vector<std::vector<Eigen::Matrix4f>> results;

    for(int i = 0; i < scene_clouds.size(); i++){
      pcl::PointCloud<pcl::PointNormal>::Ptr scene_cloud = scene_clouds[i];
        // build model description

        results.push_back(std::vector<Eigen::Matrix4f>());

        for(int j = 0; j < model_clouds.size(); j++){
            // The d_dist for the scene must match the d_dist for the model, so
            // we need to re-compute (or at least re-downsample, which is about
            // as expensive) the scene PPFs fpr each model.
            Scene *scene = new Scene(scene_cloud.get(), model_d_dists[j], ref_point_downsample_factor);
            pcl::PointCloud<pcl::PointNormal>::Ptr model_cloud = model_clouds[j];
            Model *model = new Model(model_cloud.get(), model_d_dists[j], vote_count_threshold,
                                     cpu_clustering, use_l1_norm, use_averaged_clusters);

            model->ppf_lookup(scene);

            Eigen::Matrix4f T;
            if(cpu_clustering){
                T = model->cpu_transformations[0].pose.matrix();
            }
            else{
                // TODO: copy only the first transformations instead of the entire vector.
                thrust::host_vector<float> transformations =
                    thrust::host_vector<float>(model->getTransformations());
                for(int r = 0; r < 4; r++){
                    for(int c = 0; c < 4; c++){
                        T(r,c) = transformations[model->max_idx*16 + r*4+c];
                    }
                }
                thrust::host_vector<float3> transformation_trans(*model->transformation_trans);
                thrust::host_vector<float4> transformation_rots(*model->transformation_rots);
                // quat2hrotmat(transformation_rots[model->max_idx], (float (*)[4]) transformations.data());
                T(0, 3) = transformation_trans[model->max_idx].x;
                T(1, 3) = transformation_trans[model->max_idx].y;
                T(2, 3) = transformation_trans[model->max_idx].z;
            }

            BOOST_LOG_TRIVIAL(info) << "Found transformation:\n" << T;
            results.back().push_back(T);
            delete model;
            delete scene;
        }
    }


    cudaDeviceReset();

    return results;
}
