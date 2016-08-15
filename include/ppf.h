#ifndef __PPF_H
#define __PPF_H

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


std::vector<std::vector<Eigen::Matrix4f>> ppf_registration(
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> scene_clouds,
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> model_clouds,
    std::vector<float> model_d_dists, unsigned int ref_point_downsample_factor,
    float vote_count_threshold, bool cpu_clustering,
    bool use_l1_norm, bool use_averaged_clusters,
    int devUse, float *model_weights);

#endif /* __PPF_H */
