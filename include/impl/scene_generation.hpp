#ifndef SCENE_GENERATION_H
#define SCENE_GENERATION_H

#include <cmath>
#include <cstdlib>
#include <ctime>

#include <Eigen/Core>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector_types.h>

#include "ppf.h"
#include "vector_ops.h"

inline double randd(unsigned int seed=0){
    return double(rand()) / RAND_MAX;
}

inline float3 RandomTranslation(){
    float3 t;
    t.x = randd();
    t.y = randd();
    t.z = randd();
    return t;
}

// Uniform Random Rotation
// Ken Shoemake
// Graphics Gems III, pg 124-132
inline float4 RandomRotation(){
    double x0 = randd();
    double x1 = randd();
    double x2 = randd();
    double th1 = 2*M_PI*x1;
    double th2 = 2*M_PI*x2;
    double s1 = sin(th1);
    double c1 = cos(th1);
    double s2 = sin(th2);
    double c2 = cos(th2);
    double r1 = sqrt(1-x0);
    double r2 = sqrt(x0);
    float4 q;
    q.x = s1*r1;
    q.y = c1*r1;
    q.z = s2*r2;
    q.w = c2*r2;
    return q;
}

// Apply rotation and translation (in that order) to model and append model to scene.
template <typename Point>
Eigen::Matrix4f GenerateSceneWithModel(typename pcl::PointCloud<Point>& model,
                                       typename pcl::PointCloud<Point>& scene,
                                       float3& translation, float4& rotation,
                                       typename pcl::PointCloud<Point>& new_scene){
    if(!to_bool(translation)){
        translation = RandomTranslation();
    }

    if(!to_bool(rotation)){
        rotation = RandomRotation();
    }

    Point model_centroid, scene_centroid;
    pcl::computeCentroid(model, model_centroid);
    pcl::computeCentroid(scene, scene_centroid);

    Eigen::Affine3f t =
        Eigen::Translation3f(scene_centroid.x, scene_centroid.y, scene_centroid.z) *
        Eigen::Translation3f(translation.x, translation.y, translation.z) *
        Eigen::Quaternionf(rotation.x, rotation.y, rotation.z, rotation.w) *
        Eigen::Translation3f(-1*model_centroid.x, -1*model_centroid.y, -1*model_centroid.z);

    Eigen::Matrix4f T = t.matrix();
    // cout << "generated T:" << T << endl;

    pcl::PointCloud<Point> transformed_model = pcl::PointCloud<Point>();
    pcl::transformPointCloudWithNormals(model, transformed_model, T);

    new_scene += scene;
    new_scene += transformed_model;
    return T;
}

template <typename Point>
void CenterScene(typename pcl::PointCloud<Point>& scene){
    // TODO: check return value
    Point centroid;
    pcl::computeCentroid(scene, centroid);

    Eigen::Affine3f t;
    // Having the scene be in a different octant breaks things.
    t = Eigen::Translation3f(-1*centroid.x+1, -1*centroid.y+1, -1*centroid.z+1);
    Eigen::Matrix4f T = t.matrix();

    pcl::PointCloud<Point> transformed_scene = pcl::PointCloud<Point>();
    pcl::transformPointCloudWithNormals(scene, transformed_scene, T);
    // is this a memory leak?
    scene = transformed_scene;
}


#endif /* SCENE_GENERATION_H */
