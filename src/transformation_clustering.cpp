/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Alexandru-Eugen Ichim
 *                      Willow Garage, Inc
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
// #include <pcl/features/ppf.h>
#include <pcl/common/transforms.h>

// #include <pcl/features/pfh.h>

#include "transformation_clustering.h"

//   // Cluster poses for filtering out outliers and obtaining more precise results
//   PoseWithVotesList results;
//   clusterPoses (voted_poses, results);

//   pcl::transformPointCloud (*input_, output, results.front ().pose);

//   transformation_ = final_transformation_ = results.front ().pose.matrix ();
//   converged_ = true;
// }

//////////////////////////////////////////////////////////////////////////////////////////////
void clusterPoses(PoseWithVotesList &poses, PoseWithVotesList &result,
                  float trans_thresh, float rot_thresh){
  BOOST_LOG_TRIVIAL(info) << "Clustering poses ...";
  // Start off by sorting the poses by the number of votes
  sort(poses.begin(), poses.end(), poseWithVotesCompareFunction);
  std::vector<PoseWithVotesList> clusters;
  std::vector<std::pair<size_t, unsigned int> > cluster_votes;
  for (size_t poses_i = 0; poses_i < poses.size(); ++ poses_i)
  {
    BOOST_LOG_TRIVIAL(info) <<
      boost::format("pose, nclusters: %lu, %lu") % poses_i % clusters.size();
    bool found_cluster = false;
    for (size_t clusters_i = 0; clusters_i < clusters.size(); ++ clusters_i)
    {
        if(posesWithinErrorBounds(poses[poses_i].pose, clusters[clusters_i].front().pose,
                                  trans_thresh, rot_thresh))
      {
        found_cluster = true;
        clusters[clusters_i].push_back (poses[poses_i]);
        cluster_votes[clusters_i].second += poses[poses_i].votes;
        break;
      }
    }

    if (found_cluster == false)
    {
      // Create a new cluster with the current pose
      PoseWithVotesList new_cluster;
      new_cluster.push_back (poses[poses_i]);
      clusters.push_back (new_cluster);
      cluster_votes.push_back (std::pair<size_t, unsigned int> (clusters.size () - 1, poses[poses_i].votes));
    }
 }
  // Sort clusters by total number of votes
  std::sort (cluster_votes.begin (), cluster_votes.end (), clusterVotesCompareFunction);
  // Compute pose average and put them in result vector
  /// @todo some kind of threshold for determining whether a cluster has enough votes or not...
  /// now just taking the first three clusters
  result.clear ();
  size_t max_clusters = (clusters.size () < 3) ? clusters.size () : 3;
  for (size_t cluster_i = 0; cluster_i < max_clusters; ++ cluster_i){
    BOOST_LOG_TRIVIAL(info) <<
      boost::format("Winning cluster has #votes: %d and #poses voted: %d.") %
      cluster_votes[cluster_i].second %
      clusters[cluster_votes[cluster_i].first].size();
    Eigen::Vector3f translation_average (0.0, 0.0, 0.0);
    Eigen::Vector4f rotation_average (0.0, 0.0, 0.0, 0.0);
    for (typename PoseWithVotesList::iterator v_it = clusters[cluster_votes[cluster_i].first].begin (); v_it != clusters[cluster_votes[cluster_i].first].end (); ++ v_it)
    {
      translation_average += v_it->pose.translation ();
      /// averaging rotations by just averaging the quaternions in 4D space - reference "On Averaging Rotations" by CLAUS GRAMKOW
      rotation_average += Eigen::Quaternionf (v_it->pose.rotation ()).coeffs ();
    }
    translation_average /= static_cast<float> (clusters[cluster_votes[cluster_i].first].size ());
    rotation_average /= static_cast<float> (clusters[cluster_votes[cluster_i].first].size ());

    Eigen::Affine3f transform_average;
    transform_average.translation ().matrix () = translation_average;
    transform_average.linear ().matrix () = Eigen::Quaternionf (rotation_average).normalized().toRotationMatrix ();
    result.push_back (PoseWithVotes (transform_average, cluster_votes[cluster_i].second));
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////
bool posesWithinErrorBounds(Eigen::Affine3f &pose1, Eigen::Affine3f &pose2,
                            float trans_thresh, float rot_thresh){
  float position_diff = (pose1.translation() - pose2.translation ()).norm();
  Eigen::AngleAxisf rotation_diff_mat((pose1.rotation().inverse().lazyProduct (pose2.rotation()).eval()));

  float rotation_diff_angle = fabsf (rotation_diff_mat.angle());

  if (position_diff < trans_thresh && rotation_diff_angle < rot_thresh)
    return true;
  else return false;
}


//////////////////////////////////////////////////////////////////////////////////////////////
bool poseWithVotesCompareFunction(const PoseWithVotes &a, const PoseWithVotes &b){
  return (a.votes > b.votes);
}


//////////////////////////////////////////////////////////////////////////////////////////////
bool clusterVotesCompareFunction(const std::pair<size_t, unsigned int> &a,
                            const std::pair<size_t, unsigned int> &b){
  return (a.second > b.second);
}
