#ifndef TRANSFORMATION_CLUSTERING_H
#define TRANSFORMATION_CLUSTERING_H

struct PoseWithVotes{
    PoseWithVotes(Eigen::Affine3f a_pose, unsigned int a_votes)
        : pose (a_pose),
          votes (a_votes)
    {}

    Eigen::Affine3f pose;
    unsigned int votes;
};
typedef std::vector<PoseWithVotes, Eigen::aligned_allocator<PoseWithVotes> > PoseWithVotesList;


//////////////////////////////////////////////////////////////////////////////////////////////
void clusterPoses(PoseWithVotesList &poses, PoseWithVotesList &result,
                  float trans_thresh, float rot_thresh);
bool posesWithinErrorBounds(Eigen::Affine3f &pose1, Eigen::Affine3f &pose2,
                            float trans_thresh, float rot_thresh);
bool poseWithVotesCompareFunction(const PoseWithVotes &a, const PoseWithVotes &b);
bool clusterVotesCompareFunction(const std::pair<size_t, unsigned int> &a,
                                 const std::pair<size_t, unsigned int> &b);

#endif /* TRANSFORMATION_CLUSTERING_H */
