#include "model.h"

#include <algorithm>
#include <cstdio>
#include <iterator>

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <cuda.h>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector_types.h>

#include "impl/parallel_hash_array.hpp"
#include "impl/scene_generation.hpp"
#include "impl/util.hpp"
#include "kernel.h"
#include "linalg.h"
#include "transformation_clustering.h"


template <int value>
struct is_greaterthan{
    __host__ __device__ bool operator()(const unsigned int x){
        return x > value;
    }
};

struct float16 {
    float f[16];
};

Model::Model(pcl::PointCloud<pcl::PointNormal> *cloud, float d_dist,
             float vote_count_threshold, bool cpu_clustering,
             bool use_l1_norm, bool use_averaged_clusters){
    this->cloud_ptr = cloud;
    thrust::host_vector<float3> *points =
        new thrust::host_vector<float3>(cloud_ptr->size());
    thrust::host_vector<float3> *normals =
        new thrust::host_vector<float3>(cloud_ptr->size());

    for (int i = 0; i < cloud_ptr->size(); i++){
        (*points)[i].x = (*cloud_ptr)[i].x;
        (*points)[i].y = (*cloud_ptr)[i].y;
        (*points)[i].z = (*cloud_ptr)[i].z;
        (*normals)[i].x = (*cloud_ptr)[i].normal_x;
        (*normals)[i].y = (*cloud_ptr)[i].normal_y;
        (*normals)[i].z = (*cloud_ptr)[i].normal_z;
    }

    this->d_dist = d_dist;
    this->vote_count_threshold = vote_count_threshold;
    this->cpu_clustering = cpu_clustering;
    this->use_l1_norm = use_l1_norm;
    this->use_averaged_clusters = use_averaged_clusters;
    this->initPPFs(points, normals, cloud_ptr->size(), d_dist);
    this->modelPointVoteWeights = thrust::device_vector<float>(n, 1.0);

    // For each element of data, compute a 32-bit hash,
    thrust::device_vector<unsigned int> nonunique_hashkeys =
        thrust::device_vector<unsigned int>(this->modelPPFs->size());
    int blocks = std::min(((int)(this->modelPPFs->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_hash_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                           thrust::raw_pointer_cast(nonunique_hashkeys.data()),
                                           this->modelPPFs->size());

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    this->search_array = ParallelHashArray<unsigned int>(nonunique_hashkeys);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}

Model::~Model(){
    // TODO
    if(!this->cpu_clustering){
        delete this->vote_counts_out;
    }
}

void Model::SetModelPointVoteWeights(thrust::device_vector<float> modelPointVoteWeights){
    this->modelPointVoteWeights = modelPointVoteWeights;
}

void Model::ComputeUniqueVotes(Scene *scene){
    thrust::device_vector<std::size_t> *sceneIndices =
        this->search_array.GetIndices(*(scene->getHashKeys()));

    thrust::device_vector<std::size_t> *ppf_vote_counts =
        new thrust::device_vector<unsigned long>(scene->getHashKeys()->size());
    int blocks = std::min(((int)(scene->getHashKeys()->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_vote_count_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(scene->getHashKeys()), RAW_PTR(sceneIndices),
         RAW_PTR(this->search_array.GetHashkeys()), RAW_PTR(this->search_array.GetCounts()),
         RAW_PTR(ppf_vote_counts), scene->getHashKeys()->size());
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    thrust::device_vector<std::size_t> *ppf_vote_indices =
        new thrust::device_vector<std::size_t>(scene->getHashKeys()->size());
    thrust::exclusive_scan(ppf_vote_counts->begin(),
                           ppf_vote_counts->end(),
                           ppf_vote_indices->begin());

    // TODO: don't copy the entire vector for just the last element
    thrust::host_vector<std::size_t> *host_ppf_vote_counts =
        new thrust::host_vector<std::size_t>(*ppf_vote_counts);
    thrust::host_vector<std::size_t> *host_ppf_vote_indices =
        new thrust::host_vector<std::size_t>(*ppf_vote_indices);

    std::size_t num_votes = host_ppf_vote_counts->back() + host_ppf_vote_indices->back();
    BOOST_LOG_TRIVIAL(debug) << boost::format("num_nonunique_votes: %lu") % num_votes;

    delete ppf_vote_counts;
    delete host_ppf_vote_counts;
    delete host_ppf_vote_indices;

    thrust::device_vector<unsigned long> *nonunique_nonempty_votes =
        new thrust::device_vector<unsigned long>(num_votes);

    blocks = std::min(((int)(scene->getHashKeys()->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_vote_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(scene->getHashKeys()), RAW_PTR(sceneIndices),
         RAW_PTR(this->search_array.GetHashkeys()), RAW_PTR(this->search_array.GetCounts()),
         RAW_PTR(this->search_array.GetFirstHashkeyIndices()),
         RAW_PTR(this->search_array.GetHashkeyToDataMap()),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         this->n, RAW_PTR(scene->getModelPoints()),
         RAW_PTR(scene->getModelNormals()), scene->numPoints(),
         RAW_PTR(ppf_vote_indices),
         RAW_PTR(nonunique_nonempty_votes),
         scene->getHashKeys()->size(),
         this->d_dist);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    delete ppf_vote_indices;

    thrust::sort(nonunique_nonempty_votes->begin(), nonunique_nonempty_votes->end());
    this->votes = new thrust::device_vector<unsigned long>();
    this->voteCounts = new thrust::device_vector<unsigned int>();
    histogram(*nonunique_nonempty_votes, *(this->votes), *(this->voteCounts));
    BOOST_LOG_TRIVIAL(debug) << boost::format("num_unique_votes: %lu") % this->votes->size();
    delete nonunique_nonempty_votes;

    thrust::sort_by_key(this->voteCounts->begin(),
                        this->voteCounts->end(),
                        this->votes->begin(),
                        thrust::greater<float>());

    thrust::host_vector<unsigned int> temp_votecounts(*this->voteCounts);
    BOOST_LOG_TRIVIAL(debug) << boost::format("1, 5, 10, 50: %u, %u, %u, %u") %
        temp_votecounts[1] % temp_votecounts[5] % temp_votecounts[10] % temp_votecounts[50];

    float min_votecount = vote_count_threshold * temp_votecounts[0];
    std::size_t num_top_votes = thrust::count_if(
      voteCounts->begin(), voteCounts->end(),
      [min_votecount] __device__ (unsigned int x){ return x > min_votecount; });
    BOOST_LOG_TRIVIAL(debug) << boost::format("num_top_votes: %lu") % num_top_votes;
    this->votes->resize(num_top_votes);
    this->voteCounts->resize(num_top_votes);
}

thrust::device_vector<float>
Model::ComputeWeightedVoteCounts(thrust::device_vector<unsigned long> votes,
                                 thrust::device_vector<unsigned int> vote_counts,
                                 thrust::device_vector<float> modelpoint_vote_weights){
    thrust::device_vector<float> weighted_vote_counts =
        thrust::device_vector<float>(votes.size());

    int blocks = std::min(((int)(votes.size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    vote_weight_kernel<<<blocks,BLOCK_SIZE>>>
        (thrust::raw_pointer_cast(votes.data()),
         thrust::raw_pointer_cast(vote_counts.data()),
         thrust::raw_pointer_cast(modelpoint_vote_weights.data()),
         thrust::raw_pointer_cast(weighted_vote_counts.data()),
         votes.size());

    return weighted_vote_counts;
}

void Model::ComputeTransformations(Scene *scene){
    this->transformations = thrust::device_vector<float>(this->votes->size()*16);
    int blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    trans_calc_kernel2<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(votes),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         RAW_PTR(scene->getModelPoints()), RAW_PTR(scene->getModelNormals()),
         thrust::raw_pointer_cast(this->transformations.data()),
         this->votes->size());
}

thrust::device_vector<float> *Model::ClusterTransformations(){
    this->transformation_trans =
        new thrust::device_vector<float3>(this->votes->size());
    this->transformation_rots =
        new thrust::device_vector<float4>(this->votes->size());
    int blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    mat2transquat_kernel<<<blocks,BLOCK_SIZE>>>
        (thrust::raw_pointer_cast(this->transformations.data()),
         thrust::raw_pointer_cast(transformation_trans->data()),
         thrust::raw_pointer_cast(transformation_rots->data()),
         this->votes->size());
    thrust::device_vector<unsigned int> nonunique_trans_hash =
        thrust::device_vector<unsigned int>(this->votes->size());
    thrust::device_vector<unsigned int> adjacent_trans_hash =
        thrust::device_vector<unsigned int>(27*this->votes->size());
    trans2idx_kernel<<<blocks,BLOCK_SIZE>>>
    (thrust::raw_pointer_cast(transformation_trans->data()),
     thrust::raw_pointer_cast(nonunique_trans_hash.data()),
     thrust::raw_pointer_cast(adjacent_trans_hash.data()),
     this->votes->size(), this->d_dist);
    ParallelHashArray<unsigned int> trans_search_array =
        ParallelHashArray<unsigned int>(nonunique_trans_hash);

    thrust::device_vector<std::size_t> *transIndices =
        trans_search_array.GetIndices(adjacent_trans_hash);
    thrust::device_vector<float> *vote_counts_out =
        new thrust::device_vector<float>(this->weightedVoteCounts.size());
    blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    rot_clustering_kernel<<<blocks,BLOCK_SIZE>>>
        (thrust::raw_pointer_cast(transformation_trans->data()),
         thrust::raw_pointer_cast(transformation_rots->data()),
         thrust::raw_pointer_cast(this->weightedVoteCounts.data()),
         thrust::raw_pointer_cast(adjacent_trans_hash.data()),
         RAW_PTR(transIndices), RAW_PTR(trans_search_array.GetHashkeys()),
         RAW_PTR(trans_search_array.GetCounts()),
         RAW_PTR(trans_search_array.GetFirstHashkeyIndices()),
         RAW_PTR(trans_search_array.GetHashkeyToDataMap()),
         thrust::raw_pointer_cast(vote_counts_out->data()),
         this->votes->size(), this->d_dist,
         this->use_l1_norm, this->use_averaged_clusters);
    delete transIndices;
    return vote_counts_out;
}

PoseWithVotesList Model::ClusterTransformationsCPU(){
    thrust::host_vector<float> transformations(this->transformations);
    thrust::host_vector<float> vote_counts(*this->voteCounts);
    PoseWithVotesList transformations_list = PoseWithVotesList();
    for(int i = 0; i < this->votes->size(); i++){
        Eigen::Matrix4f transmat;
        float vote_count = (float) vote_counts[i];
        memcpy(transmat.data(), transformations.data()+ 16*i, 16*sizeof(float));
        // Eigen is column-major by default.
        transmat.transposeInPlace();
        Eigen::Affine3f transaff;
        transaff = transmat;
        transformations_list.push_back(PoseWithVotes(transaff, vote_count));
    }

    PoseWithVotesList clustered_transformations_list = PoseWithVotesList();
    clusterPoses(transformations_list, clustered_transformations_list,
                 this->d_dist, D_ANGLE0);

    return clustered_transformations_list;
}


void Model::ppf_lookup(Scene *scene){

    #ifdef DEBUG
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
    #endif

    ComputeUniqueVotes(scene);
    ComputeTransformations(scene);
    BOOST_LOG_TRIVIAL(debug) << boost::format("votes_size: %lu") % this->votes->size();


    this->weightedVoteCounts =
        ComputeWeightedVoteCounts(*(this->votes),
                                  *(this->voteCounts),
                                  this->modelPointVoteWeights);

    if(this->cpu_clustering){
        this->cpu_transformations = ClusterTransformationsCPU();
    }
    else{
        this->vote_counts_out = ClusterTransformations();
        this->max_idx =
            thrust::max_element(this->vote_counts_out->begin(),
                                this->vote_counts_out->end()) - this->vote_counts_out->begin();
    }

#ifdef DEBUG
    // end cuda timer
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    BOOST_LOG_TRIVIAL(info) << boost::format("Time to lookup model:  %3.1f ms") % elapsedTime;
#endif
}

thrust::device_vector<float> Model::getTransformations(){
    return this->transformations;
}
