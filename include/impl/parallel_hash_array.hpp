#ifndef PARALLEL_HASH_ARRAY_H
#define PARALLEL_HASH_ARRAY_H

#include <cstdint>
#include <cstdlib>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "impl/util.hpp"

template<typename T>
class ParallelHashArray {

public:
    ParallelHashArray();
    ParallelHashArray(thrust::device_vector<T>& data);

    // ~ParallelHashArray();

    thrust::device_vector<std::size_t> *GetIndices(thrust::device_vector<unsigned int>&
                                                   data_hashkeys);
    thrust::device_vector<unsigned int> *GetHashkeys();
    thrust::device_vector<std::size_t> *GetCounts();
    thrust::device_vector<std::size_t> *GetFirstHashkeyIndices();
    thrust::device_vector<std::size_t> *GetHashkeyToDataMap();


protected:
    // Size of data.
    std::size_t n;

    // Indices into data.
    // nonunique_hashkeys[i] == hash(data[hashkeyToDataMap[i]])
    thrust::device_vector<std::size_t> hashkeyToDataMap;

    // *unique* hashkeys.
    thrust::device_vector<unsigned int> hashkeys;
    // number of occurances of each hashkey
    thrust::device_vector<std::size_t> counts;
    // Indices in hashkeyToDataMap where blocks of identical hashkeys begin.
    thrust::device_vector<std::size_t> firstHashkeyIndex;
};

template <typename T>
ParallelHashArray<T>::ParallelHashArray(){
    // TODO
}


template <typename T>
ParallelHashArray<T>::ParallelHashArray(thrust::device_vector<T>& data){
    this->n = data.size();

    this->hashkeyToDataMap = thrust::device_vector<std::size_t>(this->n);
    thrust::sequence(hashkeyToDataMap.begin(), hashkeyToDataMap.end());

    // Sort nonunique_hashkeys and hashkeyToDataMap.
    thrust::sort_by_key(data.begin(),
                        data.end(),
                        hashkeyToDataMap.begin());

    // Create array of unique hashkeys and their associated counts.
    this->hashkeys = thrust::device_vector<unsigned int>();
    this->counts = thrust::device_vector<std::size_t>();
    histogram(data, this->hashkeys, this->counts);

    // Find the indices in hashkeyToDataMap of the beginning of each block of identical hashkeys.
    this->firstHashkeyIndex = thrust::device_vector<std::size_t>(this->hashkeys.size());
    thrust::exclusive_scan(this->counts.begin(),
                           this->counts.end(),
                           this->firstHashkeyIndex.begin());
}

// TODO: create version which hashes data for you
template <typename T>
thrust::device_vector<std::size_t> *ParallelHashArray<T>::GetIndices(thrust::device_vector<unsigned int>&
                                                                     data_hashkeys){
    // find possible starting indices of blocks matching Model hashKeys
    thrust::device_vector<std::size_t> *dataIndices =
        new thrust::device_vector<std::size_t>(data_hashkeys.size());
    thrust::lower_bound(this->hashkeys.begin(),
                        this->hashkeys.end(),
                        data_hashkeys.begin(),
                        data_hashkeys.end(),
                        dataIndices->begin());
    return dataIndices;
}

template <typename T>
thrust::device_vector<unsigned int> *ParallelHashArray<T>::GetHashkeys(){
    return &this->hashkeys;
}

template <typename T>
thrust::device_vector<std::size_t> *ParallelHashArray<T>::GetCounts(){
    return &this->counts;
}

template <typename T>
thrust::device_vector<std::size_t> *ParallelHashArray<T>::GetFirstHashkeyIndices(){
    return &this->firstHashkeyIndex;
}

template <typename T>
thrust::device_vector<std::size_t> *ParallelHashArray<T>::GetHashkeyToDataMap(){
    return &this->hashkeyToDataMap;
}

#endif /* PARALLEL_HASH_ARRAY_H */
