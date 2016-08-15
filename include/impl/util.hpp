#ifndef UTIL_IMPL_H
#define UTIL_IMPL_H

#include <cstdio>
#include <cstdlib>

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <cuda.h>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>


static void HandleError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err),
                file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

#define RAW_PTR(V) thrust::raw_pointer_cast(V->data())

template <typename Vector1, typename Vector2, typename Vector3>
void histogram(const Vector1& input,  // assumed to be already sorted
               Vector2& histogram_values,
               Vector3& histogram_counts){
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector3::value_type IndexType; // histogram index type

    thrust::device_vector<ValueType> data(input);
    IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                               data.begin() + 1,
                                               IndexType(1),
                                               thrust::plus<IndexType>(),
                                               thrust::not_equal_to<ValueType>());
    histogram_values.resize(num_bins);
    histogram_counts.resize(num_bins);

    BOOST_LOG_TRIVIAL(debug) << boost::format("num_bins: %d") % num_bins;

    thrust::reduce_by_key(data.begin(), data.end(),
                          thrust::constant_iterator<IndexType>(1),
                          histogram_values.begin(),
                          histogram_counts.begin());
}


template <typename T>
void write_array(const char *filename, T *data, int n){
    FILE *fp;

    if(!(fp = fopen(filename,"wb"))){
        BOOST_LOG_TRIVIAL(error) << boost::format("Error opening file file %s.") % filename;
        exit(1);
    }

    if(fwrite(data, sizeof(T), n, fp) != n){
        BOOST_LOG_TRIVIAL(error) << boost::format("Error writing to file %s.") % filename;
        exit(2);
    }

    if(fclose(fp)){
        BOOST_LOG_TRIVIAL(info) << boost::format("Error closing file %s.") % filename;
        exit(3);
    }
}


template <typename T>
void write_device_array(const char *filename, T *data, int n){
    T *host_array = new T[n];
    if(cudaMemcpy(host_array, data, n*sizeof(T), cudaMemcpyDeviceToHost)
       != cudaSuccess){
        BOOST_LOG_TRIVIAL(error) << "Error copying data from device to host.";
        exit(4);
    }
    write_array(filename, host_array, n);
    delete []host_array;
}

template <typename T>
void write_device_vector(const char *filename, thrust::device_vector<T> *data){
    write_device_array(filename, RAW_PTR(data), data->size());
}


// http://eigen.tuxfamily.org/bz/show_bug.cgi?id=622
template<typename Derived>
std::istream &operator>>(std::istream &s,
                           Eigen::MatrixBase<Derived> &m){
    for(int i = 0; i < m.rows(); ++i){
        for(int j = 0; j < m.cols(); j++){
            s >> m(i,j);
        }
    }
    return s;
}


#endif /* UTIL_IMPL_H */
