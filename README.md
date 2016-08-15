# ppf-registration-spatial-hashing

This is an implementation of GPU point pair feature registration
[1],[2]. Also present is an implementation of a parallel hash array, a
new associative array data structure optimized for parallel batch
operations, and an implementation of a new GPU-based spatial hashing
clustering algorithm that is used for the clustering step in the point
pair feature registration algorithm.

A complete description of this project is in my masters thesis,
ppf-registration-spatial-hashing.pdf.

https://github.com/nicolasavru/objective-slam contains the development
history of this project.

I would like to eventually contribute this code to Point Cloud
Library, but significant cleanup and testing will need to be done
first.


## Dependencies

* NVIDIA CUDA 8.0 (though it may work with earlier versions)
* Boost
* Point Cloud Library 1.8
* CMake
* GCC 5


## Building

```
$ mkdir build && cd build
$ cmake ..
$ make
```


## Usage

The expected input is one or more scene files and one or more model
files. Each scene and model file should be a ply file with normals
(you may also need to strip out the "face" field). A sample dataset
(the Mian dataset) can be obtained from [3] (the "Laser Scanner"
dataset). Unfortunately the ply files in that dataset do not contain
normals, so you will have to compute them yourself. One way to do this
is with the included compute_normals MATLAB script (assuming that the
Mian dataset was extracted to /tmp/UWA):

```
>> files = dir('/tmp/UWA/*.ply')
>> for file = files'
>>  compute_normals(file.name, strcat('/tmp/UWA/norms/', file.name), [0; 0; 0])
>> end
```

compute_normals relies on the PLY_IO library at [4].

Once you have generated normals for the dataset, you can run the ppf
registration algorithm with:

```
$ ./alignment --scene_files=/tmp/UWA/norms/rs1_0.ply \
              --model_files=$(echo /tmp/UWA/norms/{cheff,T-rex_high,parasaurolophus_high,chicken_high}_0.ply | tr ' ' ,) \
              --validation_files=$(echo /tmp/UWA/{chef,T-rex,parasaurolophus,chicken}-rs1.xf | tr ' ' ,) \
              --tau_d=0.030,0.035,0.030,0.050 \
              --ref_point_df=5 \
              --scene_leaf_size=7 \
              --vote_count_threshold=0.3 \
              --show_normals=false \
              --cpu_clustering=false \
              --use_l1_norm=false \
              --use_averaged_clusters=false \
              --dev=0 \
              --loglevel=debug \
              --logfile=run1.log
```


## References

[1] Bertram Drost et al. “Model globally, match locally: Efficient and
robust 3D object recognition”. In: Computer Vision and Pattern
Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010, pp. 998–1005.

[2] Renato F. Salas-Moreno et al. “SLAM++: Simultaneous Localisation
and Mapping at the Level of Objects”. In: Computer Vision and Pattern
Recognition (CVPR), 2013 IEEE Conference on. IEEE, 2013, pp.
1352–1359.

[3] http://vision.deis.unibo.it/keypoints3d/?page_id=2

[4] http://people.sc.fsu.edu/~jburkardt/m_src/ply_io/ply_io.html
