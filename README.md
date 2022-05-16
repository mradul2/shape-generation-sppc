# Shape Generation Using Spatially Partitioned Point Clouds 

This repository implements the following utilities:

1. Spatially partitioned point cloud
2. PCA
3. Iterative Point Ordering
4. GAN

## Point Cloud Data

PCD file first contains a header containing information about the point cloud data followed by the data itself.

## Poisson Disc Sampling



## K Dimensional Tree

Binary search tree in which data in each node is a K-dimensional point in space. 

Building a K-d tree for 3D Data: 

1. Select the initial axis and sort the data according to it, and insert median of that data as the root. 
2. 

## Principal Component Analysis

Principal Component Analysis (PCA) is a method to for dimensionality reduction of a dataset and to speed up the applied algorithm. 
Before applying PCA to any dataset, first the dataset needs to be standardized onto a unit scale where the mean is 0 and variance is 1.

In this case, we apply PCA on the sorted point cloud by generating a matrix of size 3N x S, where N is number of points in each shape and S is the number of shapes in the dataset. And then perform PCA on the matrix: P = UΣV, resulting in the linear shape basis U and projections V. By default, the size of shape basis is chosen to be B = 100. 