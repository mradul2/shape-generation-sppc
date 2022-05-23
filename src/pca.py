import random
from ast import Num
from audioop import avg
from turtle import shape

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


class PCA_():
    """
        Class for PCA operations and optimization
    """
    def __init__(self, size_basis: int, num_data: int):
        """
            Initialize the PCA object

            Args:
                size_basis: The number of basis vectors to use
                num_data: The number of data points to use
        """
        self.num_data = num_data
        self.num_point_cloud = 1000
        self.num_point_cloud_dim = 3
        self.size_basis = size_basis

    def fit_once(self, matrix):
        """
            Function to fit the PCA model

            Args:
                matrix: The matrix to fit the model to
        """
        self.pca = PCA(n_components=self.size_basis)
        matrix_reshaped = np.reshape(matrix, (matrix.shape[0], self.num_point_cloud * self.num_point_cloud_dim))
        matrix_reshaped = self.normalize(matrix_reshaped)
        self.pca.fit(matrix_reshaped)

    def normalize(self, matrix):
        """
            Function to the normalise the point cloud data
            before fitting the PCA model

            Args:
                matrix: The matrix to normalise

            Returns:
                The normalised matrix
        """
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        
        # Normalise the data
        matrix = (matrix - mean) / std
        # The shape of normalized matrix is (num_data, num_point_cloud * num_point_cloud_dim)!
        return matrix

    def transform_data(self, matrix):
        """
            Function to transform the original point cloud data 
            to the new basis useinf the trained PCA model

            Args:
                matrix: The matrix to transform

            Returns:
                The transformed matrix
        """
        # First reshape the input matrix to (num_data, num_point_cloud * num_point_cloud_dim)
        matrix_reshape = np.reshape(matrix, (matrix.shape[0], self.num_point_cloud * self.num_point_cloud_dim))
       
        # Perform the transformation
        matrix_transformed = self.pca.transform(matrix_reshape)
        # Shape of transformed matrix is (num_data, num_basis)!
        return matrix_transformed

    def inverse_transform_data(self, matrix):
        """
            Function to inverse transform the new basis to the original point cloud data

            Args:
                matrix: The matrix to inverse transform of shape (num_data, num_basis)

            Returns:
                The inverse transformed matrix
        """
        # Perform the inverse transformation
        # Shape of the inverse transformed matrix is (num_data, num_point_cloud * num_point_cloud_dim)
        matrix_inverse_transformed = self.pca.inverse_transform(matrix)

        # Reshape the matrix to (num_data, num_point_cloud, num_point_cloud_dim)
        matrix_inverse_transformed_reshaped = np.reshape(matrix_inverse_transformed, (matrix_inverse_transformed.shape[0], self.num_point_cloud, self.num_point_cloud_dim))
        return matrix_inverse_transformed_reshaped

    def reconstruction_error(self, matrix):
        """
            Function to calculate the reconstruction error of the point cloud data

            Args:
                matrix: The matrix to calculate the reconstruction error of

            Returns:
                The reconstruction error
        """
        # Using library to compute the loss (faster than manual numpy code)
        if matrix.ndim == 2:
            shape_vector = np.reshape(matrix, (1, 3000))
        if matrix.ndim == 3:
            shape_vector = np.reshape(matrix, (5000, 3000))
        shape_vector_projected = self.pca.transform(shape_vector)
        shape_vector_unprojected = self.pca.inverse_transform(shape_vector_projected)
        return np.sum((shape_vector - shape_vector_unprojected) ** 2, axis=1).mean()

    def optimize_point_ordering_vectorized(self, K: int, matrix):
        """
            Function to optimize the point ordering of the point cloud data
            using the vectorized implementation

            Args:
                K: The number of iterations to perform
                matrix: The matrix to optimize the point ordering of

            Returns:
                Total reconstruction error and the optimized matrix
        """
        total_error_prev = self.reconstruction_error(matrix.copy()) # (5000, 1000, 3)
        print("Before Re-ordering: ", total_error_prev)
        for _ in tqdm(range(K)):
            matrix_copy = matrix.copy() # (1000, 3)
            recon_error_prev = self.reconstruction_error(matrix_copy)

            random_nums = random.sample(range(0, self.num_point_cloud), 2)
            # Matrices were not properly swapping up if these copies are not made
            temp0 = matrix_copy[:, random_nums[0]].copy()
            temp1 = matrix_copy[:, random_nums[1]].copy() 

            # Swap the two random points and calc the recon error
            matrix_copy[:, random_nums[0]] = temp1
            matrix_copy[:, random_nums[1]] = temp0
            recon_error = self.reconstruction_error(matrix_copy)
            # If the reconstruction error decrease then swap
            if recon_error < recon_error_prev:
                matrix[shape][:, random_nums[1]] = temp0
                matrix[shape][:, random_nums[0]] = temp1
                print("WOW")

        total_error = self.reconstruction_error(matrix.copy()) # (5000, 1000, 3)
        print("After Re-ordering: ", total_error)
        return total_error, matrix

    def optimize_point_ordering(self, K: int, matrix):
        """
            Function to optimize the point ordering of the point cloud data
            without using the vectorized implementation

            Args:
                K: The number of iterations to perform
                matrix: The matrix to optimize the point ordering of

            Returns:
                Total reconstruction error and the optimized matrix
        """
        total_error_prev = self.reconstruction_error(matrix.copy()) # (5000, 1000, 3)
        print("Before Re-ordering: ", total_error_prev)
        for shape in tqdm(range(self.num_data)):
            for _ in range(K):
                matrix_copy = matrix[shape].copy() # (1000, 3)
                recon_error_prev = self.reconstruction_error(matrix_copy)

                random_nums = random.sample(range(0, self.num_point_cloud), 2)
                # Matrices were not properly swapping up if these copies are not made
                temp0 = matrix_copy[random_nums[0]].copy()
                temp1 = matrix_copy[random_nums[1]].copy()

                # Swap the two random points and calc the recon error
                matrix_copy[random_nums[0]] = temp1
                matrix_copy[random_nums[1]] = temp0
                recon_error = self.reconstruction_error(matrix_copy)

                # If the reconstruction error decrease then swap
                if recon_error < recon_error_prev:
                    matrix[shape][random_nums[1]] = temp0
                    matrix[shape][random_nums[0]] = temp1

        total_error = self.reconstruction_error(matrix.copy()) # (5000, 1000, 3)
        print("After Re-ordering: ", total_error)
        return total_error, matrix