import random
from ast import Num
from audioop import avg
from turtle import shape

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


class PCA_():
    def __init__(self, size_basis: int, num_data: int):
        self.num_data = num_data
        self.num_point_cloud = 1000
        self.num_point_cloud_dim = 3
        self.size_basis = size_basis

    def fit_once(self, matrix):
        self.pca = PCA(n_components=self.size_basis)
        matrix_reshaped = np.reshape(matrix, (matrix.shape[0], self.num_point_cloud * self.num_point_cloud_dim))
        matrix_reshaped = self.normalize(matrix_reshaped)
        self.pca.fit(matrix_reshaped)

    def rereshape(self, matrix):
        matrix_reshape = np.reshape(matrix, (matrix.shape[0], self.num_point_cloud, self.num_point_cloud_dim))
        return matrix_reshape

    def normalize(self, matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        matrix = (matrix - mean) / std
        return matrix

    def transform_data(self, matrix):
        matrix_reshape = np.reshape(matrix, (matrix.shape[0], self.num_point_cloud * self.num_point_cloud_dim))
        matrix_transformed = self.pca.transform(matrix_reshape)
        return matrix_transformed

    def inverse_transform_data(self, matrix):
        matrix_inverse_transformed = self.pca.inverse_transform(matrix)
        return self.rereshape(matrix_inverse_transformed)

    def reconstruction_error(self, matrix):
        # Using library to compute the loss (faster than manual numpy code)
        if matrix.ndim == 2:
            shape_vector = np.reshape(matrix, (1, 3000))
        if matrix.ndim == 3:
            shape_vector = np.reshape(matrix, (5000, 3000))
        shape_vector_projected = self.pca.transform(shape_vector)
        shape_vector_unprojected = self.pca.inverse_transform(shape_vector_projected)
        return np.sum((shape_vector - shape_vector_unprojected) ** 2, axis=1).mean()

    def optimize_point_ordering_vectorized(self, K: int, matrix):
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