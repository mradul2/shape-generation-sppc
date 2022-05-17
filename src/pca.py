from ast import Num
from audioop import avg
import numpy as np
from sklearn.decomposition import PCA
import random
from tqdm import tqdm
from numpy.linalg import norm

class PCA_():
    def __init__(self, matrix: np.ndarray, size_basis: int):
        # (N, M, 3)
        self.matrix = matrix 
        self.num_data = self.matrix.shape[0]
        self.num_point_cloud = self.matrix.shape[1]
        self.num_point_cloud_dim = self.matrix.shape[2]
        # (N, M * 3)
        self.size_basis = size_basis
        self.pca = PCA(n_components=self.size_basis)
        self.fit_once()

    def fit_once(self):
        self.matrix_reshaped = self.reshape(self.matrix)
        self.matrix_reshaped = self.normalize(self.matrix_reshaped)
        self.pca.fit(self.matrix_reshaped)

    def reshape(self, matrix):
        matrix_reshape = np.reshape(matrix, (self.num_data, self.num_point_cloud * self.num_point_cloud_dim))
        return matrix_reshape

    def rereshape(self, matrix):
        matrix_reshape = np.reshape(matrix, (self.num_data, self.num_point_cloud, self.num_point_cloud_dim))
        return matrix_reshape

    def normalize(self, matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        matrix = (matrix - mean) / std
        return matrix

    def transform_data(self, matrix):
        matrix_reshape = self.reshape(matrix)
        matrix_transformed = self.pca.transform(matrix_reshape)
        return matrix_transformed

    def inverse_transform_data(self, matrix):
        matrix_inverse_transformed = self.pca.inverse_transform(matrix)
        return self.rereshape(matrix_inverse_transformed)

    def reconstruction_error(self, matrix):
        shape_matrix = np.reshape(matrix, (3000, 1))
        matrix = self.reshape(self.matrix)
        myu = np.mean(matrix, axis=0).reshape(3000, 1)
        loss = (shape_matrix - myu).T @ self.pca.components_.T @ self.pca.components_ + myu - shape_matrix
        loss = norm(loss) ** 2
        return loss

    def optimize_point_ordering(self, K: int):
        avg_loss = 0
        for shape in tqdm(range(self.num_data)):
            shape_matrix = self.matrix[shape].copy() # (1000, 3)
            for _ in range(K):
                recon_error_prev = self.reconstruction_error(shape_matrix)
                random_nums = random.sample(range(0, self.num_point_cloud), 2)
                temp = shape_matrix[random_nums[0]]
                shape_matrix[random_nums[0]] = shape_matrix[random_nums[1]]
                shape_matrix[random_nums[1]] = temp
                recon_error = self.reconstruction_error(shape_matrix)
                if recon_error < recon_error_prev:
                    self.matrix[shape][random_nums[0]] = shape_matrix[random_nums[0]]
                    self.matrix[shape][random_nums[1]] = shape_matrix[random_nums[1]]
            avg_loss += self.reconstruction_error(shape_matrix)
        avg_loss /= self.num_data
        print(avg_loss)
        self.fit_once()
        

                 
 