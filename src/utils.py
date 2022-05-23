import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualise_point_cloud(np_data: np.ndarray):
    """
        Function to visualize the point cloud.

        Args:
            np_data: Numpy array of the point cloud data.

        Returns:
            None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np_data[:,0], np_data[:,1], np_data[:,2], zdir='z')
    plt.show()

def visualise_point_cloud_gradient(np_data: np.ndarray):
    """
        Function to visualize the point cloud with a color gradient.

        Args:
            np_data: Numpy array of the point cloud data.

        Returns:
            None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cm = plt.get_cmap("RdYlGn")
    col = [cm(float(i)/(1000)) for i in range(1000)]
    ax.scatter(np_data[:,0], np_data[:,1], np_data[:,2], zdir='z', Color=col)
    plt.show()