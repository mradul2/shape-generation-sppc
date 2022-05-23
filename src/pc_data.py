import os 
import glob
import numpy as np

class PCD():
    """
        Class for the Point Cloud Data.
    """
    def __init__(self, file_path):
        """
            Initialize the PCD class.

            Args:
                file_path: Path to the point cloud file.

            Returns:
                None
        """
        # Opening the .pcd file 
        self.file_path = file_path
        self.file = open(file_path, 'r')

        # Reading the header of the file
        self.version = self.file.readline()
        self.fields =  [i for i in self.file.readline().split()]
        self.size = [i for i in self.file.readline().split()]
        self.type = [i for i in self.file.readline().split()]
        self.count = [i for i in self.file.readline().split()]
        self.width = [i for i in self.file.readline().split()]
        self.height = self.file.readline()
        self.viewpoints = [i for i in self.file.readline().split()]
        self.points = self.file.readline()
        self.data = self.file.readline()

        # List of point coordinates
        self.data_points = []

        # Reading the point coordinates
        for line in self.file:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                self.data_points.append(line)
        
        # Convert the list of points to numpy array
        # Numpy array of shape (num_points, 3) is created!
        self.np_data = np.array(self.data_points)

    def __len__(self):
        """
            Return the length of the point cloud.
        """
        return len(self.data_points)