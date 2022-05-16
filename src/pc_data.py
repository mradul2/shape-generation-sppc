import os 
import glob
import numpy as np

class PCD():
    """
        Reads a .pcd file and saves the list of point cloud data
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'r')
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

        self.data_points = []
        for line in self.file:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                self.data_points.append(line)
        
        self.np_data = np.array(self.data_points)

    def __len__(self):
        return len(self.data_points)