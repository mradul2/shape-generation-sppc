from src.pc_data import PCD
import numpy as np
import time

class KdTree():
    """
       Class for the Kd Tree implementation and its functions.
    """
    def __init__(self, points: np.ndarray):
        """
            Function to initialize the Kd Tree object

            Args:
                points: Point cloud data to sort using the Kd Tree
        """
        self.points = points.copy()
        self.num = self.points.shape[0]
        self.build_tree()

    def build_tree(self):
        """
            Function to build the Kd Tree
            After calling this function the input point cloud array is sorted
        """
        start_time = time.time()
        # Call the recursive partition function with initial depth = 0
        self.partition(self.points, depth=0, start=0, end=self.num-1)
        end_time = time.time()
        # print("Time elapsed for building the Kd Tree (In seconds): ", end_time - start_time)

    def partition(self, points: np.ndarray, depth: int, start: int, end: int):
        """
            Recursive function to partition the point cloud data
        """
        if end <= start:
            return 
        # Alternating axes with the increasing depth
        dimension = depth%3
        # Sort the points based on the current axis
        points[start:end+1] = points[start + points[start:end+1, dimension].argsort()]
        # Find the median
        median = start + ((end - start + 1) // 2)
        depth += 1
        # Recursively call the partition function for the left and right sub-arrays
        self.partition(points, depth, start, median-1)
        self.partition(points, depth, median+1, end)