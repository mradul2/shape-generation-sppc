from src.pc_data import PCD
import numpy as np
import time

class KdTree():
    def __init__(self, points: np.ndarray):
        self.points = points.copy()
        self.num = self.points.shape[0]
        self.build_tree()

    def build_tree(self):
        start_time = time.time()
        self.partition(self.points, depth=0, start=0, end=self.num-1)
        end_time = time.time()
        # print("Time elapsed for building the Kd Tree (In seconds): ", end_time - start_time)

    def partition(self, points: np.ndarray, depth: int, start: int, end: int):
        if end <= start:
            return 
        dimension = depth%3
        points[start:end+1] = points[start + points[start:end+1, dimension].argsort()]
        median = start + ((end - start + 1) // 2)
        depth += 1
        self.partition(points, depth, start, median-1)
        self.partition(points, depth, median+1, end)