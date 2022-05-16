from pc_data import PCD

class Node():
    def __init__(self, point, parent, depth, left_children, right_children):
        self.point = point
        self.parent = parent
        self.depth = depth
        self.splitting_axes = self.depth%3
        self.left_childern = left_children
        self.right_children = right_children
    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False

class KdTree():
    def __init__(self, pcd: PCD):
        self.pcd = pcd  
        self.num = len(self.pcd)
        self.points = self.pcd.points

    def build_tree(self):
        median, left, right = self.partition(self.points, 0)
        self.root = Node(median, None, 0)

        
    def partition(points: list, depth: int):
        dimension = depth%3
        points.sort(key = lambda list: list[dimension])    
        median = len(points)//2
        left_children = points[:median]
        right_children = points[median + 1:]

        return points[median], left_children, right_children

    def sorted_list(self):
        pass