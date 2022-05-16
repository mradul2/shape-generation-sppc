from src.pc_data import PCD
from src.kd_tree import KdTree

pcd1 = PCD('/Users/mradulagrawal/shape-generation-sppc/data/shapenet-chairs-pcd/1.pcd')
print(len(pcd1))
print(pcd1.points[2])

kdtree1 = KdTree(pcd1)
kdtree1.build_tree()
sorted_list = kdtree1.sorted_list()