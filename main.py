from src.pc_data import PCD
from src.kd_tree import KdTree
from src.utils import visualise_point_cloud, visualise_point_cloud_gradient


def main():
    pcd1 = PCD('/Users/mradulagrawal/shape-generation-sppc/data/shapenet-chairs-pcd/45.pcd')
    pcd1 = pcd1.np_data
    visualise_point_cloud_gradient(pcd1)

    kdtree = KdTree(pcd1).points
    visualise_point_cloud_gradient(kdtree)

if __name__ == '__main__':
    main()
