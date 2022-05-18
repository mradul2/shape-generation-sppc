import argparse
import os

import numpy as np

from src.kd_tree import KdTree
from src.pc_data import PCD
import wandb
from src.pca import PCA_
from src.utils import visualise_point_cloud, visualise_point_cloud_gradient

from src.gan import GAN

def process_data(args):
    root_dir_path = args.load_path
    list_of_files = os.listdir(root_dir_path)
    print("Number of Shapes: ", len(list_of_files))
    
    print("Building KdTree for all shapes...")
    matrix = []
    for file in list_of_files:
        pcd = PCD(os.path.join(root_dir_path, file))
        pcd_np = pcd.np_data
        kdtree = KdTree(pcd_np)
        kdtree_np = kdtree.points
        matrix.append(kdtree_np)
    print("KdTree build complete")
    matrix_np = np.array(matrix)

    pca = PCA_(matrix_np, 100)
    pca.fit_once()
    # output = pca.transform_data(matrix_np)
    # output = pca.inverse_transform_data(output)
    # visualise_point_cloud_gradient(output[0])

    print("Optimizing Point Ordering...")
    I = 1000
    K = 10000
    for _ in range(I):
        pca.optimize_point_ordering(K)
    print("Point ordering completed")

    final_matrix = pca.transform_data(matrix_np)
    save_path = os.path.join(args.save_path, 'processed_data.npy')
    np.save(save_path, final_matrix)
    print("Processed data saved as: ", save_path)
    

def train(args):
    print("Training function called...")
    gan = GAN(args)
    gan.train()
    gan.save_model()

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-m', '--mode',
        default='process_data',
        help='Mode')
    argparser.add_argument(
        '--load_path',
        default='/Users/mradulagrawal/shape-generation-sppc/data/shapenet-chairs-pcd',
        help='Directory path to laod the data')
    argparser.add_argument(
        '--save_path',
        default='/Users/mradulagrawal/shape-generation-sppc/outputs',
        help='Directory path to save the data')
    argparser.add_argument(
        '--bs',
        default=16,
        type=int,
        help='Training batch size')
    argparser.add_argument(
        '--glr',
        default=0.0025,
        type=float,
        help='Learning rate for generator')
    argparser.add_argument(
        '--dlr',
        default=0.0001,
        type=float,
        help='Learning rate for discriminator')
    argparser.add_argument(
        '--epoch',
        default=1000,
        type=int,
        help='Total number of Epochs')
    argparser.add_argument(
        '--wandb',
        default=False,
        type=bool,
        help='Option for WandB logging')
    
    args = argparser.parse_args()

    if args.mode == 'process_data':
        process_data(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
    else:
        print("Wrong mode provided")

if __name__ == '__main__':
    main()
