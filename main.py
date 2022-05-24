import argparse
import os
import pickle

import numpy as np
import wandb

from src.gan import GAN
from src.kd_tree import KdTree
from src.pc_data import PCD
from src.pca import PCA_
from src.utils import visualise_point_cloud, visualise_point_cloud_gradient


def process_data(args):
    """
        Function to take the point clouds as input and save the 
        transformed points along with the PCA parameters. 

        Apply KD Tree sorting to the point clouds followed by 
        iterative point ordering. 

        Args:
            args: Arguments passed from the command line

        Returns:
            None
    """
    # List of all the point cloud files
    root_dir_path = args.load_path
    list_of_files = os.listdir(root_dir_path)
    print("Number of Shapes: ", len(list_of_files))
    
    # Create the PCD object for all the point clouds and sort them using KD Tree
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

    # Create the PCA object for the point clouds for shape basis = 100
    pca = PCA_(size_basis=100, num_data=matrix_np.shape[0])
    # Fit the PCA object on the point clouds
    pca.fit_once(matrix_np)

    # Visualize some point clouds
    visualise_point_cloud_gradient(matrix_np[0])
    output = pca.transform_data(matrix_np)
    output = pca.inverse_transform_data(output)
    visualise_point_cloud_gradient(output[0])

    # If the point ordering option is selected, perform the point ordering
    if args.point_ordering: 
        print("Optimizing Point Ordering...")
        # Values of I and K according to the paper
        I = 1000 
        K = 10000
        # For I number of iterations:
        for i in range(I):
            print("Iteration: ", i)
            # For K times, perform the point ordering
            avg_recon_error, matrix_np = pca.optimize_point_ordering(K, matrix_np)
            # After every shape is processed, recompute the PCA basis
            pca.fit_once(matrix_np)
        print("Point ordering completed")

    # Save the PCA parameters as a pickle file
    save_path = os.path.join(args.save_path, ("pca.pkl"))
    pickle.dump(pca, open(save_path, "wb"))
    print("Pickle object saved as: ", save_path)

    # Save the transformed point clouds as a numpy file
    final_matrix = pca.transform_data(matrix_np)
    save_path = os.path.join(args.save_path, 'processed_data.npy')
    np.save(save_path, final_matrix)
    print("Processed data saved as: ", save_path)
    

def train(args):
    """
        Function to take the transformed point cloud data 
        as input and train a GAN using it

        Args:
            args: Arguments passed from the command line

        Returns:    
            None
    """
    # Create the GAN object
    gan = GAN(args)
    # Call the train functio
    gan.train()
    # Save the model weights
    gan.save_model()

def generate(args):
    """
        Function to take the trained Generator network and PCA
        parameters as input and genrate num point clouds using the 
        Generator's output and pca's inverse transform

        Args:
            args: Arguments passed from the command line

        Returns:
            None
    """
    # Load the trained GAN
    trained_gan = GAN(args)
    trained_gan.load_weights()

    # Generate num number of point clouds using the trained GAN
    num = 10
    gan_output = trained_gan.generate_output(num)

    # Load the PCA parameters from the input directory
    pca_file_path = os.path.join(args.load_path, 'pca.pkl')
    pca = pickle.load(open(pca_file_path, "rb"))

    # Inverse transform the generated point clouds using GAN data and PCA
    output = pca.inverse_transform_data(gan_output)
    output = output.reshape(output.shape[0], 1000, 3)

    # Visualize those generated point clouds
    for i in range(num):
        visualise_point_cloud(output[i])


def main():
    """
        Main function to take the arguments from the command line
        and call the appropriate functions    
    """

    # Parse the arguments
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
        '--point_ordering',
        type=bool,
        default=False,
        help='Option to perform the point ordering')
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

    # According to the mode, call the appropriate function
    if args.mode == 'process_data':
        print("Processing data...")
        process_data(args)
    elif args.mode == 'train':
        print("Training GAN...")
        train(args)
    elif args.mode == 'generate':
        print("Generating point clouds using trained GAN...")
        generate(args)
    else:
        print("Wrong mode provided")

if __name__ == '__main__':
    main()