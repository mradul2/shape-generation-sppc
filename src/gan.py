import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Generator(nn.Module):
    """
        Class for the Generator Network
    """
    def __init__(self, num_input, num_output):
        """
            Initialize the Generator Network

            Args:
                num_input: Number of input features
                num_output: Number of output features
        """
        super(Generator, self).__init__()
        self.in_feat = num_input
        self.out_feat = num_output

        self.fc1 = nn.Linear(self.in_feat, 100)
        self.act1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(100)

        self.fc3 = nn.Linear(100, 100)
        self.act3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(100)

        self.fc4 = nn.Linear(100, self.out_feat)
        self.act4 = nn.Tanh()

    def forward(self, input):
        """
            Forward pass of the Generator Network

            Args:
                input: Input features

            Returns:
                output: Output features
        """
        x = self.fc1(input)
        x = self.act1(x)
        x = self.batch_norm1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.batch_norm2(x)

        x = self.fc3(x)
        x = self.act3(x)
        x = self.batch_norm3(x)

        x = self.fc4(x)
        output = self.act4(x)

        return output

class Discriminator(nn.Module):
    """
        Class for the Discriminator Network
    """
    def __init__(self, num_input, num_output):
        """
            Function to initialize the Discriminator Network

            Args:
                num_input: Number of input features
                num_output: Number of output features
        """
        super(Discriminator, self).__init__()
        self.in_feat = num_input
        self.out_feat = num_output

        self.fc1 = nn.Linear(self.in_feat, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.act1 = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(100, 100)
        self.batch_norm2 = nn.BatchNorm1d(100)
        self.act2 = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(100, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)
        self.act3 = nn.LeakyReLU(0.2)

        self.fc4 = nn.Linear(100, self.out_feat)
        self.act4 = nn.Sigmoid()

    def forward(self, input):
        """
            Function to forward pass the Discriminator Network

            Args:
                input: Input features

            Returns:
                output: Intermediate features and Output features
        """
        x_1 = self.fc1(input)
        x_1_a = self.act1(x_1)
        x_1 = self.batch_norm1(x_1_a)

        x_2 = self.fc2(x_1)
        x_2_a = self.act2(x_2)
        x_2 = self.batch_norm2(x_2_a)

        x_3 = self.fc3(x_2)
        x_3_a = self.act3(x_3)
        x_3 = self.batch_norm3(x_3_a)

        x_4 = self.fc4(x_3)
        x_4_a = self.act4(x_4)

        feature = x_3_a
        output = x_4_a
        return feature, output

class TrainingDataset(Dataset):
    """
        Class for the training dataset
    """
    def __init__(self, npy_file_path: str):
        """
            Function to initialize the training dataset

            Args:
                npy_file_path: Path to the npy file containing the training data
        """
        self.npy_file_path = npy_file_path
        self.encodings = np.load(self.npy_file_path) # (5000, 100)
    def __len__(self):
        """
            Function to return the length of the dataset
        """
        return self.encodings.shape[0]
    def __getitem__(self, index):
        """
            Function to return the item at the given index

            Args:
                index: Index of the item to return

            Returns:
                item: Item at the given index
        """
        output = self.encodings[index] # (100,)
        # Convert the numpy array to torch tensor
        output = torch.from_numpy(output)
        # output = (output - torch.mean(output)) / torch.std(output)
        # Normalised the data between (-1,1) for the input to the discriminator
        # To normalise the data betweem (-1, 1)
        output = (2 * (output - torch.min(output)) / (torch.max(output) - torch.min(output))) - 1
        return output

class GAN():
    """
        GAN Class for training the GAN, saving the model and generating data
    """
    def __init__(self, args):
        """
            Initialize the GAN class

            Args:
                args: Command line arguments
        """
        self.args = args

        # Training parameters
        self.num_feat = 100
        self.discriminator_lr = self.args.dlr
        self.generator_lr = self.args.glr
        self.num_epoch = self.args.epoch
        self.batch_size = self.args.bs

        # If WandB logging is enabled, then initialize the wandb run
        if args.wandb:
            wandb.init(project="shape-gan")

        # Set torch.device to cuda if available
        self.device = 'cpu'
        self.is_cuda_available = torch.cuda.is_available()
        if self.is_cuda_available:
            self.device = 'cuda'                                                                                                                

        # Various paths for training, saving and generating data
        self.train_data_path = self.args.load_path
        self.pretrained_path = self.args.load_path
        self.model_save_dir = self.args.save_path

        # Create the model objects for the discriminator and generator
        self.generator = Generator(self.num_feat, self.num_feat).to(self.device)
        self.discriminator = Discriminator(self.num_feat, self.num_feat).to(self.device)

        # Adam optimizers for both discriminator and generator
        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.generator_lr)
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), self.discriminator_lr)

    def train_discriminator(self, fake_data, real_data):
        """
            Function to train the discriminator network

            Args:
                fake_data: Fake data generated by the generator
                real_data: Real data from the dataset

            Returns:    
                loss: Loss of the discriminator
        """
        self.discriminator.train()
        self.d_optim.zero_grad()

        feature_real, prediction_real = self.discriminator(real_data) # (batch size, num feat)
        loss_real = nn.BCELoss()(prediction_real, torch.ones((real_data.shape[0], self.num_feat)).to(self.device))

        feature_fake, prediction_fake = self.discriminator(fake_data) # (batch size, num feat)
        loss_fake = nn.BCELoss()(prediction_fake, torch.zeros((fake_data.shape[0], self.num_feat)).to(self.device))

        loss = loss_real + loss_fake
        loss.backward()

        self.d_optim.step()
        return loss.item()

    def acc_discriminator(self, real_data, fake_data):
        """
            Function to calculate the accuracy of the discriminator

            Args:
                real_data: Real data
                fake_data: Fake data

            Returns:
                acc: Accuracy of the discriminator
        """
        with torch.no_grad():
            self.discriminator.eval()
            feature_real, prediction_real = self.discriminator(real_data) # (batch size, num feat)
            acc_real = (prediction_real > 0.5).sum() / (real_data.shape[0] * self.num_feat)
            feature_fake, prediction_fake = self.discriminator(fake_data) # (batch size, num feat)
            acc_fake = (prediction_fake < 0.5).sum() / (fake_data.shape[0] * self.num_feat)
            return acc_real, acc_fake

    def train_generator(self, fake_data, real_data):
        """
            Function to train the generator

            Args:
                fake_data: (torch.tensor) The fake data generated by the generator
                real_data: (torch.tensor) The real data obtained from the dataset
            
            Returns:
                loss: (float) The loss of the generator
        """
        self.generator.train()
        self.g_optim.zero_grad()

        feature_fake, prediction_fake = self.discriminator(fake_data) # (batch size, num feat)
        feature_real, prediction_real = self.discriminator(real_data) # (batch size, num feat)
        
        intermediate_act_fake = feature_fake
        intermediate_act_real = feature_real

        # Loss function mentioned in the paper which uses intermediate activations of the generator
        # And calculate the L2 norm of mean and variance between them
        loss_E = nn.MSELoss()(torch.mean(intermediate_act_fake, 0), torch.mean(intermediate_act_real, 0))
        loss_C = nn.MSELoss()(torch.var(intermediate_act_fake, 0), torch.var(intermediate_act_real, 0))

        loss = loss_E + loss_C

        # Vanilla loss for generator:
        # loss = nn.BCELoss()(prediction_fake, torch.ones((fake_data.shape[0], self.num_feat)).to(self.device))
       
        loss.backward()

        self.g_optim.step()
        return loss.item()

    def acc_generator(self, fake_data):
        """
            Function to calculate the accuracy of the generator

            Args:
                fake_data: The generated data by the generator

            Returns:   
                acc: The accuracy of the generator
        """
        with torch.no_grad():
            self.generator.eval()
            # Obtain the prediction from the generator
            feature_fake, prediction_fake = self.discriminator(fake_data)
            # Calculate the accuracy of that prediction to be true
            acc = (prediction_fake > 0.5).sum() / (fake_data.shape[0] * self.num_feat)
            return acc

    def train(self):    
        """
            Function to train the GAN
        """
        # Create the training dataset and dataloader
        self.train_dataset = TrainingDataset(self.train_data_path)
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        self.discriminator.train()
        self.generator.train()

        # Main training loop
        for epoch in tqdm(range(self.num_epoch)):
            avg_g_loss = 0
            avg_d_loss = 0
            prev_d_loss = 0
            for batch in self.train_loader:
                data = batch.float().to(self.device) # (batch size, num feat)
                noise = self.generate_noise(data.shape[0]).to(self.device)

                # First Train the Discriminator with fake data generated and real data present
                fake_data = self.generator(noise) # (batch size, num feat)
                real_data = data # (batch size, num feat)
                
                # According to the paper, discriminator is trained only when the accuracy is below 80%
                if acc_g < 0.8:
                    d_loss = self.train_discriminator(fake_data.detach(), real_data)
                    avg_d_loss += d_loss
                    prev_d_loss = d_loss
                else:
                    avg_d_loss += prev_d_loss
                # Discriminator training step completed

                # Second, Train the Generator with fake data
                g_loss = self.train_generator(fake_data, real_data)
                avg_g_loss += g_loss
                # Generator training step completed

                # Update accuracy
                acc_d_real, acc_dfake = self.acc_discriminator(real_data, fake_data)
                acc_g = self.acc_generator(fake_data)

            print("For Epoch: ", epoch)
            print("Generator Loss: ", avg_g_loss/len(self.train_loader))
            print("Discriminator Loss: ", avg_d_loss/len(self.train_loader))
            print("Discriminator Accuracy Real: ", acc_d_real)
            print("Discriminator Accuracy Fake: ", acc_dfake)
            print("Generator Accuracy: ", acc_g)

            # If WanbB logging is enabled, then log the various training values on wandb
            if self.args.wandb:
                wandb.log({
                    "Epoch": epoch,
                    "Generator Loss": avg_g_loss/len(self.train_loader),
                    "Discriminator Loss": avg_d_loss/len(self.train_loader),
                    "Generator Accuracy": self.g_acc,
                    "Discriminator Accuracy Real": self.d_acc[0],
                    "Discriminator Accuracy Fake": self.d_acc[1]
                })

    def save_model(self):
        """
            Function to save the model checkpoints
        """
        # Save the state dict of the generator and discriminator at model_save_dir directory
        torch.save(self.generator.state_dict(), os.path.join(self.model_save_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_save_dir, 'discriminator.pth'))

    def generate_noise(self, num_points: int):
        """
            Function to genrate a random noise vector of size num_points
            from of uniform distribution of (0, 1)

            Args:
                num_points: Number of data points to generate

            Returns:
                noise: A tensor of size (num_points, num_feat)
        """
        # torch.rand produces random numbers from a uniform distribution of (0, 1)
        # Generate noise from the uniform distribution of (0,1) and then scale it to (-1, 1)
        noise = (-2 * torch.rand((num_points, self.num_feat))) + 1
        return noise

    def load_weights(self):
        """
            Function to load the pretrained weights of the generator into the model
        """
        # Pretrained weights file (.pth) of the generator
        g_model_file = os.path.join(self.pretrained_path, 'generator.pth')

        # Load the pretrained weights of generator according to the device
        if self.device == 'cpu':
            self.generator.load_state_dict(torch.load(g_model_file, map_location=torch.device('cpu')))
        else:
            self.generator.load_state_dict(torch.load(g_model_file))

    def generate_output(self, num: int):
        """
            Function to generate output from the trained model

            Args:
                num: Number of point clouds to be generated

            Returns:
                Generated point cloud vectors from the trained generator model
        """
        # Generate noise from the uniform distribution of (-1,1)
        noise = self.generate_noise(num).to(self.device)
        # Generate output from the trained generator model
        fake_data = self.generator(noise)
        # Shape of the returned data is (num, num_feat)!
        # Which will be later inverse transformed to get the original point cloud
        return fake_data.detach().numpy()