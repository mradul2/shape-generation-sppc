import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from cv2 import transform
from jinja2 import pass_environment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Generator(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator, self).__init__()
        self.in_feat = num_input
        self.out_feat = num_output

        self.fc1 = nn.Linear(self.in_feat, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(100, 100)
        self.batch_norm2 = nn.BatchNorm1d(100)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(100, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(100, self.out_feat)
        self.act4 = nn.Tanh()

    def forward(self, input):
        x = self.fc1(input)
        x = self.batch_norm1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.act3(x)

        x = self.fc4(x)
        # No avtivation function
        # x = self.act4(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_input, num_output):
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
        x_1 = self.fc1(input)
        x_1 = self.batch_norm1(x_1)
        x_1 = self.act1(x_1)

        x_2 = self.fc2(x_1)
        x_2 = self.batch_norm2(x_2)
        x_2 = self.act2(x_2)

        x_3 = self.fc3(x_2)
        x_3 = self.batch_norm3(x_3)
        x_3 = self.act3(x_3)

        x_4 = self.fc4(x_3)
        x_4 = self.act4(x_4)

        return x_4, x_3, x_2, x_1

class TrainingDataset(Dataset):
    def __init__(self, npy_file_path: str):
        self.npy_file_path = npy_file_path
        self.encodings = np.load(self.npy_file_path) # (5000, 100)
    def __len__(self):
        return self.encodings.shape[0];
    def __getitem__(self, index):
        output = self.encodings[index] # (100,)
        # Convert the numpy array to torch tensor
        output = torch.from_numpy(output)
        # Normalised the data between (-1,1) for the input to the discriminator
        # To normalise the data betweem (-1, 1)
        # output = nn.Tanh()(output)
        return output

class GAN():
    def __init__(self, args):
        self.args = args
        self.num_feat = 100
        self.discriminator_lr = self.args.dlr
        self.generator_lr = self.args.glr
        self.num_epoch = self.args.epoch
        self.batch_size = self.args.bs

        if args.wandb:
            wandb.init(project="shape-gan")

        self.device = 'cpu'
        self.is_cuda_available = torch.cuda.is_available()
        if self.is_cuda_available:
            self.device = 'cuda'

        self.train_data_path = self.args.load_path
        self.pretrained_path = self.args.load_path
        self.model_save_dir = self.args.save_path

        self.generator = Generator(self.num_feat, self.num_feat).to(self.device)
        self.discriminator = Discriminator(self.num_feat, self.num_feat).to(self.device)

        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.generator_lr)
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), self.discriminator_lr)

        self.g_criterion = nn.MSELoss()
        self.d_criterion = nn.BCELoss()

        self.g_acc = 0.0
        self.d_acc = 0.0

    def train_discriminator(self, fake_data, real_data):
        self.d_optim.zero_grad()

        prediction_real, f_3, f_2, f_1 = self.discriminator(real_data) # (batch size, num feat)
        loss_real = self.d_criterion(prediction_real, torch.ones((real_data.shape[0], self.num_feat)).to(self.device))

        prediction_fake, r_3, r_2, r_1 = self.discriminator(fake_data) # (batch size, num feat)
        loss_fake = self.d_criterion(prediction_fake, torch.zeros((fake_data.shape[0], self.num_feat)).to(self.device))

        loss = loss_real + loss_fake
        loss.backward()

        self.d_optim.step()
        return loss.item()

    def acc_discriminator(self, real_data, fake_data):
        with torch.no_grad():
            prediction_real, f_3, f_2, f_1 = self.discriminator(real_data) # (batch size, num feat)
            acc_real = (prediction_real > 0.5).sum() / (real_data.shape[0] * self.num_feat)
            prediction_fake, r_3, r_2, r_1 = self.discriminator(fake_data) # (batch size, num feat)
            acc_fake = (prediction_fake < 0.5).sum() / (fake_data.shape[0] * self.num_feat)
            self.d_acc = (acc_fake + acc_real) / 2

    def train_generator(self, fake_data, real_data):
        self.g_optim.zero_grad()

        prediction_fake, f_3, f_2, f_1 = self.discriminator(fake_data) # (batch size, num feat)
        prediction_real, r_3, r_2, r_1 = self.discriminator(real_data) # (batch size, num feat)
        
        intermediate_act_fake = prediction_fake + f_3/2 + f_2/4 + f_1/8
        intermediate_act_real = prediction_real + r_3/2 + r_2/4 + r_1/8

        # Loss function mentioned in the paper which uses intermediate activations of the generator
        # And calculate the L2 norm of mean and variance between them
        loss_E = self.g_criterion(intermediate_act_fake, intermediate_act_real)
        loss_C = self.g_criterion(torch.var(intermediate_act_fake, 0), torch.var(intermediate_act_real, 0))

        loss = loss_E + loss_C

        # Vanilla loss for generator:
        # loss = self.d_criterion(prediction_fake, torch.ones((fake_data.shape[0], self.num_feat)).to(self.device))
       
        loss.backward()

        self.g_optim.step()
        return loss.item()

    def acc_generator(self, fake_data):
        with torch.no_grad():
            prediction_fake, f_3, f_2, f_1 = self.discriminator(fake_data)
            acc = (prediction_fake > 0.5).sum() / (fake_data.shape[0] * self.num_feat)
            self.g_acc = acc

    def train(self):    
        self.train_dataset = TrainingDataset(self.train_data_path)
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        self.discriminator.train()
        self.generator.train()

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
                if self.g_acc < 0.8:
                    d_loss = self.train_discriminator(fake_data.detach(), real_data)
                    avg_d_loss += d_loss
                    prev_d_loss = d_loss
                else:
                    avg_d_loss += prev_d_loss
                # Discriminator training step completed

                # Second, Train the Generator with another fake data
                g_loss = self.train_generator(fake_data, real_data)
                avg_g_loss += g_loss
                # Generator training step completed

                # Update accuracy
                self.acc_discriminator(real_data, fake_data)
                self.acc_generator(fake_data)


            print("For Epoch: ", epoch)
            print("Generator Loss: ", avg_g_loss/len(self.train_loader))
            print("Discriminator Loss: ", avg_d_loss/len(self.train_loader))
            print("Discriminator Accuracy: ", self.d_acc)
            print("Generator Accuracy: ", self.g_acc)

            if self.args.wandb:
                wandb.log({
                    "Epoch": epoch,
                    "Generator Loss": avg_g_loss/len(self.train_loader),
                    "Discriminator Loss": avg_d_loss/len(self.train_loader),
                    "Generator Accuracy": self.g_acc,
                    "Discriminator Accuracy": self.d_acc
                })

    def save_model(self):
        torch.save(self.generator.state_dict(), os.path.join(self.model_save_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_save_dir, 'discriminator.pth'))

    def generate_noise(self, num_points: int):
        noise = torch.randn((num_points, self.num_feat))
        return noise

    def load_weights(self):
        g_model_file = os.path.join(self.pretrained_path, 'generator.pth')

        if self.device == 'cpu':
            self.generator.load_state_dict(torch.load(g_model_file, map_location=torch.device('cpu')))
        else:
            self.generator.load_state_dict(torch.load(g_model_file))

    def generate_output(self, num: int):
        noise = self.generate_noise(num).to(self.device)
        fake_data = self.generator(noise)
        return fake_data.detach().numpy()