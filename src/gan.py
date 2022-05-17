from cv2 import transform
from jinja2 import pass_environment
from torchvision import transforms
import torch 
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
        x = self.act4(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_input, num_output):
        super(Discriminator, self).__init__()
        self.in_feat = num_input
        self.out_feat = num_output

        self.fc1 = nn.Linear(self.in_feat, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.act1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(100, 100)
        self.batch_norm2 = nn.BatchNorm1d(100)
        self.act2 = nn.LeakyReLU()

        self.fc3 = nn.Linear(100, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)
        self.act3 = nn.LeakyReLU()

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
        # Normalised the data between (0,1) for the input to the discriminator
        mean, std = torch.mean(output), torch.std(output)
        output = (output - mean) / std
        return output

class GAN():
    def __init__(self, args):
        self.args = args
        self.num_feat = 100
        self.discriminator_lr = self.args.dlr
        self.generator_lr = self.args.glr
        self.num_epoch = self.args.epoch
        self.batch_size = self.args.bs

        self.train_data_path = self.args.load_path
        self.model_save_dir = self.args.save_path

        self.generator = Generator(self.num_feat, self.num_feat)
        self.discriminator = Discriminator(self.num_feat, self.num_feat)

        self.train_dataset = TrainingDataset(self.train_data_path)
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.generator_lr)
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), self.discriminator_lr)

        self.g_criterion = nn.MSELoss()
        self.d_criterion = nn.BCELoss()

        self.test_noise = self.generate_noise(16)

    def train_discriminator(self, fake_data, real_data):
        self.d_optim.zero_grad()

        prediction_real, f_3, f_2, f_1 = self.discriminator(real_data) # (batch size, num feat)
        loss_real = self.d_criterion(prediction_real, torch.ones((real_data.shape[0], self.num_feat)))
        loss_real.backward()

        prediction_fake, r_3, r_2, r_1 = self.discriminator(fake_data) # (batch size, num feat)
        loss_fake = self.d_criterion(prediction_fake, torch.zeros((fake_data.shape[0], self.num_feat)))
        loss_fake.backward()

        self.d_optim.step()
        return (loss_real.item() + loss_fake.item()) / 2

    def train_generator(self, fake_data, real_data):
        self.g_optim.zero_grad()

        prediction_fake, f_3, f_2, f_1 = self.discriminator(fake_data) # (batch size, num feat)
        prediction_real, r_3, r_2, r_1 = self.discriminator(real_data) # (batch size, num feat)
        
        intermediate_act_fake = prediction_fake + f_3/2 + f_2/4 + f_1/8
        intermediate_act_real = prediction_real + r_3/2 + r_2/4 + r_1/8

        loss_E = self.g_criterion(intermediate_act_fake, intermediate_act_real)
        loss_C = self.g_criterion(torch.var(intermediate_act_fake, 0), torch.var(intermediate_act_real, 0))

        loss = loss_E + loss_C
        loss.backward()

        self.g_optim.step()
        return loss.item()

    def train(self):    
        self.discriminator.train()
        self.generator.train()

        for epoch in range(5):
            avg_g_loss = 0
            avg_d_loss = 0
            for batch in self.train_loader:
                data = batch.float() # (batch size, num feat)

                # First Train the Discriminator with fake data generated and real data present
                fake_data = self.generator(self.generate_noise(data.shape[0])) # (batch size, num feat)
                real_data = data # (batch size, num feat)
                d_loss = self.train_discriminator(fake_data, real_data)
                avg_d_loss += d_loss

                # Second, Train the Generator with another fake data
                fake_data = self.generator(self.generate_noise(data.shape[0]))
                g_loss = self.train_generator(fake_data, real_data)
                avg_g_loss += g_loss

            print("For Epoch: ", epoch)
            print("Generator Loss: ", avg_g_loss/len(self.train_loader))
            print("Discriminator Loss: ", avg_d_loss/len(self.train_loader))



    def save_model(self):
        torch.save(self.generator.state_dict(), os.path.join(self.model_save_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_save_dir, 'discriminator.pth'))

    def generate_noise(self, num_points: int):
        noise = torch.randn((num_points, self.num_feat))
        return noise