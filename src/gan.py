import torch 
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_input):
        super(Generator, self).__init__
        self.fc1 = nn.Linear(num_input, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(100, 100)
        self.batch_norm2 = nn.BatchNorm1d(100)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(100, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(100, 100)
        self.act4 = nn.Sigmoid()

    def forward(self, input):
        x = self.fc1(input)
        x = self.batch_norm1(x)
        x = self.act1(x)

        x = self.fc2(input)
        x = self.batch_norm2(x)
        x = self.act2(x)

        x = self.fc3(input)
        x = self.batch_norm3(x)
        x = self.act3(x)

        x = self.fc4(input)
        x = self.act4(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_input):
        super(Discriminator, self).__init__
        self.fc1 = nn.Linear(num_input, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.act1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(100, 100)
        self.batch_norm2 = nn.BatchNorm1d(100)
        self.act2 = nn.LeakyReLU()

        self.fc3 = nn.Linear(100, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)
        self.act3 = nn.LeakyReLU()

        self.fc4 = nn.Linear(100, 100)
        self.act4 = nn.Sigmoid()

    def forward(self, input):
        x = self.fc1(input)
        x = self.batch_norm1(x)
        x = self.act1(x)

        x = self.fc2(input)
        x = self.batch_norm2(x)
        x = self.act2(x)

        x = self.fc3(input)
        x = self.batch_norm3(x)
        x = self.act3(x)

        x = self.fc4(input)
        x = self.act4(x)

        return x

