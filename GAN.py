import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

from Div2K import Div2K_Dataset

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1 , padding=1)
        self.relu1 = nn.PReLU()

        # Residual Block

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffler = nn.PixelShuffle(upscale_factor=2)
        self.relu3 = nn.PReLU()
        
        self.conv6 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffler2 = nn.PixelShuffle(upscale_factor=2)
        self.relu4 = nn.PReLU()

        self.conv7 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        residual = x

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn2(x)

        x = x + residual

        x = self.conv4(x)
        x = self.bn3(x)

        x = x + residual

        x = self.conv5(x)
        x = self.pixel_shuffler(x)
        x = self.relu3(x)

        x = self.conv6(x)
        x = self.pixel_shuffler2(x)
        x = self.relu4(x)

        x = self.conv7(x)

        return x

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()

        self.fc1 = nn.Linear(64, 128)
        self.relu3 = nn.LeakyReLU()

        self.fc2 = nn.Linear(128, 1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        x = x.view(-1, 64) # Negative one so we can vary the batch size

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.sigmoid1(x)
        
        return x

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])

root_dir = os.path.dirname(os.path.abspath(__file__))
train_dataset = Div2K_Dataset(root_dir, 'train', scale = 4, transform=transform)
validation_dataset = Div2K_Dataset(root_dir, 'valid', scale = 4, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)

generator =  Generator()
discriminator = Discriminator()

loss_function = nn.BCELoss()

generator_optim = optim.Adam(generator.parameters(), lr=0.001)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=0.001)

total_d_loss = 0
total_g_loss = 0

epochs = 100

for epoch in range(epochs):
    for lr_images, hr_images in train_loader:

        # Generate fake images once
        fake_images = generator(lr_images)

        # ------------ Train Discriminator -------------

        discriminator_optim.zero_grad()

        # Real HR Images should be classified as REAL (1)

        real_pred = discriminator(hr_images)
        real_labels = torch.ones(real_pred.size(0), 1).float()
        real_loss = loss_function(real_pred, real_labels)

        # Generated images should be classified as FAKE (0)

        fake_pred = discriminator(fake_images.detach())
        fake_labels = torch.zeros(fake_pred.size(0), 1).float()
        fake_loss = loss_function(fake_pred, fake_labels)

        d_loss = real_loss + fake_loss # Discriminator Loss
        d_loss.backward() # Backward Propagation
        discriminator_optim.step() # Optimizer step

        # ----------- Train Generator ------------

        generator_optim.zero_grad()

        # Fool the discriminator

        gen_pred = discriminator(fake_images)
        gen_labels = torch.zeros(gen_pred.size(0), 1).float()

        gen_loss = loss_function(gen_pred, gen_labels) 
        gen_loss.backward()
        generator_optim.step()

        total_d_loss += d_loss
        total_g_loss += gen_loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {total_d_loss:.4f} | G Loss: {total_g_loss:.4f}")