import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from datasets import load_dataset
from torchmetrics.image import PeakSignalNoiseRatio
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import random

from Div2K import Div2K_Dataset

class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)

        return x
    
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])

root_dir = os.path.dirname(os.path.abspath(__file__))
train_dataset = Div2K_Dataset(root_dir, 'train', scale = 4, transform=transform)
validation_dataset = Div2K_Dataset(root_dir, 'valid', scale = 4, transform=transform)

train_subset = Subset(train_dataset, list(range(85)))

print(len(train_subset))

val_subset = Subset(validation_dataset, list(range(15)))

train_subset, test_dataset = random_split(train_subset, [70, 15])

# Split 70 - 15 - 15

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

learning_rate = 0.001
epochs = 100

model = SRCNN()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
test_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []

psnr = PeakSignalNoiseRatio()

num_batches = 0

# Training Loop

for epoch in range(epochs):
    model.train()
    train_loss = 0
    total = 0

    for images, labels in train_loader:

        num_batches += 1
        print(num_batches)
        #print(labels.shape)

        labels = labels.squeeze().float()
        optimizer.zero_grad() # Reset Gradients (NOT WEIGHTS)

        outputs = model(images).float() # Forward Pass
        
        #print(outputs.shape)
        #print(labels.shape)

        loss = criterion(outputs, labels) # Loss function
        loss.backward() # Backward propagation
        optimizer.step() # Update weights

        train_loss += loss.item()
        total += labels.size(0)

        psnr_score = psnr(outputs, labels)
        train_psnr = psnr_score.item()

    train_losses.append(train_loss / len(train_loader))

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, PSNR: {train_psnr:.4f}')

    # Validation Loop

    model.eval()
    val_loss = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:

            labels = labels.squeeze().float()

            outputs = model(images) # Forward Pass
            loss = criterion(outputs, labels) # Loss Function
            
            # Track Validation Loss
            val_loss += loss.item()
            psnr_score = psnr(outputs, labels)
            val_psnr = psnr_score.item()

    val_losses.append(val_loss / len(val_loader))

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {val_loss/len(val_loader):.4f}, PSNR: {val_psnr:.4f}')

# Testing Loop

"""

model.eval()

total_loss = 0
test_loss = 0
test_psnr = 0
total = 0

with torch.no_grad():

    for images, labels in test_dataset:

        labels = labels.squeeze().float()

        output = model(images) # Forward Pass
        loss = criterion(outputs, labels) # Loss

        # Track Testing Loss
        test_loss += loss.item()
        psnr_score = psnr(outputs, labels)
        test_psnr += psnr_score.item()

    test_losses.append(test_loss / len(val_loader))

print(f'Epoch [{epoch+1}/{epochs}], Loss: {test_loss/len(val_loader):.4f}, PSNR: {test_psnr:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss
axes[0].plot(epochs, train_losses, label="Training Loss", marker='o', color = 'blue')
axes[0].plot(epochs, val_losses, label="Validation Loss", marker='o', color = 'orange')
axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Plot Accuracy
axes[1].plot(epochs, train_accuracies, label="Training Accuracy", marker='o', color = 'blue')
axes[1].plot(epochs, val_accuracies, label="Validation Accuracy", marker='o', color = 'orange')
axes[1].set_title("Accuracy vs Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

# Adjust spacing between plots
plt.tight_layout()
plt.show()"
"""