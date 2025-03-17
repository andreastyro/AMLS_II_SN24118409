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

    # Define the save directory


root_dir = os.path.dirname(os.path.abspath(__file__))
train_dataset = Div2K_Dataset(root_dir, 'train', scale = 2, transform=transform)
validation_dataset = Div2K_Dataset(root_dir, 'valid', scale = 2, transform=transform)

train_subset = Subset(train_dataset, list(range(80)))
val_subset = Subset(validation_dataset, list(range(10)))

train_subset, test_dataset = random_split(train_subset, [70, 10])

# Split 70 - 15 - 15

train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

learning_rate = 0.001
epochs = 100

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move it to GPU
model = SRCNN().to(device)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
test_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []

psnr = PeakSignalNoiseRatio().to(device)

num_batches = 0

save_dir = os.path.join(root_dir, "Plots")
image_dir = os.path.join(root_dir, "Images")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Training Loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    total = 0
    count = 0

    for images, labels in train_loader:

        count = count + 1

        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(images)  # Forward pass

        loss = criterion(outputs, labels)  # Loss function
        loss.backward()  # Backward propagation
        optimizer.step()  # Update weights

        train_loss += loss.item()
        total += labels.size(0)

        psnr_score = psnr(outputs, labels)
        train_psnr = psnr_score.item()

        # Save side-by-side images (LR, Generated, HR) for the first batch of each epoch
        if count == 1:
            # Extract the first images from the batch
            lr_image = images[1].cpu().clone().detach()      # Low-resolution input image
            hr_image = labels[1].cpu().clone().detach()        # High-resolution ground truth image
            gen_image = outputs[1].clamp(0, 1).cpu().clone().detach()  # Generated image (clamped to [0, 1])
            
            # Convert tensors to PIL images for visualization
            to_pil = transforms.ToPILImage()
            lr_pil = to_pil(lr_image)
            gen_pil = to_pil(gen_image)
            hr_pil = to_pil(hr_image)
            
            # Plot the images side by side
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].imshow(lr_pil)
            axs[0].set_title("LR Image")
            axs[0].axis('off')
            
            axs[1].imshow(gen_pil)
            axs[1].set_title("Generated Image")
            axs[1].axis('off')
            
            axs[2].imshow(hr_pil)
            axs[2].set_title("HR Image")
            axs[2].axis('off')
            
            # Save the combined image; you can also use plt.show() to display it
            fig.savefig(os.path.join(image_dir, f"epoch_{epoch+1}_comparison.png"))
            plt.close(fig)

    train_losses.append(train_loss / len(train_loader))

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, PSNR: {train_psnr:.4f}')

    # Validation Loop
    model.eval()
    val_loss = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Loss function

            val_loss += loss.item()
            psnr_score = psnr(outputs, labels)
            val_psnr = psnr_score.item()

    val_losses.append(val_loss / len(val_loader))

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {val_loss/len(val_loader):.4f}, PSNR: {val_psnr:.4f}')

# Testing Loop
model.eval()

test_loss = 0
test_psnr = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:  # Use test_loader instead of test_dataset
        images, labels = images.to(device), labels.to(device)  # Move data to GPU

        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Loss

        test_loss += loss.item()
        psnr_score = psnr(output, labels)
        test_psnr += psnr_score.item()

test_losses.append(test_loss / len(test_loader))

print(f'Test Loss: {test_loss/len(test_loader):.4f}, PSNR: {test_psnr:.4f}')

# Plotting the results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss
axes[0].plot(range(epochs), train_losses, label="Training Loss", marker='o', color = 'blue')
axes[0].plot(range(epochs), val_losses, label="Validation Loss", marker='o', color = 'orange')
axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Plot PSNR
axes[1].plot(range(epochs), train_psnr, label="Training PSNR", marker='o', color = 'blue')
axes[1].plot(range(epochs), val_psnr, label="Validation PSNR", marker='o', color = 'orange')
axes[1].set_title("PSNR vs Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("PSNR")
axes[1].legend()

fig.savefig(os.path.join(save_dir, "Loss, PSNR.png"))
plt.close(fig)

# Adjust spacing between plots
#plt.tight_layout()
#plt.show()
