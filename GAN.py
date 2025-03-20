import os
import pandas as pd
import pynvml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split, Subset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, models
from torchvision.models import vgg19
from pytorch_msssim import ms_ssim

from Div2K import Div2K_Dataset
from SR_Loss import EnhancedSRLoss

scale = 2

class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
  
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual

        return out

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1 , padding=1)
        self.relu1 = nn.PReLU()

        self.res_blocks = nn.ModuleList([Residual_Block() for i in range(16)])

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffler1 = nn.PixelShuffle(upscale_factor=2)
        self.relu2 = nn.PReLU()

        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffler2 = nn.PixelShuffle(upscale_factor=2)
        self.relu3 = nn.PReLU()

        self.conv5 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        residual = x

        for block in self.res_blocks:
            x = block(x)
        
        x = self.conv2(x)
        x = self.bn1(x)

        x = x + residual

        x = self.conv3(x)
        x = self.pixel_shuffler1(x)
        x = self.relu2(x)

        if scale == 4:
            x = self.conv4(x)
            x = self.pixel_shuffler2(x)
            x = self.relu3(x)

        x = self.conv5(x)

        return x

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu8 = nn.LeakyReLU()

        self.fc1 = nn.Linear(512, 1024)
        self.relu9 = nn.LeakyReLU()

        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn4(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn5(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bn6(x)
        x = self.relu7(x)
        
        x = self.conv8(x)
        x = self.bn7(x)
        x = self.relu8(x)

        x = x.view(-1, 512)

        x = self.fc1(x)
        x = self.relu9(x) 

        x = self.fc2(x)
        x = self.sigmoid1(x)
        
        return x

# Check if CUDA is available and initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained VGG19 model (feature extraction only)
vgg = models.vgg19(pretrained=True).features
vgg.eval()  # Set to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Define VGG-based feature extractor
class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg, layers=7):  # Extract up to the first 7 layers
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = nn.Sequential(*list(vgg[:layers]))  # Extract layers

    def forward(self, x):
        return self.vgg(x)

# Initialize the VGG feature extractor
vgg_features = VGGFeatureExtractor(vgg, layers=7).to(device)

# Define VGG-based perceptual loss (MSE Loss)
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, gen_out, hr_images):
        # Normalize input (if needed)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        gen_out = (gen_out - mean) / std
        hr_images = (hr_images - mean) / std

        # Compute feature maps using VGG
        gen_features = vgg_features(gen_out)
        hr_features = vgg_features(hr_images)

        # Compute loss
        loss = self.criterion(gen_features, hr_features)
        return loss

# Initialize the VGG loss function
vgg_loss = VGGPerceptualLoss().to(device)

# Initialize models and move them to device
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Set up loss function and optimizers
adversarial_loss_function = nn.BCELoss()
content_loss_function = nn.MSELoss()
#content_loss_function = VGGPerceptualLoss().to(device)
#enhanced_loss = EnhancedSRLoss(device).to(device)

generator_optim = optim.Adam(generator.parameters(), lr=0.001)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])

root_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(root_dir, "Plots")
image_dir = os.path.join(root_dir, "Images")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

train_dataset = Div2K_Dataset(root_dir, 'train', scale=scale, transform=transform)
validation_dataset = Div2K_Dataset(root_dir, 'valid', scale=scale, transform=transform)

train_subset = Subset(train_dataset, list(range(450)))
val_subset = Subset(validation_dataset, list(range(50)))
train_subset, test_subset = random_split(train_subset, [400, 50])

train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)

epochs = 100

total_d_loss = 0
total_d_loss_val = 0
total_d_loss_test = 0

total_g_loss = 0
total_g_loss_val = 0
total_g_loss_test = 0
vgg.to(device)

# Training loop
for epoch in range(epochs):

    count = 0

    total_d_loss = 0
    total_d_loss_val = 0

    total_g_loss = 0
    total_g_loss_val = 0

    for lr_images, hr_images in train_loader:

        count = count + 1

        # Move data to device (GPU or CPU)
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        # Generate fake images once
        fake_images = generator(lr_images)

        # ------------ Train Discriminator -------------
        discriminator_optim.zero_grad()

        # Real HR Images should be classified as REAL (1)
        real_pred = discriminator(hr_images)
        real_labels = torch.ones(real_pred.size(0), 1).float().to(device)
        real_loss = adversarial_loss_function(real_pred, real_labels)

        # Generated images should be classified as FAKE (0)
        fake_pred = discriminator(fake_images.detach())
        fake_labels = torch.zeros(fake_pred.size(0), 1).float().to(device)
        fake_loss = adversarial_loss_function(fake_pred, fake_labels)

        d_loss = (real_loss + fake_loss) * 0.5 # Discriminator Loss
        d_loss.backward()  # Backward Propagation
        discriminator_optim.step()  # Optimizer step

        # ----------- Train Generator ------------
        generator_optim.zero_grad()

        # Fool the discriminator
        gen_pred = discriminator(fake_images)
        gen_labels = torch.ones(gen_pred.size(0), 1).float().to(device)

        content_loss = content_loss_function(fake_images, hr_images)
        adversarial_loss = adversarial_loss_function(gen_pred, gen_labels)
        v_loss = vgg_loss(fake_images, (hr_images).to(device).float())

        # Combined loss (with appropriate weighting)
        # Content loss is more important early on, adversarial loss becomes more important later
        adversarial_weight = 0.001

        gen_loss =  v_loss + content_loss + adversarial_loss * adversarial_weight

        #gen_loss = content_weight * content_loss + adversarial_weight * adversarial_loss + perceptual_loss * perceptual_weight
        gen_loss.backward() 
        generator_optim.step()

        total_d_loss = total_d_loss + d_loss
        total_g_loss = total_g_loss + gen_loss.item()

            # Save side-by-side images (LR, Generated, HR) for the first batch of each epoch
        if count == 1:
            # Extract the first images from the batch
            lr_image = lr_images[1].cpu().clone().detach()      # Low-resolution input image
            hr_image = hr_images[1].cpu().clone().detach()        # High-resolution ground truth image
            gen_image = fake_images[1].clamp(0, 1).cpu().clone().detach()  # Generated image (clamped to [0, 1])
            
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

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {total_d_loss/len(train_loader):.4f} | G Loss: {total_g_loss/len(train_loader):.4f}")

    discriminator.eval()
    generator.eval()

    count = 0

    with torch.no_grad():
        for lr_images, hr_images in val_loader:

            count = count + 1

            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            # Generate Fake Image

            fake_images_val = generator(lr_images)

            # -------- Validate Discriminator ---------

            # REAL
            real_pred_val = discriminator(hr_images)
            real_labels_val = torch.ones(real_pred_val.size(0), 1).float().to(device)
            real_loss_val = adversarial_loss_function(real_pred_val, real_labels_val)

            # FAKE
            fake_pred_val = discriminator(fake_images_val.detach())
            fake_labels_val = torch.zeros(fake_pred_val.size(0), 1).float().to(device)
            fake_loss_val = adversarial_loss_function(fake_pred_val, fake_labels_val)

            d_loss_val = (real_loss_val + fake_loss_val) * 0.5

            # -------- Validate Generator ---------            

            gen_pred_val = discriminator(fake_images_val)
            gen_labels_val = torch.ones(gen_pred_val.size(0), 1).float().to(device)

            content_loss_val = content_loss_function(fake_images_val, hr_images)
            adversarial_loss_val = adversarial_loss_function(gen_pred_val, gen_labels_val)
            v_loss_val = vgg_loss(fake_images_val, (hr_images).to(device).float())

            #perceptual_loss_val = ms_ssim_loss(fake_images_val, hr_images)
            #sr_loss_val = enhanced_loss(fake_images_val, hr_images)

            # Use the same weighting as in training
            adversarial_weight = 0.001

            gen_loss_val = v_loss_val + content_loss_val + adversarial_loss_val * adversarial_weight

            #gen_loss_val = content_weight * content_loss_val + adversarial_weight * adversarial_loss_val + perceptual_loss_val * perceptual_weight

            total_d_loss_val = total_d_loss_val + d_loss_val
            total_g_loss_val = total_g_loss_val + gen_loss_val.item()

            # Save side-by-side images (LR, Generated, HR) for the first batch of each epoch
            if count == 1:
                # Extract the first images from the batch
                lr_image = lr_images[0].cpu().clone().detach()      # Low-resolution input image
                hr_image = hr_images[0].cpu().clone().detach()        # High-resolution ground truth image
                gen_image = fake_images_val[0].clamp(0, 1).cpu().clone().detach()  # Generated image (clamped to [0, 1])
                
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
                fig.savefig(os.path.join(image_dir, f"epoch_{epoch+1}_comparison_val.png"))
                plt.close(fig)
    
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {total_d_loss_val/len(val_loader):.4f} | G Loss: {total_g_loss_val/len(val_loader):.4f}")

discriminator.eval()
generator.eval()

with torch.no_grad():
    for lr_images, hr_images in test_loader:

        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        # Generate Image

        fake_images_test = generator(lr_images)

        # ------ Test Discriminator ------

        # REAL

        real_pred_test = discriminator(hr_images)
        real_labels_test = torch.ones(real_pred_test.size(0), 1).float().to(device)
        
        real_loss_test = adversarial_loss_function(real_pred_test, real_labels_test)

        # FAKE

        fake_pred_test = discriminator(lr_images)
        fake_labels_test = torch.zeros(fake_pred_test.size(0), 1).float().to(device)
        fake_loss_test = adversarial_loss_function(fake_pred_test, fake_labels_test)
        d_loss_test = (real_loss_test + fake_loss_test) * 0.5

        # -------- Test Generator ------- 

        gen_pred_test = discriminator(fake_images_test)
        gen_labels_test = torch.ones(gen_pred_test.size(0), 1).to(device)

        content_loss_test = content_loss_function(fake_images_test, hr_images)
        adversarial_loss_test = adversarial_loss_function(gen_pred_test, gen_labels_test)
        v_loss_test = vgg_loss(fake_images_test, torch.from_numpy(hr_images).to(device).float())
        #perceptual_loss_test = ms_ssim_loss(fake_images_test, hr_images)
        #sr_loss_test = enhanced_loss(fake_images_test, hr_images)

        adversarial_weight = 0.001

        gen_loss_test = v_loss_test + content_loss_test + adversarial_loss * adversarial_weight

        #gen_loss_test = content_loss_test * content_weight + adversarial_loss_test * adversarial_weight + perceptual_loss_test * perceptual_weight

        total_d_loss_test = total_d_loss_test + d_loss_test
        total_g_loss_test = total_g_loss_test + gen_loss_test.item()

print(f"Test D Loss: {total_d_loss_test/len(test_loader):.4f} | Test G Loss: {total_g_loss_test/len(test_loader):.4f}")
