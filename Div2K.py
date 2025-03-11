import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

import os 
from PIL import Image
import random

class Div2K_Dataset(Dataset):

    def __init__(self, root_dir, subset, scale, patch_size=4, transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.scale = scale
        self.patch_size = patch_size
        self.hr_patch_size = patch_size * scale
        self.transform = transform

        lr_dir = os.path.join(root_dir, 'Bicubic', f'DIV2K_{subset}_LR_bicubic_X{scale}', f'DIV2K_{subset}_LR_bicubic', f'X{scale}')
        hr_dir = os.path.join(root_dir, 'HR', f'DIV2K_{subset}_HR', f'DIV2K_{subset}_HR')

        self.lr_images = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg'))])
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))])

        # Store full paths 
        self.lr_paths = [os.path.join(lr_dir, img) for img in self.lr_images]
        self.hr_paths = [os.path.join(hr_dir, img) for img in self.hr_images]

        # Find minimum width and height
        self.min_lr_width = min(Image.open(path).size[0] for path in self.lr_paths)
        self.min_lr_height = min(Image.open(path).size[1] for path in self.lr_paths)

        self.min_hr_width = min(Image.open(path).size[0] for path in self.lr_paths)
        self.min_hr_height = min(Image.open(path).size[1] for path in self.lr_paths)


    def __len__(self):
        """
        Returns the total number of image pairs in the dataset
        """
        return len(self.lr_images)  

    def __getitem__(self, index):
        """
        Loads and returns a single image pair
        
        Returns:
        - low_res_tensor: Low-resolution image tensor
        - high_res_tensor: High-resolution image tensor
        """
        # Load low-resolution and high-resolution images
        lr_image = Image.open(self.lr_paths[index])
        hr_image = Image.open(self.hr_paths[index])

        # Get image sizes
        #lr_width, lr_height = lr_image.size
        #hr_width, hr_height = hr_image.size

        # Select a **random patch** location
        #lr_x = random.randint(0, lr_width - self.patch_size)
        #lr_y = random.randint(0, lr_height - self.patch_size)

        #hr_x = lr_x * self.scale
        #hr_y = lr_y * self.scale


        # Crop Patches
        #lr_patch = lr_image.crop((lr_x, lr_y, lr_x + self.patch_size, lr_y + self.patch_size))
        #hr_patch = hr_image.crop((hr_x, hr_y, hr_x + self.patch_size, hr_y + self.patch_size))

        # Crop images
        lr_image = lr_image.crop((0, 0, self.min_lr_width, self.min_lr_height))
        hr_image = hr_image.crop((0, 0, self.min_hr_height, self.min_hr_height))

        # Upscale LR image to match HR image
        lr_size = (hr_image.width, hr_image.height)
        lr_image = lr_image.resize(lr_size, Image.BICUBIC)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image
