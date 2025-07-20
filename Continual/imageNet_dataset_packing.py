
from os.path import expanduser, isfile
from annoy import AnnoyIndex
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import numpy as np
from tqdm import tqdm
import time
import argparse
import torch
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from networks import *
import pickle
from SimCLR.data_aug.gaussian_blur import GaussianBlur
from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator
from SimCLR.exceptions.exceptions import InvalidDatasetSelection
from SimCLR.models.resnet_simclr import ResNetSimCLR
from SimCLR.simclr import SimCLR
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import os
import os
import tarfile
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def create_tar_archive(input_dir, output_dir, archive_name, img_size=(32, 32)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, archive_name)
    
    # Define the image resizing transform
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()  # Convert to tensor (if needed)
    ])
    
    with tarfile.open(archive_path, 'w') as tar:
        # Iterate over each class directory
        for class_dir in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            # Process each image in the class directory
            for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_dir}"):
                img_path = os.path.join(class_path, img_name)
                
                # Load image and apply resizing transform
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img)  # Resize the image (still a PIL Image)
                    
                    # Save the resized image to a temporary in-memory file
                    temp_path = f"/tmp/{img_name}"
                    torch.save(img,temp_path)
                    
                    # Add the resized image to the tar archive
                    tar.add(temp_path, arcname=f"{class_dir}/{img_name}")
                    os.remove(temp_path)  # Clean up the temporary file
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


# Example usage
input_train_dir = '/fs/ess/PAS1289/ImageNet/imagenet/train'
input_valid_dir = '/fs/ess/PAS1289/ImageNet/imagenet/val'
output_dir = '/users/PAS1289/oiocha/Persistent_Message_Passing/Continual/data/ImageNet'
create_tar_archive(input_train_dir, output_dir, 'train_images.tar', img_size=(32, 32))
create_tar_archive(input_valid_dir, output_dir, 'valid_images.tar', img_size=(32, 32))