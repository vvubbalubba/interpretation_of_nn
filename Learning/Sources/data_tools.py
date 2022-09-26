import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import csv
import random
import numpy as np
import cv2
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import io
from PIL import Image

labels_map = {
    "Benign": 0,
    "InSitu": 1,
    "Invasive": 2,
}

class ImageDataset(Dataset):
    def __init__(self, annotations_file, paths_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        f = open(paths_file, 'r')
        self.img_paths = f.read().split('\n')
        f.close()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        img_folder = int(img_path.split("/")[3])
        label_name = self.img_labels.iloc[img_folder - 1][4]
        label = torch.tensor(labels_map[label_name])
        if self.transform:
            image = image.float()
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image.float()
        return image, label
    
labels_map_2cl = {
    "Benign": 0,
    "Malignant": 1,
}
    
class ImageDataset_2cl(Dataset):
    def __init__(self, annotations_file, paths_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        f = open(paths_file, 'r')
        self.img_paths = f.read().split('\n')
        f.close()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        img_folder = int(img_path.split("/")[3])
        label_name = self.img_labels.iloc[img_folder - 1][4]
        if label_name == "InSitu" or label_name == "Invasive":
            label_name = "Malignant"
        label = torch.tensor(labels_map_2cl[label_name])
        if self.transform:
            image = image.float()
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image.float()
        return image, label
    
labels_map = {
    "Dog": 0,
    "Cat": 1,
}

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    
class CatDogsDataset(Dataset):
    def __init__(self, paths_file, transform=None, target_transform=None):
        f = open(paths_file, 'r')
        self.img_paths = f.read().split('\n')
        f.close()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = pil_loader(img_path)
#         image = read_image(img_path)
        label_name = img_path.split("/")[4]
        label = torch.tensor(labels_map[label_name])
        if self.transform:
#             image = image.float()
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
#         image = image.float()
        return image, label

labels_map = {
    "NotCat": 0,
    "Cat": 1,
}

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    
class CatsDataset(Dataset):
    def __init__(self, paths_file, transform=None, target_transform=None):
        f = open(paths_file, 'r')
        self.img_paths = f.read().split('\n')
        f.close()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = pil_loader(img_path)
#         image = read_image(img_path)
        label_name = img_path.split("/")[4]
        label = torch.tensor(labels_map[label_name])
        if self.transform:
#             image = image.float()
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
#         image = image.float()
        return image, label
