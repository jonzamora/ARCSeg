'''
DataLoader for Semantic Segmentation on Surgical Datasets
NOTE: Does not load all data into memory. Use if you don't have enough RAM for storing data.
'''

import torch
from torch.utils.data import Dataset
from torchvision.io.image import ImageReadMode
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image

import os
import json
import random

class SegNetDataset(Dataset):
    '''
    Dataset Class for Semantic Segmentation on Surgical Data
    '''

    def __init__(self, root_dir, transform=None, json_path=None, sample=None, 
                 dataset=None, image_size=[256, 256], horizontal_flip=True, brightness=True, contrast=True,
                 rotate=True, vertical_flip=True):
        '''
        args:

        root_dir (str) = File Directory with Input Surgical Images

        transform (callable) = Optional torchvision transforms for data augmentation on training samples

        json_path (str) = File with Semantic Segmentation Class information

        sample (str) = Specify whether the sample is from train, test, or validation set

        dataset (str) = Specify whether the Segmentation dataset is from Synapse, Cholec, or Miccai
        '''
        
        # General Parameters
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'groundtruth')
        self.image_list = [f for f in os.listdir(self.img_dir) if (f.endswith(".png") or f.endswith(".jpg"))]
        self.transform = transform
        self.sample = sample
        self.dataset = dataset

        # Data Augmentation Parameters
        self.resizedHeight = image_size[0]
        self.resizedWidth = image_size[1]
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate = rotate
        self.brightness = brightness
        self.contrast = contrast

        if json_path:
            self.classes = json.load(open(json_path))["classes"]
        
        '''
        Load Dataset into Memory
        '''

        self.images, self.gt_images = [], []

        for idx, val in enumerate(self.image_list):
            img_name = os.path.join(self.img_dir, self.image_list[idx])

            if self.dataset == "synapse":
                gt_file_name = self.image_list[idx][0:-4] + ".png"
            elif self.dataset == "cholec":
                gt_file_name = self.image_list[idx][0:-4] + "_color_mask.png"
            elif self.dataset == "miccai":
                gt_file_name = self.image_list[idx][0:-4] + "_gt.png"
            else:
                raise ValueError("Ground Truth File Name Does Not Exist")

            gt_name = os.path.join(self.gt_dir, gt_file_name)

            image = Image.open(img_name)
            image = image.convert("RGB")

            gt = Image.open(gt_name)
            gt = gt.convert("RGB")

            self.images.append(image)
            print(f"loaded image: {img_name}")
            self.gt_images.append(gt)
            print(f"loaded gt: {gt_name}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.img_dir, self.image_list[idx])

        #if self.dataset == "synapse":
        #    gt_file_name = self.image_list[idx][0:-4] + ".png"
        #elif self.dataset == "cholec":
        #    gt_file_name = self.image_list[idx][0:-4] + "_color_mask.png"
        #elif self.dataset == "miccai":
        #    gt_file_name = self.image_list[idx][0:-4] + "_gt.png"
        #else:
        #    raise ValueError("Ground Truth File Name Does Not Exist")

        #gt_name = os.path.join(self.gt_dir, gt_file_name)

        #image = Image.open(img_name)
        #image = image.convert("RGB")

        #gt = Image.open(gt_name)
        #gt = gt.convert("RGB")

        image, gt = self.images[idx], self.gt_images[idx]

        if self.sample == "train":
            # Random Horizontal Flip
            if self.horizontal_flip and random.random() > 0.5:
                image, gt = TF.hflip(image), TF.hflip(gt)
            
            # Random Vertical Flip
            if self.vertical_flip and random.random() > 0.5:
                image, gt = TF.vflip(image), TF.vflip(gt)
            
            image = TF.resize(image, [self.resizedHeight, self.resizedWidth], interpolation=Image.BILINEAR)
            gt = TF.resize(gt, [self.resizedHeight, self.resizedWidth], interpolation=Image.NEAREST)
            
            # Random Rotate
            if self.rotate and random.random() > 0.5:
                image, gt = TF.rotate(image, 90), TF.rotate(gt, 90)

            # Brightness Adjustment
            if self.brightness and random.random() > 0.5:
                bright_factor = random.uniform(0.9, 1.1)
                image = TF.adjust_brightness(image, bright_factor)

            # Contrast Adjustment
            if self.contrast and random.random() > 0.5:
                cont_factor = random.uniform(0.9, 1.1)
                image = TF.adjust_contrast(image, cont_factor)
        
        if self.transform:
            image, gt = self.transform(image), self.transform(gt)

        return image, gt