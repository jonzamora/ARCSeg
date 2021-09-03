'''
DataLoader for Semantic Segmentation on Surgical Datasets
NOTE: This dataloader Loads all data into RAM. If your computer doesn't have enough RAM, use V1 dataloader.
'''

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from tqdm import tqdm

import numpy as np
from PIL import Image

import os
import json
import random

class SegNetDataset(Dataset):
    '''
    Dataset Class for Semantic Segmentation on Surgical Data
    '''

    def __init__(self, root_dir, crop_size=-1, json_path=None, sample=None, 
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
        self.image_list = np.array([f for f in os.listdir(self.img_dir) if (f.endswith(".png") or f.endswith(".jpg"))])
        self.crop_size = crop_size
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
        
        self.key = self.generateKey(self.classes)
        
        '''
        Load Dataset into Memory
        '''

        self.images, self.gt_images, self.labels = [], [], []

        data_loading = tqdm(enumerate(self.image_list), total=len(self.image_list))

        for idx, val in data_loading:

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

            gt_image = Image.open(gt_name)
            gt_image = gt_image.convert("RGB")

            to_tensor = ToTensor()
            image, gt_image = to_tensor(image), to_tensor(gt_image)

            if self.sample == 'train':
                image = TF.resize(image, [540, 960], interpolation=Image.BILINEAR)
                gt = TF.resize(gt_image, [540, 960], interpolation=Image.NEAREST)

                if self.crop_size != -1:
                    image_crops = TF.five_crop(img=image, size=self.crop_size)
                    gt_crops = TF.five_crop(img=gt_image, size=self.crop_size)

                    for im in image_crops:
                        self.images.append(im)
                    
                    # Generate Label For Cross Entropy Loss
                    for gt in gt_crops:
                        gt_label = gt.permute(1, 2, 0)
                        gt_label = (gt_label * 255).long()
                        catMask = torch.zeros((gt_label.shape[0], gt_label.shape[1]))

                        # Iterate over all the key-value pairs in the class Key dict
                        for k in range(len(self.key)):
                            rgb = torch.Tensor(self.key[k])
                            mask = torch.all(gt_label == rgb, axis=2)
                            assert mask.shape == catMask.shape, f"mask shape {mask.shape} unequal to catMask shape {catMask.shape}"
                            catMask[mask] = k
                    
                        catMask = catMask.unsqueeze(0) # expands dimension to [1, self.resizedHeight, self.resizedWidth]
                        self.labels.append(catMask)
                        self.gt_images.append(gt)
                else:
                    gt_label = gt.permute(1, 2, 0)
                    gt_label = (gt_label * 255).long()
                    catMask = torch.zeros((gt_label.shape[0], gt_label.shape[1]))
                    
                    # Iterate over all the key-value pairs in the class Key dict
                    for k in range(len(self.key)):
                        rgb = torch.Tensor(self.key[k])
                        mask = torch.all(gt_label == rgb, axis=2)
                        assert mask.shape == catMask.shape, f"mask shape {mask.shape} unequal to catMask shape {catMask.shape}"
                        catMask[mask] = k
                    
                    catMask = catMask.unsqueeze(0) # expands dimension to [1, self.resizedHeight, self.resizedWidth]
                    self.labels.append(catMask)
                    self.images.append(image)
                    self.gt_images.append(gt)
            elif self.sample == 'test':
                image = TF.resize(image, [1080, 1920], interpolation=Image.BILINEAR)
                gt = TF.resize(gt_image, [1080, 1920], interpolation=Image.NEAREST)

                gt_label = gt.permute(1, 2, 0)
                gt_label = (gt_label * 255).long()
                catMask = torch.zeros((gt_label.shape[0], gt_label.shape[1]))
                
                # Iterate over all the key-value pairs in the class Key dict
                for k in range(len(self.key)):
                    rgb = torch.Tensor(self.key[k])
                    mask = torch.all(gt_label == rgb, axis=2)
                    assert mask.shape == catMask.shape, f"mask shape {mask.shape} unequal to catMask shape {catMask.shape}"
                    catMask[mask] = k
                
                catMask = catMask.unsqueeze(0) # expands dimension to [1, self.resizedHeight, self.resizedWidth]
                self.labels.append(catMask)
                self.images.append(image)
                self.gt_images.append(gt)


            data_loading.set_description(f"Loading {self.sample} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, gt, label = self.images[idx], self.gt_images[idx], self.labels[idx]

        if self.sample == "train":
            # Random Horizontal Flip
            if self.horizontal_flip and random.random() > 0.5:
                image, gt, label = TF.hflip(image), TF.hflip(gt), TF.hflip(label)
            
            # Random Vertical Flip
            if self.vertical_flip and random.random() > 0.5:
                image, gt, label = TF.vflip(image), TF.vflip(gt), TF.vflip(label)
            
            # Random Rotate
            if self.rotate and random.random() > 0.5:
                image, gt, label = TF.rotate(image, 90), TF.rotate(gt, 90), TF.rotate(label, 90)
            
            label = label.squeeze()

            # Brightness Adjustment
            if self.brightness and random.random() > 0.5:
                bright_factor = random.uniform(0.9, 1.1)
                image = TF.adjust_brightness(image, bright_factor)

            # Contrast Adjustment
            if self.contrast and random.random() > 0.5:
                cont_factor = random.uniform(0.9, 1.1)
                image = TF.adjust_contrast(image, cont_factor)
            
            image, gt, label = self.random_crop(image, gt, label, self.resizedWidth, self.resizedHeight)

        
        if self.sample == "test":
            label = label.squeeze()
        
        gt = gt * 255

        return image.type(torch.float32), gt.type(torch.int64), label.type(torch.int64)
    
    def generateKey(self, key):
        '''
            Disentangles the key for class and labels obtained from the
            JSON file
            Returns a python dictionary of the form:
                {Class Id: RGB Color Code as numpy array}
        '''
        dKey = {}
        for i in range(len(key)):
            class_id = int(key[i]['id'])
            c = key[i]['color']
            c = c.split(',')
            c0 = int(c[0][1:])
            c1 = int(c[1])
            c2 = int(c[2][:-1])
            color_array = np.asarray([c0,c1,c2])
            dKey[class_id] = color_array

        return dKey
    
    def random_crop(self, img, mask, label, width, height):
        assert img.shape[1] >= height, f"img.shape[0]: {img.shape[0]} is not >= height: {height}"
        assert img.shape[2] >= width, f"img.shape[2]: {img.shape[2]} is not >= width: {width}"
        assert img.shape[1] == mask.shape[1], f"img.shape[1] {img.shape[1]} != mask.shape[1]: {mask.shape[1]}"
        assert img.shape[2] == mask.shape[2], f"img.shape[2] {img.shape[2]} != mask.shape[2]: {mask.shape[2]}"
        x = random.randint(0, img.shape[2] - width)
        y = random.randint(0, img.shape[1] - height)
        img = img[:, y:y+height, x:x+width]
        mask = mask[:, y:y+height, x:x+width]
        label = label[y:y+height, x:x+width]
        return img, mask, label