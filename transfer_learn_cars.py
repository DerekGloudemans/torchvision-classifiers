"""
In this file are defined dataset objects for training and validating a model
for classification of images as cars/not cars. This model will be built upon
a pretrained pytorch classifier such as vgg-19 or ResNet (pending tests)

Extensions will then include:
    additional transforms for data augmentation, including occlusion
    further classification into vehicle class (car, truck, minivan, suv, etc.)
    further classification by make and model
    
Again, much of the code was informed by:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    
Datasets used in this classifier include:
    Stanford Cars
        @inproceedings{KrauseStarkDengFei-Fei_3DRR2013,
          title = {3D Object Representations for Fine-Grained Categorization},
          booktitle = {4th International IEEE Workshop on  3D Representation and Recognition (3dRR-13)},
          year = {2013},
          address = {Sydney, Australia},
          author = {Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
        }
    Imagenet - for base model training
       
"""

#--------------- Include necessary and unnecessary packages ------------------#

# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division

# torch and specific torch packages for convenience
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim import lr_scheduler

# for convenient data loading, image representation and dataset management
from torchvision import datasets, models, transforms
from PIL import Image

# always good to have
import time
import os
import numpy as np    
import _pickle as pickle


#--------------------------- Definitions section -----------------------------#
class Train_Dataset():
    """
    Defines dataset and transforms for training data
    """
    
    def __init__(self, directory_path):

        # use os module to get a list of all files in the data/temp directory
        self.file_list =  [directory_path+'/'+f for f in os.listdir(directory_path)]
        
        self.transforms = transforms.Compose([\
        transforms.Resize(300),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        #Denotes the total number of samples
        return len(self.file_list)

    def __getitem__(self, index):
        #Generates one sample of data
        # Select sample
        file_name = self.file_list[index]
        im = Image.open(file_name).convert('RGB')
        X = self.transforms(im)

        return X#, y


class Test_Dataset():
    pass

def load_model():
    pass

def train():
    pass

def show_output():
    pass

#------------------------------ Main code here -------------------------------#
    