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
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    
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
import random


#--------------------------- Definitions section -----------------------------#
class Train_Dataset(data.Dataset):
    """
    Defines dataset and transforms for training data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self, positives_path,negatives_path):

        # use os module to get a list of positive and negative training examples
        # note that shuffling is essential because examples are in order
        pos_list =  [positives_path+'/train/'+f for f in os.listdir(positives_path+ '/train/')]
        neg_list =  [negatives_path+'/train/'+f for f in os.listdir(negatives_path+ '/train/')]
        self.file_list = pos_list + neg_list

        # create labels (1 for positive, 0 for negative)
        pos_labels = [1 for i in range(len(pos_list))]
        neg_labels = [0 for i in range(len(neg_list))]
        self.labels = pos_labels + neg_labels
        
        self.transforms = transforms.Compose([\
        transforms.RandomRotation(90),
        transforms.RandomAffine(20),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        #Denotes the total number of samples
        return len(self.file_list)

    def __getitem__(self, index):
        #Generates one sample of data
        file_name = self.file_list[index]
        im = Image.open(file_name).convert('RGB')
        X = self.transforms(im)
        y = self.labels[index]
        return X, y

class Test_Dataset(data.Dataset):
    """
    Defines dataset and transforms for testing data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self, positives_path,negatives_path):

        # use os module to get a list of positive and negative training examples
        # note that shuffling is essential because examples are in order
        pos_list =  [positives_path+'/test/'+f for f in os.listdir(positives_path+'/test/')]
        neg_list =  [negatives_path+'/test/'+f for f in os.listdir(negatives_path+'/test/')]
        self.file_list = pos_list + neg_list

        # create labels (1 for positive, 0 for negative)
        pos_labels = [1 for i in range(len(pos_list))]
        neg_labels = [0 for i in range(len(neg_list))]
        self.labels = pos_labels + neg_labels
        
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
        file_name = self.file_list[index]
        im = Image.open(file_name).convert('RGB')
        X = self.transforms(im)
        y = self.labels[index]
        return X, y
    

def load_model():
    pass

def train():
    pass

def show_output():
    pass


def flatten_image_directory():
    from shutil import copyfile

    train_directory = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader/train"
    test_directory = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader/test"
    base_directory = "/media/worklab/data_HDD/cv_data/images/imagenet_images"
    sub_dir_list = os.listdir(base_directory)
    for subdir in sub_dir_list:
        file_list = os.listdir(os.path.join(base_directory,subdir))
        count = 0
        for file in file_list:
            if count < 10:
                copyfile(os.path.join(base_directory,subdir,file),os.path.join(train_directory,file))
                count +=1
            else:
                copyfile(os.path.join(base_directory,subdir,file),os.path.join(test_directory,file))
        print("Subdirectory copied.")


#------------------------------ Main code here -------------------------------#

# for repeatability
random.seed = 1

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# create training params
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 5

# create training dataloader
pos_path = "/media/worklab/data_HDD/cv_data/images/data_stanford_cars"
neg_path = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader"
train_data = Train_Dataset(pos_path,neg_path)
trainloader = data.DataLoader(train_data, **params)

# create testing dataloader
test_data = Test_Dataset(pos_path,neg_path)
testloader = data.DataLoader(test_data, **params)