# exploratory file for working with torchvision classifiers

# much of the code is from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils import data
    

class ToyDataset(data.Dataset):
    '''
    Creates a dataset that points to a file for validation of Resnet classifier, no training labels
    '''

    def __init__(self, directory_path):

        # use os module to get a list of all files in the data/temp directory
        self.file_list =  [directory_path+'/'+f for f in os.listdir(directory_path)]
        
        self.transforms = transforms.Compose([\
        transforms.Resize(620),
        transforms.CenterCrop(600),
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
    

# load pretrained model
try:
    resnet18
    print("Model already loaded")
except:
    resnet18 = models.resnet18(pretrained = True)
    print("Reloaded model")
    
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 2}
max_epochs = 1

# Generators
test_set = ToyDataset('data/train')
training_generator = data.DataLoader(test_set, **params)

# Test data_loader
if False:
    n = 1
    for local_batch in training_generator:
        # Transfer to GPU
        local_batch = local_batch.to(device)
        print("Loaded batch {} --> size: {}".format(n,local_batch.shape))
        n += 1
        
    torch.cuda.empty_cache()
    del training_generator
    
# run one batch through the model
batch = next(iter(training_generator))
with torch.set_grad_enabled(False):
    batch = batch.to(device)
    resnet18 = resnet18.to(device)
    out = resnet18(batch)
    probs = F.softmax(out,dim = 0)
    out = out.cpu().numpy()
    del batch, probs
    
torch.cuda.empty_cache()
del training_generator