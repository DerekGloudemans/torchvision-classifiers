"""
Code for the PackageNet() pytorch network. So called bacause it places objects 
in 3D boxes and puts labels (classifications) on them.
"""


# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division
from parallel_regression_classification import load_model

# torch and specific torch packages for convenience
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim import lr_scheduler
from torch import multiprocessing

# for convenient data loading, image representation and dataset management
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from scipy.ndimage import affine_transform

# always good to have
import time
import os
import numpy as np    
import _pickle as pickle
import random
import copy
import matplotlib.pyplot as plt
import math


class PackageNet(nn.Module):
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(PackageNet, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        self.vgg = models.vgg19(pretrained=True)
        del self.vgg.classifier[2:]

        # get size of some layers
        start_num = self.vgg.classifier[0].out_features
        mid_num0 = int(np.sqrt(start_num))
        mid_num1 = int(start_num**0.667)
        mid_num2 = int(start_num**0.333)
        
        cls_out_num = 9 # car or non-car (for now)
        reg_out_num = 16 # 8 3D bounding box coords
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num0,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num0,cls_out_num,bias = True),
                          nn.Softmax(dim = 1)
                          )
        
        # define regressor
        # try relu and tanh, also try without bias
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num1,bias=True),
                          nn.Sigmoid(),
                          nn.Linear(mid_num1,mid_num2,bias = True),
                          nn.Sigmoid(),
                          nn.Linear(mid_num2,reg_out_num,bias = True),
                          nn.Sigmoid()
                          
                          )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        vgg_out = self.vgg(x)
        cls_out = self.classifier(vgg_out)
        reg_out = self.regressor(vgg_out)
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out