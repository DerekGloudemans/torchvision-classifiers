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
    
import _pickle as pickle

class ToyDataset(data.Dataset):
    '''
    Creates a dataset that points to a file for validation of Resnet classifier, no training labels
    '''

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
    
def parse_label_file():
    '''
    loads text file, parses into a list of labels, and pickles result
    '''
    f = open('imagenet_classes.txt','rb')
    label_dict = {}    
    for line in f:
        string = line.decode()
        parts = string.rstrip().split(":")
        idx = int(parts[0][1:])
        label = parts[1].split("'")[1].split(",")[0]
        label_dict[idx] = label
    f.close()
    
    label_list = []
    for i in range(0,len(label_dict)):
        label_list.append(label_dict[i]) 
    
    new_f = open("imagenet_label_list.cpkl", 'wb')    
    pickle.dump(label_list,new_f)
    new_f.close()
    
def load_label_list(pickle_name = "imagenet_label_list.cpkl"):
    try:
        f = open(pickle_name, 'rb')
        label_list = pickle.load(f)
    except:
        parse_label_file()
        f = open(pickle_name, 'rb')
        label_list = pickle.load(f)
    return label_list
    
def print_preds(preds,label_list):
    # get index of max prob for each item
    for item in out:
        idx = np.argmax(item)
        print ("Predicted class: {}".format(label_list[idx]))


def imshow(inp, title=None):
    """
    Imshow for Tensor.
    """
    inp = inp.cpu().numpy()
    print (inp.shape)
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize = (10,80))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


##############################################################################


# load labels
label_list = load_label_list()
 
# load pretrained model
try:
    model
    print("Model already loaded")
except:
    #model = models.densenet161(pretrained = True)
    model = models.resnet152(pretrained = True)
    print("Reloaded model")

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 2}
max_epochs = 1

# Generators
#test_set = ToyDataset('data_stanford_cars/train')
test_set = ToyDataset('data_ILSVRC_2017/test')
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
    model = model.to(device)
    pred = model(batch)
    probs = F.softmax(pred,dim = 1)
    out = probs.cpu().numpy()

 # clean up some resources   
torch.cuda.empty_cache()
del training_generator

plot = torchvision.utils.make_grid(batch)
preds = [label_list[np.argmax(out[i])] for i in range(len(out))] 
imshow(plot,preds)

