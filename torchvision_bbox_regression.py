"""

Similar to in transfer_learn_cars.py, in this file are defined dataset object 
for training and validating a model for regression of vehicle bounding boxes and
binary classification using a pretrained pytorch classifier such as
(vgg-19)

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
        pos_list.sort()
        neg_list =  [negatives_path+'/train/'+f for f in os.listdir(negatives_path+ '/train/')]
        neg_list.sort()
        self.file_list = pos_list + neg_list

        # load labels (first 4 are bbox coors, then class (1 for positive, 0 for negative)
        pos_labels = np.load(positives_path+'/labels/train_bboxes.npy')
        neg_labels = np.load(negatives_path+'/labels/train_bboxes.npy')
        self.labels = np.concatenate((pos_labels,neg_labels),0)
        
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
        y = self.labels[index,:]
        #im, y = random_transforms(im,y)
        
        # normalize and convert to 
        #X = self.transforms(im)
        
        return im, y


def random_transforms(im,y,imsize = 224,tighten = 0.05):
    """
    Performs transforms that affect both X and y, as the transforms package 
    of torchvision doesn't do this elegantly
    inputs: im - image
             y  - 1 x 5 numpy array of bbox corners and class. the order is: 
                min x, min y, max x max y
    outputs: im - transformed image
             y  - 1 x 5 numpy array of bbox corners and class
    """
    
    #define parameters for random transform
    scale = min(2.5,max(random.gauss(0.5,1),imsize/min(im.size))) # verfify that scale will at least accomodate crop size
    shear = (random.random()-0.5)*30 #angle
    rotation = (random.random()-0.5) * 60.0 #angle
    
    # transform matrix
    im = transforms.functional.affine(im,rotation,(0,0),scale,shear)
    (xsize,ysize) = im.size
    

    # image transformation matrix
    shear = math.radians(-shear)
    rotation = math.radians(-rotation)
    M = np.array([[scale*np.cos(rotation),-scale*np.sin(rotation+shear)], 
                  [scale*np.sin(rotation), scale*np.cos(rotation+shear)]])
    
    # only transform coordinates for positive examples (negatives are [0,0,0,0,0])
    # clockwise from top left corner
    if y[4] == 1:
        # add 5th point corresponding to image center
        corners = np.array([[y[0],y[1]],[y[2],y[1]],[y[2],y[3]],[y[0],y[3]],[int(xsize/2),int(ysize/2)]])
        new_corners = np.matmul(corners,M)
        
        # Resulting corners make a skewed, tilted rectangle - realign with axes
        y = np.ones(5)
        y[0] = np.min(new_corners[:4,0])
        y[1] = np.min(new_corners[:4,1])
        y[2] = np.max(new_corners[:4,0])
        y[3] = np.max(new_corners[:4,1])
        
        # shift so transformed image center aligns with original image center
        xshift = xsize/2 - new_corners[4,0]
        yshift = ysize/2 - new_corners[4,1]
        y[0] = y[0] + xshift
        y[1] = y[1] + yshift
        y[2] = y[2] + xshift
        y[3] = y[3] + yshift
        y = y.astype(int)
        
        # brings bboxes in slightly on positive examples
        if tighten != 0 and y[4] == 1:
            xdiff = y[2] - y[0]
            ydiff = y[3] - y[1]
            y[0] = y[0] + xdiff*tighten
            y[1] = y[1] + ydiff*tighten
            y[2] = y[2] - xdiff*tighten
            y[3] = y[3] - ydiff*tighten
        
    # get crop location with normal distribution at image center
    crop_x = int(random.gauss(im.size[0]/2,xsize/10/scale)-imsize/2)
    crop_y = int(random.gauss(im.size[1]/2,ysize/10/scale)-imsize/2)
    
    # move crop if too close to edge
    pad = 50
    if crop_x < pad:
        crop_x = im.size[0]/2 - imsize/2 # center
    if crop_y < pad:
        crop_y = im.size[1]/2 - imsize/2 # center
    if crop_x > im.size[0] - imsize - pad:
        crop_x = im.size[0]/2 - imsize/2 # center
    if crop_y > im.size[0] - imsize - pad:
        crop_y = im.size[0]/2 - imsize/2 # center  
    im = transforms.functional.crop(im,crop_y,crop_x,imsize,imsize)
    
    # transform bbox points into cropped coords
    if y[4] == 1:
        y[0] = y[0] - crop_x
        y[1] = y[1] - crop_y
        y[2] = y[2] - crop_x
        y[3] = y[3] - crop_y
    
    return im,y,crop_x,crop_y
    
# will need to be redefined
def show_output(model,loader):
    batch,labels = next(iter(loader))
    batch = batch.to(device)
    
    out = model(batch)
    _,preds = torch.max(out,1)
    
    batch = batch.to('cpu')
    preds = preds.to('cpu')
    
    # define figure subplot grid
    batch_size = loader.batch_size
    fig, axs = plt.subplots((batch_size+7)//8, 8, constrained_layout=True)
    
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        im =  batch[i].numpy().transpose((1,2,0))
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = std * im + mean
        im = np.clip(im, 0, 1)
        
        if preds[i] == 1:
            label = "pred: car"
        else:
            label = "pred: non-car"
        
        axs[i//8,i%8].imshow(im)
        axs[i//8,i%8].set_title(label)
        axs[i//8,i%8].set_xticks([])
        axs[i//8,i%8].set_yticks([])
        
        plt.pause(0.0001)



def load_bbox_mat():
    import scipy.io as sio
    temp = sio.loadmat("/media/worklab/data_HDD/cv_data/images/data_stanford_cars/labels/cars_test_annos.mat",byte_order = "=")
    temp2 = temp['annotations']
    temp3 = temp2[0]
    bbox_idxs = np.zeros([len(temp3),4])
    for i,item in enumerate(temp3):
        item = item.item()
        bbox_idxs[i][0] = item[0][0][0]
        bbox_idxs[i][1] = item[1][0][0]
        bbox_idxs[i][2] = item[2][0][0]
        bbox_idxs[i][3] = item[3][0][0]
    np.save("outfile.npy".bbox_idxs)

#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        print("If multiprocessing context wasn't already set, error")
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    
    # for repeatability
    random.seed = 0
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    
    # create training params
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 6}
    num_epochs = 3
    
    checkpoint_file = None# "checkpoint_2.pt"
    
    # create dataloaders
    pos_path = "/media/worklab/data_HDD/cv_data/images/data_stanford_cars"
    neg_path = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader"
    train_data = Train_Dataset(pos_path,neg_path)
    #test_data = Test_Dataset(pos_path,neg_path)
    trainloader = data.DataLoader(train_data, **params)
    #testloader = data.DataLoader(test_data, **params)
    print("Dataloaders created.")
    
    im , yin = train_data[9]
    im2, y,cropx,cropy= random_transforms(im,yin,tighten = 0)
    im_array = np.array(im2)
    new_im = cv2.rectangle(im_array,(y[0],y[1]),(y[2],y[3]),(255,0,0),3)
    plt.imshow(new_im)
#    im_array = cv2.circle(np.array(im),(int(cropx),int(cropy)),10,(255,0,0),-1)
#    plt.imshow(im_array)