"""
This file defines a network architecture for classifying objects in the KITTI 
dataset as well as regressing a 3D bounding box on the object. Separate heads
are trained for classification and regression.

During a second step of training, the regression head is copied and trained
specifically for pedestrians/bicycles, or cars,trucks,trams. 

This file defines a training cycle for the network, as well as custom loss functions:
    - im_space_loss - MSE of the 8 image coordinates
    - im_space_bbox_loss - bbox loss of front and back of bbox (approx)
                        
This file defines a training dataset, which implements transforms and converts 
examples to tensors. Transforms include:
    -random rotate, scale and crop (no shear)
    -normalize inputs
    -normalize labelspace between 0 and 1, according to some scaling factor that 
     controls the value assigned to the edges of the image (so points outside
     of the image can be predicted)
    - convert to tensor
"""


#--------------- Include necessary and unnecessary packages ------------------#

# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division
from parallel_regression_classification import load_model, score_pred, SplitNet

# torch and specific torch packages for convenience
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim import lr_scheduler
from torch import multiprocessing
from torch.autograd import Variable

# for convenient data loading, image representation and dataset management
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from scipy.ndimage import affine_transform

# always good to have
import time
import os
import sys
import numpy as np    
import _pickle as pickle
import random
import copy
import matplotlib.pyplot as plt
import math

global wer #window expansion size, controls how far out of frame bbox can be predicted 
wer = 5
    
#--------------------------- Definitions section -----------------------------#
class Train_Dataset_3D(data.Dataset):
    """
    Defines dataset and transforms for training data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self,directory, max_scaling = 2):

        self.im_dir = os.path.join(directory,"images")
        self.max_scaling = max_scaling
        self.class_dict = {
                'car': 0,
                'van': 1,
                'truck': 2,
                'pedestrian':3,
                'person_sitting':4,
                'cyclist':5,
                'tram': 6,
                'misc': 7
                }
        
        # use os module to get a list of training image files
        # note that shuffling in dataloader is essential because these are ordered
        self.images =  [f for f in os.listdir(os.path.join(directory,"images"))]
        self.images.sort()
        
        with open(os.path.join(directory,"labels.cpkl"),'rb') as f:
            self.labels = pickle.load(f)
        
        
        # get selection of examples with an equal number from each class
        all_idx_lists = []
        for key in self.class_dict.keys():
            idx_list = []
            for i in range(0,len(self.labels)):
                if self.labels[i]['class'].lower() == key:
                    idx_list.append(i)
            all_idx_lists.append(idx_list)
        
        # find shortest idx_list
        shortest = np.inf
        for idx_list in all_idx_lists:
            if len(idx_list) < shortest:
                shortest = len(idx_list)
        
        # maximum class imbalance = 5 times        
        shortest = shortest * 5
        
        # select examples
        final_idx = []
        for idx_list in all_idx_lists:        
            final_idx = final_idx + idx_list[:shortest]    
        new_images = []
        new_labels = []
        for idx in final_idx:
            new_images.append(self.images[idx])
            new_labels.append(self.labels[idx])
        self.labels = new_labels
        self.images = new_images
        
        # define transform
        self.transforms = transforms.Compose([\
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        #Denotes the total number of samples
        return len(self.images) 

    def __getitem__(self, index):
        #Generates one sample of data
        file_name = os.path.join(self.im_dir,self.images[index])
        
        # load relevant files
        im = Image.open(file_name).convert('RGB')
        y = self.labels[index]
        bbox_2d = y['bbox2d']
        bbox_3d = y['bbox3d']
        calib = y['calib']
        cls = y['class']
            
        cls =  self.class_dict[cls.lower()]
        
        # transform both image and label (note that the 2d and 3d bbox coords must both be scaled)
        im,bbox_2d,bbox_3d = self.random_scale_crop(im,bbox_2d,bbox_3d,imsize = 224, tighten = 0)
        
        # normalize and convert image to tensor
        X = self.transforms(im)
                
        # normalize 2d and 3d corners wrt image size and convert to tensor
        # pad to wer times image size because bbox may fall outside of visible coordinates
        bbox_2d[0] = (bbox_2d[0]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
        bbox_2d[1] = (bbox_2d[1]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
        bbox_2d[2] = (bbox_2d[2]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
        bbox_2d[3] = (bbox_2d[3]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
        
        # bbox covers whole area
        #bbox_2d = np.array([0.0,0.0,1.0,1.0])
        
        bbox_2d = torch.from_numpy(bbox_2d).float()
        
        bbox_3d[0,:] = (bbox_3d[0,:]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
        bbox_3d[1,:] = (bbox_3d[1,:]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
        bbox_3d = torch.from_numpy(np.reshape(bbox_3d,16)).float()
        
        
        
        calib = torch.from_numpy(calib).float()
        cls = torch.LongTensor([cls])
        # y is a tuple of four tensors: cls,2dbbox, 3dbbox, and camera calibration matrix
        y = (cls,bbox_2d,bbox_3d,calib)
        
        return X, y
    
    def random_scale_crop(self,im,bb2d,bb3d,imsize = 224,tighten = 0):
        """
        Performs transforms that affect both X and y, as the transforms package 
        of torchvision doesn't do this elegantly
        inputs: im - image
                 bb2d - 1 x 4 numpy array of bbox corners. the order is: 
                    min x, min y, max x max y
                bb3d - 2 x 8 numpy array of 3d bbox corners. first row is x, second row is y
        outputs: im - transformed image
                 bbox_2d - 1 x 4 numpy array of bbox corners and class
                 new_corners_3d - 2 x 8 numpy array of bbox corners and class
        """
    
        #define parameters for random transform
        scale = min(self.max_scaling,max(random.gauss(1.5,0.5),imsize/min(im.size))) # verfify that scale will at least accomodate crop size
        shear = 0 #(random.random()-0.5)*30 #angle
        rotation = (random.random()-0.5) * 20.0 #angle
        
        # transform matrix
        im = transforms.functional.affine(im,rotation,(0,0),scale,shear)
        (xsize,ysize) = im.size
        
        
            
        # image transformation matrix
        shear = math.radians(-shear)
        rotation = math.radians(-rotation)
        M = np.array([[scale*np.cos(rotation),-scale*np.sin(rotation+shear)], 
                      [scale*np.sin(rotation), scale*np.cos(rotation+shear)]])
        
        
        # add 5th point corresponding to image center
        corners = np.array([[bb2d[0],bb2d[1]],[bb2d[2],bb2d[1]],
                            [bb2d[2],bb2d[3]],[bb2d[0],bb2d[3]],
                            [int(xsize/2),int(ysize/2)]])
        new_corners = np.matmul(corners,M)
        
        # do the same thing for the 8 bb3d points
        bb3d = np.transpose(bb3d)
        new_corners_3d = np.matmul(bb3d,M)
        new_corners_3d = np.transpose(new_corners_3d)
        
        # Resulting corners make a skewed, tilted rectangle - realign with axes
        bbox_2d = np.ones(4)
        bbox_2d[0] = np.min(new_corners[:4,0])
        bbox_2d[1] = np.min(new_corners[:4,1])
        bbox_2d[2] = np.max(new_corners[:4,0])
        bbox_2d[3] = np.max(new_corners[:4,1])
        
        # shift so transformed image center aligns with original image center
        xshift = xsize/2 - new_corners[4,0]
        yshift = ysize/2 - new_corners[4,1]
        bbox_2d[0] = bbox_2d[0] + xshift
        bbox_2d[1] = bbox_2d[1] + yshift
        bbox_2d[2] = bbox_2d[2] + xshift
        bbox_2d[3] = bbox_2d[3] + yshift
        
        new_corners_3d[0,:] = new_corners_3d[0,:] + xshift
        new_corners_3d[1,:] = new_corners_3d[1,:] + yshift
        
        # brings bboxes in slightly on positive examples
        if tighten != 0:
            xdiff = bbox_2d[2] - bbox_2d[0]
            ydiff = bbox_2d[3] - bbox_2d[1]
            bbox_2d[0] = bbox_2d[0] + xdiff*tighten
            bbox_2d[1] = bbox_2d[1] + ydiff*tighten
            bbox_2d[2] = bbox_2d[2] - xdiff*tighten
            bbox_2d[3] = bbox_2d[3] - ydiff*tighten
            
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
        bbox_2d[0] = bbox_2d[0] - crop_x
        bbox_2d[1] = bbox_2d[1] - crop_y
        bbox_2d[2] = bbox_2d[2] - crop_x
        bbox_2d[3] = bbox_2d[3] - crop_y
        
        new_corners_3d[0,:] = new_corners_3d[0,:] - crop_x
        new_corners_3d[1,:] = new_corners_3d[1,:] - crop_y
        
        return im, bbox_2d, new_corners_3d
    
    def show(self, index,plot_3D = False):
        #Generates one sample of data
        file_name = os.path.join(self.im_dir,self.images[index])
        
        # load relevant files
        im = Image.open(file_name).convert('RGB')
        y = self.labels[index]
        bbox_2d = y['bbox2d']
        bbox_3d = y['bbox3d']
        cls = y['class']
            
        cls = self.class_dict[cls.lower()]
        
        # transform both image and label (note that the 2d and 3d bbox coords must both be scaled)
        im,bbox_2d,bbox_3d = self.random_scale_crop(im,bbox_2d,bbox_3d,imsize = 224, tighten = 0)
        
        im_array = np.array(im)
        
        if plot_3D:
            new_im = im_array.copy()
            coords = np.round(np.transpose(bbox_3d)).astype(int)
            #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
            edge_array= np.array([[0,1,0,1,1,0,0,0],
                                  [1,0,1,0,0,1,0,0],
                                  [0,1,0,1,0,0,1,1],
                                  [1,0,1,0,0,0,1,1],
                                  [1,0,0,0,0,1,0,1],
                                  [0,1,0,0,1,0,1,0],
                                  [0,0,1,0,0,1,0,1],
                                  [0,0,0,1,1,0,1,0]])
        
            # plot lines between indicated corner points
            for i in range(0,8):
                for j in range(0,8):
                    if edge_array[i,j] == 1:
                        cv2.line(new_im,(coords[i,0],coords[i,1]),(coords[j,0],coords[j,1]),(10,230,160),1)
        else:
            bbox_2d = bbox_2d.astype(int)
            new_im = cv2.rectangle(im_array,(bbox_2d[0],bbox_2d[1]),(bbox_2d[2],bbox_2d[3]),(10,230,160),2)
        
        plt.imshow(new_im)


class Test_Dataset_3D(data.Dataset):
    """
    Defines dataset and transforms for testing data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self,directory):

        self.im_dir = os.path.join(directory,"images")
        self.class_dict = {
                'car': 0,
                'van': 1,
                'truck': 2,
                'pedestrian':3,
                'person_sitting':4,
                'cyclist':5,
                'tram': 6,
                'misc': 7,
                'dontcare':8
                }
        
        # use os module to get a list of training image files
        # note that shuffling in dataloader is essential because these are ordered
        self.images =  [f for f in os.listdir(os.path.join(directory,"images"))]
        self.images.sort()
        
        with open(os.path.join(directory,"labels.cpkl"),'rb') as f:
            self.labels = pickle.load(f)
        
        
        self.labels = self.labels[0:6000]
        self.images = self.labels[0:6000]
        
        self.transforms = transforms.Compose([\
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        #Denotes the total number of samples
        return len(self.images) 

    def __getitem__(self, index):
        #Generates one sample of data
        file_name = os.path.join(self.im_dir,self.images[index])
        
        # load relevant files
        im = Image.open(file_name).convert('RGB')
        y = self.labels[index]
        bbox_2d = y['bbox2d']
        bbox_3d = y['bbox3d']
        calib = y['calib']
        cls = y['class']
            
        cls = self.class_dict[cls.lower()]
        
        # transform both image and label (note that the 2d and 3d bbox coords must both be scaled)
        im,bbox_2d,bbox_3d = self.scale_crop(im,bbox_2d,bbox_3d,imsize = 224)
        
        # normalize and convert image to tensor
        X = self.transforms(im)
        
        ##### Fix after here
        
        # normalize 2d and 3d corners wrt image size and convert to tensor
        # pad to wer times image size because bbox may fall outside of visible coordinates
        bbox_2d[0] = (bbox_2d[0]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
        bbox_2d[1] = (bbox_2d[1]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
        bbox_2d[2] = (bbox_2d[2]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
        bbox_2d[3] = (bbox_2d[3]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
        bbox_2d = torch.from_numpy(bbox_2d).float()
        
        bbox_3d[0,:] = (bbox_3d[0,:]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
        bbox_3d[1,:] = (bbox_3d[1,:]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
        bbox_3d = torch.from_numpy(np.reshape(bbox_3d,16)).float()
        
        calib = torch.from_numpy(calib).float()
        cls = torch.Tensor([cls])
        # y is a tuple of four tensors: cls,2dbbox, 3dbbox, and camera calibration matrix
        y = (cls,bbox_2d,bbox_3d,calib)
        
        return X, y
    
    
    def scale_crop(self,im,bb2d,bb3d,imsize = 224):
        """
        center-crop image and adjust labels accordingly
        """
    
        #define parameters for random transform
        # verfify that scale will at least accomodate crop size
        scale = imsize / max(im.size)
        
        # transform matrix
        im = transforms.functional.affine(im,0,(0,0),scale,0)
        (xsize,ysize) = im.size
        
        # clockwise from top left corner        
        # add 5th point corresponding to image center
        corners = np.array([[bb2d[0],bb2d[1]],[bb2d[2],bb2d[1]],
                            [bb2d[2],bb2d[3]],[bb2d[0],bb2d[3]],
                            [int(xsize/2),int(ysize/2)]])
        new_corners = corners * scale
        new_corners_3d = bb3d * scale

        # Resulting corners make a skewed, tilted rectangle - realign with axes
        bbox_2d = np.ones(4)
        bbox_2d[0] = np.min(new_corners[:4,0])
        bbox_2d[1] = np.min(new_corners[:4,1])
        bbox_2d[2] = np.max(new_corners[:4,0])
        bbox_2d[3] = np.max(new_corners[:4,1])
        
        # shift so transformed image center aligns with original image center
        xshift = xsize/2 - new_corners[4,0]
        yshift = ysize/2 - new_corners[4,1]
        bbox_2d[0] = bbox_2d[0] + xshift
        bbox_2d[1] = bbox_2d[1] + yshift
        bbox_2d[2] = bbox_2d[2] + xshift
        bbox_2d[3] = bbox_2d[3] + yshift
        
        new_corners_3d[0,:] = new_corners_3d[0,:] + xshift
        new_corners_3d[1,:] = new_corners_3d[1,:] + yshift
            
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
        bbox_2d[0] = bbox_2d[0] - crop_x
        bbox_2d[1] = bbox_2d[1] - crop_y
        bbox_2d[2] = bbox_2d[2] - crop_x
        bbox_2d[3] = bbox_2d[3] - crop_y
        
        new_corners_3d[0,:] = new_corners_3d[0,:] - crop_x
        new_corners_3d[1,:] = new_corners_3d[1,:] - crop_y
        
        return im, bbox_2d, new_corners_3d
    
    
    def show(self, index,plot_3D = False):
        #Generates one sample of data
        file_name = os.path.join(self.im_dir,self.images[index])
        
        # load relevant files
        im = Image.open(file_name).convert('RGB')
        y = self.labels[index]
        bbox_2d = y['bbox2d']
        bbox_3d = y['bbox3d']
        cls = y['class']
            
        cls = self.class_dict[cls.lower()]
        
        # transform both image and label (note that the 2d and 3d bbox coords must both be scaled)
        im,bbox_2d,bbox_3d = self.scale_crop(im,bbox_2d,bbox_3d,imsize = 224)
        
        im_array = np.array(im)
        
        if plot_3D:
            new_im = im_array.copy()
            coords = np.round(np.transpose(bbox_3d)).astype(int)
            #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
            edge_array= np.array([[0,1,0,1,1,0,0,0],
                                  [1,0,1,0,0,1,0,0],
                                  [0,1,0,1,0,0,1,1],
                                  [1,0,1,0,0,0,1,1],
                                  [1,0,0,0,0,1,0,1],
                                  [0,1,0,0,1,0,1,0],
                                  [0,0,1,0,0,1,0,1],
                                  [0,0,0,1,1,0,1,0]])
        
            # plot lines between indicated corner points
            for i in range(0,8):
                for j in range(0,8):
                    if edge_array[i,j] == 1:
                        cv2.line(new_im,(coords[i,0],coords[i,1]),(coords[j,0],coords[j,1]),(10,230,160),1)
        else:
            bbox_2d = bbox_2d.astype(int)
            new_im = cv2.rectangle(im_array,(bbox_2d[0],bbox_2d[1]),(bbox_2d[2],bbox_2d[3]),(10,230,160),2)
        
        plt.imshow(new_im)

class CNNnet(nn.Module):
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNNnet, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        self.vgg = models.vgg19(pretrained=True)
        del self.vgg.classifier[2:]

        # get size of some layers
        start_num = self.vgg.classifier[0].out_features
        mid_num = int(np.sqrt(start_num))
        cls_out_num = 8 
        reg_out_num = 4 # bounding box coords
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,cls_out_num,bias = True),
                          nn.Softmax(dim = 1)
                          )
        
        # define regressor
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,reg_out_num,bias = True),
                          nn.ReLU()
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
        
        return cls_out,reg_out


def train_model(model, cls_criterion,reg_criterion,reg_criterion2, optimizer, scheduler, 
                dataloaders,dataset_sizes, num_epochs=5, start_epoch = 0):
    """
    Alternates between a training step and a validation step at each epoch. 
    Validation results are reported but don't impact model weights
    """
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if epoch > 0:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                reg_target = labels[1].to(device)
                cls_target = labels[0].long().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    cls_outputs, reg_outputs = model(inputs)
                    
#                    # make copy of reg_outputs and zero if target is 0
#                    # so that bboxes are only learned for positive examples
#                    temp = cls_target#.unsqueeze(1)
#                    mask = torch.cat((temp,temp,temp,temp),1).float()
#                    mask.requires_grad = True
#                    reg_outputs_mod = torch.mul(reg_outputs,mask)
#                    reg_outputs = reg_outputs_mod 
#                    # note that the classification loss is done using class-wise probs rather 
#                    # than a single class label?
                    dummy_mask = torch.ones(reg_outputs.shape,requires_grad = True).to(device)
                    cls_loss = cls_criterion(cls_outputs,cls_target.squeeze())
                    reg_loss = reg_criterion(reg_outputs,reg_target,dummy_mask) + 0.5*reg_criterion2(reg_outputs,reg_target)
                    
                    
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        reg_loss.backward(retain_graph = True)
                        cls_loss.backward()
#                        print(model.regressor[0].weight.grad)
                        optimizer.step()
          
                # statistics
                running_loss += (reg_loss.item()+cls_loss.item()) * inputs.size(0)
                # here we need to define a function that checks the bbox iou with correct 
                
                # convert into class label rather than probs
                _,cls_outputs = torch.max(cls_outputs,1)
                # copy data to cpu and numpy arrays for scoring
                cls_pred = cls_outputs.data.cpu().numpy()
                reg_pred = reg_outputs.data.cpu().numpy()
                actual = np.concatenate((labels[1].numpy(),labels[0].numpy()),1)
                
                
                correct,bbox_acc = score_pred(cls_pred,reg_pred,actual)
                running_corrects += correct
    
                # verbose update
                count += 1
                if count % 20 == 0:
                    print("on minibatch {} -- correct: {} -- avg bbox iou: {} ".format(count,correct,bbox_acc))
                    print("cls: {}, reg: {}".format(cls_loss.item(),reg_loss.item()))
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase] * dataloaders['train'].batch_size
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                del best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        if epoch % 5 == 0:
            # save checkpoint
            PATH = "trial1_reg_init_checkpoint_{}.pt".format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                }, PATH)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def plot_batch(model,batch):
    corrects = batch[1][0].data.numpy()
    correct_bboxes = batch[1][1].data.numpy()    
    cls_outs, reg_out = model(batch[0].to(device))
    _, cls_out = torch.max(cls_outs,1)
     
    batch = batch[0].data.cpu().numpy()
    bboxes = reg_out.data.cpu().numpy()
    preds = cls_out.data.cpu().numpy()
   
    
    # define figure subplot grid
    batch_size = len(preds)
    row_size = min(batch_size,8)
    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        im =  batch[i].transpose((1,2,0))
        bbox = bboxes[i]
        pred = preds[i]
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = std * im + mean
        im = np.clip(im, 0, 1)
        
        class_dict = {
                0:'car',
                1:'van',
                2:'truck',
                3:'pedestrian',
                4:'person_sitting',
                5:'cyclist',
                6:'tram',
                7:'misc',
                8:'dontcare'
                }
        label = class_dict[pred]
        if False:
            correct= corrects[i,0]
            bbox = correct_bboxes[i]
            label = class_dict[correct]
        # transform bbox coords back into im pixel coords
        bbox = (bbox* 224*wer - 224*(wer-1)/2).astype(int)
        # plot bboxes
        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)

        if batch_size <= 8:
            axs[i].imshow(im)
            axs[i].set_title(label)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i//row_size,i%row_size].imshow(im)
            axs[i//row_size,i%row_size].set_title(label)
            axs[i//row_size,i%row_size].set_xticks([])
            axs[i//row_size,i%row_size].set_yticks([])
            plt.pause(.0001)


class Box_Loss(nn.Module):        
    def __init__(self):
        super(Box_Loss,self).__init__()
        
    def forward(self,output,target,mask,epsilon = 1e-07):
        """ Compute the bbox iou loss for target vs output using tensors to preserve
        gradients for efficient backpropogation"""
        
        # minx miny maxx maxy
        minx,_ = torch.max(torch.cat((output[:,0].unsqueeze(1),target[:,0].unsqueeze(1)),1),1)
        miny,_ = torch.max(torch.cat((output[:,1].unsqueeze(1),target[:,1].unsqueeze(1)),1),1)
        maxx,_ = torch.min(torch.cat((output[:,2].unsqueeze(1),target[:,2].unsqueeze(1)),1),1)
        maxy,_ = torch.min(torch.cat((output[:,3].unsqueeze(1),target[:,3].unsqueeze(1)),1),1)

        zeros = torch.zeros(minx.shape).unsqueeze(1).to(device)
        delx,_ = torch.max(torch.cat(((maxx-minx).unsqueeze(1),zeros),1),1)
        dely,_ = torch.max(torch.cat(((maxy-miny).unsqueeze(1),zeros),1),1)
        intersection = torch.mul(delx,dely)
        a1 = torch.mul(output[:,2]-output[:,0],output[:,3]-output[:,1])
        a2 = torch.mul(target[:,2]-target[:,0],target[:,3]-target[:,1])
        #a1,_ = torch.max(torch.cat((a1.unsqueeze(1),zeros),1),1)
        #a2,_ = torch.max(torch.cat((a2.unsqueeze(1),zeros),1),1)
        union = a1 + a2 - intersection 
        iou = intersection / (union + epsilon)
        #iou = torch.clamp(iou,0)
        mask_sum = mask.sum()
        return 1- iou.sum()/(mask_sum+epsilon)


class FocalLoss(nn.Module): # from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
#        wt = 5
#        weights = torch.FloatTensor([1,wt,wt,wt,wt,wt,wt,wt,0]).to(device)
        self.alpha = None #weights
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim= 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


        

#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # define start epoch for consistent labeling if checkpoint is reloaded
    checkpoint_file = None#"trial_1_large_bbox_init_checkpoint.pt"
    start_epoch = 0
    num_epochs = 75
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()    
    
    if sys.platform == 'linux':
        directory =  "/media/worklab/data_HDD/cv_data/KITTI/3D_object_parsed"#_cars_vans_only"
    else:
        directory = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object_parsed"
        
    train_data = Train_Dataset_3D(directory,max_scaling = 1.5)
    test_data = Test_Dataset_3D(directory)

    if False: # test the datasets
        idx = random.randint(0,len(train_data))
        X, (cls,bbox_2d,bbox_3d,calib) = train_data[idx]
        train_data.show(idx,plot_3D = True)
        test_data.show(idx,plot_3D = True)
    
    # create training params
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 0}

    try:
        trainloader
        testloader
    except:
        trainloader = data.DataLoader(train_data, **params)
        testloader = data.DataLoader(test_data, **params)
    print("Got dataloaders.")
       
    try:
        model
    except:
        # define CNN model
        model = CNNnet()
        model = model.to(device)
    print("Got model.")
    
    # define loss functions
    reg_criterion = Box_Loss()
    reg_criterion2 = nn.MSELoss()
    wt = 10
    weights = torch.FloatTensor([1,wt,wt,wt,wt,wt,wt,wt,0]).to(device)
    cls_criterion = FocalLoss()
    
    # all parameters are being optimized, not just fc layer
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum = 0.9)    
    # Decay LR by a factor of 0.5 every epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    

    # if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        #model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)
        model,_,_ = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
        print("Checkpoint loaded.")
            
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    
    
    if True:    
    # train model
        print("Beginning training.")
        model = train_model(model, cls_criterion, reg_criterion,reg_criterion2, optimizer, 
                            exp_lr_scheduler, dataloaders,datasizes,
                            num_epochs, start_epoch)
    
    plot_batch(model,next(iter(trainloader)))