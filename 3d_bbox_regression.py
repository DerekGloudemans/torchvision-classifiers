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
from parallel_regression_classification import load_model,SplitNet
from PackageNet import PackageNet

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
import sys
import numpy as np    
import _pickle as pickle
import random
import copy
import matplotlib.pyplot as plt
import math

global wer #window expansion size, controls how far out of frame bbox can be predicted 
wer = 1
    
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
                'misc': 7,
                'dontcare':8
                }
        
        # use os module to get a list of training image files
        # note that shuffling in dataloader is essential because these are ordered
        self.images =  [f for f in os.listdir(os.path.join(directory,"images"))]
        self.images.sort()
        
        with open(os.path.join(directory,"labels.cpkl"),'rb') as f:
            self.labels = pickle.load(f)
        
        
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
        im,bbox_2d,bbox_3d = self.random_scale_crop(im,bbox_2d,bbox_3d,imsize = 224, tighten = 0)
        
        # normalize and convert image to tensor
        X = self.transforms(im)
                
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
        scale = min(self.max_scaling,max(random.gauss(1,0.5),imsize/min(im.size))) # verfify that scale will at least accomodate crop size
        shear = 0 #(random.random()-0.5)*30 #angle
        rotation = (random.random()-0.5) * 60.0 #angle
        
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




def train_model(model, cls_criterion,reg_criterion, optimizer, scheduler, 
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
        for phase in ['train']:#, 'val']:
            if phase == 'train':
                if False: #disable for Adam
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
                reg_target = labels[2].to(device)
                cls_target = labels[0].long().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    cls_outputs, reg_outputs = model(inputs)
                    
#                    # make copy of reg_outputs and zero if target is 0
#                    # so that bboxes are only learned for positive examples
#                    temp = cls_target.unsqueeze(1)
#                    mask = torch.cat((temp,temp,temp,temp),1).float()
#                    mask.requires_grad = True
#                    reg_outputs_mod = torch.mul(reg_outputs,mask)
#                    
#                    # note that the classification loss is done using class-wise probs rather 
#                    # than a single class label?
                    cls_loss = cls_criterion(cls_outputs,cls_target.squeeze())
                    reg_loss = reg_criterion(reg_outputs,reg_target)
                    
                    
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #reg_loss.backward(retain_graph = True)
                        cls_loss.backward()
#                        print(model.regressor[0].weight.grad)
                        optimizer.step()
          
                # statistics
                running_loss += (reg_loss.item()+cls_loss.item()) * inputs.size(0)
                # here we need to define a function that checks the bbox iou with correct 
                # still wrong - must be across whole batch
                
                # convert into class label rather than probs
                _,cls_outputs = torch.max(cls_outputs,1)
                # copy data to cpu and numpy arrays for scoring
                cls_pred = cls_outputs.data.cpu().numpy()
                reg_pred = reg_outputs.data.cpu().numpy()
#                actual = labels.numpy()
                
                
#                correct,bbox_acc = score_pred(cls_pred,reg_pred,actual)
#                running_corrects += correct
    
                # verbose update
                count += 1
                if count % 20 == 0:
#                    print("on minibatch {} -- correct: {} -- avg bbox iou: {} ".format(count,correct,bbox_acc))
                    print("cls: {}, reg: {}".format(cls_loss.item(),reg_loss.item()))
                    
            epoch_loss = running_loss / dataset_sizes[phase]
#            epoch_acc = float(running_corrects) / dataset_sizes[phase] * dataloaders['train'].batch_size
            

#            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                phase, epoch_loss, epoch_acc))
#
#            # deep copy the model
#            if phase == 'val' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                del best_model_wts
#                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        if epoch % 1 == 0:
            # save checkpoint
            PATH = "packagenet_centered_checkpoint_{}.pt".format(epoch)
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
    model.eval()
    correct_labels = batch[1][2].data.numpy()
    correct_classes = batch[1][0].data.numpy()
    batch = batch[0].to(device)
    cls_outs, reg_out = model(batch)
    _, cls_out = torch.max(cls_outs,1)
     
    batch = batch.data.cpu().numpy()
    bboxes = reg_out.data.cpu().numpy()
    preds = cls_out.data.cpu().numpy()
    
    # define figure subplot grid
    batch_size = len(preds)
    row_size = min(batch_size,8)
    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        im =  batch[i].transpose((1,2,0))
        pred = preds[i]
        bbox = bboxes[i].reshape(2,-1)
        
        if False:   #plot correct labels instead
            bbox = correct_labels[i].reshape(2,-1)
        
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
        label = "{} -> {}".format(class_dict[int(correct_classes[i,0])],class_dict[pred])
        
        # transform bbox coords back into im pixel coords
        bbox = np.round(bbox* 224*wer - 224*(wer-1)/2)
        # plot bboxes
        
        if True:
            new_im = im
            coords = np.transpose(bbox).astype(int)
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
            for i2 in range(0,8):
                for j2 in range(0,8):
                    if edge_array[i2,j2] == 1:
                        cv2.line(new_im,(coords[i2,0],coords[i2,1]),(coords[j2,0],coords[j2,1]),(10,230,160),1)
        

        if batch_size <= 8:
            axs[i].imshow(new_im)
            axs[i].set_title(label)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i//row_size,i%row_size].imshow(new_im)
            axs[i//row_size,i%row_size].set_title(label)
            axs[i//row_size,i%row_size].set_xticks([])
            axs[i//row_size,i%row_size].set_yticks([])
            plt.pause(.0001)

#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # define start epoch for consistent labeling if checkpoint is reloaded
    checkpoint_file = None# "/home/worklab/Documents/Checkpoints/splitnet_centered5_checkpoint_13.pt" #"packagenet_centered_checkpoint_1.pt"
    start_epoch = 0
    num_epochs = 20
    
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
        model = PackageNet()
        model = model.to(device)
    print("Got model.")
    
    # define loss functions
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    
    # all parameters are being optimized, not just fc layer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=0.1,momentum = 0.9)    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    

    # if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        #model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)
        model,_,start_epoch = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
        print("Checkpoint loaded.")
            
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    
    
    if True:    
    # train model
        print("Beginning training.")
        model = train_model(model, cls_criterion, reg_criterion, optimizer, 
                            exp_lr_scheduler, dataloaders,datasizes,
                            num_epochs, start_epoch)
    
    plot_batch(model,next(iter(testloader)))