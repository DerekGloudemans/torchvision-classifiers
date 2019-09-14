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

global wer #window expansion size, controls how far out of frame bbox can be predicted 
wer = 5
    
#--------------------------- Definitions section -----------------------------#
class Train_Dataset_3D(data.Dataset):
    """
    Defines dataset and transforms for training data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self,directory, max_scaling = 8):

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
            coords = np.transpose(bbox_3d).astype(int)
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

    






#class Test_Dataset(data.Dataset):
#    """
#    Defines dataset and transforms for training data. The positive images and 
#    negative images are stored in two different directories
#    """
#    def __init__(self, positives_path,negatives_path):
#        # use os module to get a list of positive and negative training examples
#        # note that shuffling is essential because examples are in order
#        pos_list =  [positives_path+'/test/'+f for f in os.listdir(positives_path+ '/test/')]
#        pos_list.sort()
#        neg_list =  [negatives_path+'/test/'+f for f in os.listdir(negatives_path+ '/test/')]
#        neg_list.sort() # in case not all files were correctly downloaded; in this case, the .tar file didn't download completely
#        self.file_list = pos_list + neg_list
#
#        # load labels (first 4 are bbox coors, then class (1 for positive, 0 for negative)
#        pos_labels = np.load(positives_path+'/labels/test_bboxes.npy')
#        pos_labels = pos_labels[:len(pos_list)]
#        neg_labels = np.load(negatives_path+'/labels/test_bboxes.npy')
#        self.labels = np.concatenate((pos_labels,neg_labels),0)
#        
#        self.transforms = transforms.Compose([\
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ])
#    
#    def __len__(self):
#        #Denotes the total number of samples
#        return len(self.file_list) 
#
#    def __getitem__(self, index):
#        #Generates one sample of data
#        file_name = self.file_list[index]
#        im = Image.open(file_name).convert('RGB')
#        y = self.labels[index,:]
#        
#        # transform, normalize and convert to tensor
#        im,y = self.scale_crop(im,y,imsize = 224)
#        X = self.transforms(im)
#        
#        # normalize y wrt image size and convert to tensor
#        y[0] = (y[0]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
#        y[1] = (y[1]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
#        y[2] = (y[2]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
#        y[3] = (y[3]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
#        y = torch.from_numpy(y).float()
#        
#        return X, y
#    
#    
#    def scale_crop(self,im,y,imsize = 224):
#        """
#        center-crop image and adjust labels accordingly
#        """
#    
#        #define parameters for random transform
#        # verfify that scale will at least accomodate crop size
#        scale = imsize / max(im.size)
#        
#        # transform matrix
#        im = transforms.functional.affine(im,0,(0,0),scale,0)
#        (xsize,ysize) = im.size
#        
#        # only transform coordinates for positive examples (negatives are [0,0,0,0,0])
#        # clockwise from top left corner
#        if y[4] == 1:
#    
#            # add 5th point corresponding to image center
#            corners = np.array([[y[0],y[1]],[y[2],y[1]],[y[2],y[3]],[y[0],y[3]],[int(xsize/2),int(ysize/2)]])
#            new_corners = corners * scale
#            
#            # realign with axes
#            y = np.ones(5)
#            y[0] = np.min(new_corners[:4,0])
#            y[1] = np.min(new_corners[:4,1])
#            y[2] = np.max(new_corners[:4,0])
#            y[3] = np.max(new_corners[:4,1])
#            
#            # shift so transformed image center aligns with original image center
#            xshift = xsize/2 - new_corners[4,0]
#            yshift = ysize/2 - new_corners[4,1]
#            y[0] = y[0] + xshift
#            y[1] = y[1] + yshift
#            y[2] = y[2] + xshift
#            y[3] = y[3] + yshift
#            y = y.astype(int)
#            
#            
#        # crop at image center
#        crop_x = xsize/2 -imsize/2
#        crop_y = ysize/2 -imsize/2
#        im = transforms.functional.crop(im,crop_y,crop_x,imsize,imsize)
#        
#        # transform bbox points into cropped coords
#        if y[4] == 1:
#            y[0] = y[0] - crop_x
#            y[1] = y[1] - crop_y
#            y[2] = y[2] - crop_x
#            y[3] = y[3] - crop_y
#        
#        return im,y.astype(float)
#    
#    def show(self, index):
#        #Generates one sample of data
#        file_name = self.file_list[index]
#        
#        im = Image.open(file_name).convert('RGB')
#        y = self.labels[index,:]
#        
#        # transform, normalize and convert to tensor
#        im,y = self.scale_crop(im,y,imsize = 224)
#        im_array = np.array(im)
#        y = y.astype(int)
#        new_im = cv2.rectangle(im_array,(y[0],y[1]),(y[2],y[3]),(20,190,210),2)
#        plt.imshow(new_im)


#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()    
    
    directory = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object_parsed"
    train_data = Train_Dataset_3D(directory,max_scaling = 1.5)

    idx = random.randint(0,len(train_data))
    X, (cls,bbox_2d,bbox_3d,calib) = train_data[idx]
    train_data.show(idx,plot_3D = True)
