"""
This file defines a network architecture for regressing a 3D bounding box on 
KITTI objects.

This file is set up so that it can be easily run as a Jupyter Notebook on a Google
Colab instance. Wohoo!  

Datasets - two types of datasets are defined. Each takes in a mode parameter which
           can be set to either "train" or "test" for non-overlapping data:
    - Kitti 3d object image dataset 
    - 3D Cad Model dataset 

CNN - a pretrained VGG feature extractor is used with a few fully connected 
      layers at the end for final regression. This network outputs the parameters
      which define a prism in 2D space

Loss functions: yep, it's got em. It's a bit tricky to get a perfect loss function
    in 2D space, but a few were tried:
        - difference between orientation output angle and true angle
        - difference between length output and true length
        - MSE loss on corner points of prism
        - front, back, and bottom 2D iou loss from corner points
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
wer = 2
    
#--------------------------- Definitions section -----------------------------#
class Synthetic_Dataset_3D(data.Dataset):
    """
    Defines dataset and transforms for training data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self,directory, max_scaling = 4,mode= "train"):

        if mode == "train":
            self.im_dir = os.path.join(directory,"training_0")
        elif mode == "test":
            self.im_dir = os.path.join(directory,"testing_0")

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
        self.images =  [f for f in os.listdir(self.im_dir)]
        self.images.sort()
        
        with open(os.path.join(directory,"all_labels_0.cpkl"),'rb') as f:
            self.labels = pickle.load(f) # dictionary keyed by image file name
        
        # remove all images for which label was not correctly added
        removals = []
        for item in self.images:
            try:
                self.labels[item]
            except:
                removals.append(item)
        removals.reverse()
        for item in removals:
            self.images.remove(item)
            
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
        y = self.labels[self.images[index]]
        bbox_2d = y[1]
        bbox_3d = y[2]
        cls = y[0]
            
        cls =  self.class_dict[cls.lower()]
        
        # transform both image and label (note that the 2d and 3d bbox coords must both be scaled)
        im,bbox_2d,bbox_3d = self.scale_crop(im,bbox_2d,bbox_3d,imsize = 224)
        
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
        
#        bbox_3d_shifted = bbox_3d[:,[2,5,6,1,3,4,7,0]]
#        bbox_3d = bbox_3d_shifted
        # input is in order rtr rbr fbr ftr ftl fbl rbl rtl
        # needs to be order fbr fbl rbl rbr ftr ftl rtl rtr
        
        
        
        
#        new_x = bbox_3d[0,:4]
#        new_y = np.zeros([4]) 
#
#        new_y[0] = (bbox_3d[1,0]  +bbox_3d[1,1])/2
#        new_y[1] = (bbox_3d[1,2] +bbox_3d[1,3])/2
#        new_y[2] = (bbox_3d[1,4] +bbox_3d[1,5])/2
#        new_y[3] = (bbox_3d[1,6] +bbox_3d[1,7])/2
#
#        bbox_3d = np.concatenate((new_x,new_y),0)
        
        bbox_3d = torch.from_numpy(bbox_3d.reshape(16)).float()
        
        
        # clamp to prevent really large anomalous values
        bbox_3d = torch.clamp(bbox_3d,min = 0.0)
        bbox_3d = torch.clamp(bbox_3d,max = 1.0)
        
        cls = torch.LongTensor([cls])
        # y is a tuple of four tensors: cls,2dbbox, 3dbbox, and camera calibration matrix
        y = (cls,bbox_2d,bbox_3d)
        
        return X, y
    
    def scale_crop(self,im,bb2d,bb3d,imsize = 224):
        """
        center-crop image and adjust labels accordingly
        """
    
        #define parameters for random transform
        # verfify that scale will at least accomodate crop size
        scale = imsize*(1+np.random.rand()) / max(im.size)
        
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
        crop_x = int(random.gauss(im.size[0]/2,xsize/50/scale)-imsize/2)
        crop_y = int(random.gauss(im.size[1]/2,ysize/50/scale)-imsize/2)
        
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
    
    
    def show(self, index,plot_3D = True):
        #Generates one sample of data
        file_name = os.path.join(self.im_dir,self.images[index])
        
        # load relevant files
        im = Image.open(file_name).convert('RGB')
        y = self.labels[self.images[index]]
        bbox_2d = y[1]
        bbox_3d = y[2]
        cls = y[0]
            
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
        reg_out_num = 16 # bounding box coords
        

        
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
        reg_out = self.regressor(vgg_out)
        
        return reg_out


def train_model(model, reg_criterion,reg_criterion2, optimizer, scheduler, 
                dataloaders,dataset_sizes, num_epochs=5, start_epoch = 0):
    """
    Alternates between a training step and a validation step at each epoch. 
    Validation results are reported but don't impact model weights
    """
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 2.0
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
                reg_target = labels[2].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    reg_outputs = model(inputs)
                    reg_loss1 = reg_criterion(reg_outputs,reg_target) 
                    reg_loss2 = reg_criterion2(reg_outputs,reg_target)
                    reg_loss = reg_loss1 + reg_loss2
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        reg_loss.backward(retain_graph =False)
#                       print(model.regressor[0].weight.grad)
                        optimizer.step()
          
                # statistics
                running_loss += (reg_loss.item()) * inputs.size(0)
                # here we need to define a function that checks the bbox iou with correct 
                
#                correct,bbox_acc = score_pred(cls_pred,reg_pred,actual)
                correct = 0
                bbox_acc = 0
                running_corrects += correct
    
                # verbose update
                count += 1
                if count % 50 == 0:
                    #print("on minibatch {} -- correct: {} -- avg bbox iou: {} ".format(count,correct,bbox_acc))
                    print("iou loss: {}   MSE loss: {}".format(reg_loss1.item(),reg_loss2.item()))
            
            torch.cuda.empty_cache()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase] * dataloaders['train'].batch_size
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_acc:
                best_acc = epoch_loss
                del best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        if epoch % 5 == 0:
            # save checkpoint
            PATH = "trial4_checkpoint_{}.pt".format(epoch)
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



def plot_batch(model,batch,device = torch.device("cuda:0")):
    model.eval()
    correct_labels = batch[1][2].data.cpu().numpy()
    correct_classes = batch[1][0].data.cpu().numpy()
    batch = batch[0].to(device)
    reg_out = model(batch)
     
    batch = batch.data.cpu().numpy()
    bboxes = reg_out.data.cpu().numpy()

    # define figure subplot grid
    batch_size = len(reg_out)
    row_size = min(batch_size,8)
    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
    
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        im =  batch[i].transpose((1,2,0))
        
        bbox = bboxes[i]
#        new_out = np.zeros(16)
#        new_out[0:4] = bbox[0:4]
#        new_out[4:8] = bbox[0:4]
#        new_out[[8,9]] = bbox[4]
#        new_out[[10,11]] = bbox[5]
#        new_out[[12,13]] = bbox[6]
#        new_out[[14,15]] = bbox[7]
#        bbox = new_out.reshape(2,-1)
        bbox = bbox.reshape(2,-1)
         
        if False:   #plot correct labels instead
#            new_out = np.zeros(16)
#            correct_label = correct_labels[i]
#            new_out[0:4] = correct_label[0:4]
#            new_out[4:8] = correct_label[0:4]
#            new_out[[8,9]] = correct_label[4]
#            new_out[[10,11]] = correct_label[5]
#            new_out[[12,13]] = correct_label[6]
#            new_out[[14,15]] = correct_label[7]
#            bbox = new_out.reshape(2,-1)
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
        
#        label = "{}".format(class_dict[int(correct_classes[i,0])])
        label = "dummy"
        # transform bbox coords back into im pixel coords
        bbox = np.round(bbox* 224*wer - 224*(wer-1)/2)
        # plot bboxes
        
        if True:
            new_im = im.copy()
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
                    if edge_array[i2,j2] == 2:
                        cv2.line(new_im,(coords[i2,0],coords[i2,1]),(coords[j2,0],coords[j2,1]),(230,10,10),2)

        # get array from CV image (UMat style)
        #im = im.get()
        
        
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
    torch.cuda.empty_cache()
    

class Flip_Box_Loss(nn.Module):        
    def __init__(self):
        super(Flip_Box_Loss,self).__init__()
        
    def forward(self,output,target,epsilon = 1e-07):
        """ Compute the bbox iou loss for target vs output using tensors to preserve
        gradients for efficient backpropogation. input [x1,y_top,x2,y_bottom] where
        either x1 or x2 may be on the left for an example depending on 3D object
        orientation"""
        
        # 0 if x1 > x2, 1 if x1 < x2
        _, x1_on_left = torch.max(target[:,0:2],1)
        x1_on_left = x1_on_left.float()
        
        # compute areas as though x1 < x2
        minx,_ = torch.max(torch.cat((output[:,0].unsqueeze(1),target[:,0].unsqueeze(1)),1),1) # left x
        maxx,_ = torch.min(torch.cat((output[:,2].unsqueeze(1),target[:,2].unsqueeze(1)),1),1) # right x
        miny,_ = torch.max(torch.cat((output[:,1].unsqueeze(1),target[:,1].unsqueeze(1)),1),1) # top y
        maxy,_ = torch.min(torch.cat((output[:,3].unsqueeze(1),target[:,3].unsqueeze(1)),1),1) # bottom y

        zeros = torch.zeros(minx.shape).unsqueeze(1).to(device)
        delx,_ = torch.max(torch.cat(((maxx-minx).unsqueeze(1),zeros),1),1)
        dely,_ = torch.max(torch.cat(((maxy-miny).unsqueeze(1),zeros),1),1)
        intersection = torch.mul(delx,dely)
        a1 = torch.mul(output[:,2]-output[:,0],output[:,3]-output[:,1])
        a2 = torch.mul(target[:,2]-target[:,0],target[:,3]-target[:,1])
        union = a1 + a2 - intersection 
        iou = intersection / (union + epsilon)
        
        # compute areas as though x1 > x2
        maxx2,_ = torch.min(torch.cat((output[:,0].unsqueeze(1),target[:,0].unsqueeze(1)),1),1) # left x
        minx2,_ = torch.max(torch.cat((output[:,2].unsqueeze(1),target[:,2].unsqueeze(1)),1),1) # right x
        miny2,_ = torch.max(torch.cat((output[:,1].unsqueeze(1),target[:,1].unsqueeze(1)),1),1) # top y
        maxy2,_ = torch.min(torch.cat((output[:,3].unsqueeze(1),target[:,3].unsqueeze(1)),1),1) # bottom y

        delx2,_ = torch.max(torch.cat(((maxx2-minx2).unsqueeze(1),zeros),1),1)
        dely2,_ = torch.max(torch.cat(((maxy2-miny2).unsqueeze(1),zeros),1),1)
        intersection2 = torch.mul(delx2,dely2)
        a12 = torch.mul(output[:,0]-output[:,2],output[:,3]-output[:,1])
        a22 = torch.mul(target[:,0]-target[:,2],target[:,3]-target[:,1])
        union2 = a12 + a22 - intersection2
        iou2 = intersection2 / (union2 + epsilon)
        
        total_iou = torch.mul(iou,x1_on_left) + torch.mul(iou2,1-x1_on_left)
        return 1- total_iou.sum()/(output.shape[0]+epsilon)

class Flat_Loss(nn.Module):
    def __init__(self):
        super(Flat_Loss,self).__init__()
        
    def forward(self,output,target,epsilon = 1e-07):
        """ Computes 2D bbox iou by flattening 3D bbox prediction and compares
        with target"""
        # get approx 2D bbox for back of pred object
        lefx = torch.mean(torch.cat((output[:,3].unsqueeze(1),output[:,7].unsqueeze(1)),1),1).unsqueeze(1)
        rigx = torch.mean(torch.cat((output[:,6].unsqueeze(1),output[:,2].unsqueeze(1)),1),1).unsqueeze(1)
        topy = torch.mean(torch.cat((output[:,14].unsqueeze(1),output[:,15].unsqueeze(1)),1),1).unsqueeze(1)
        boty = torch.mean(torch.cat((output[:,10].unsqueeze(1),output[:,11].unsqueeze(1)),1),1).unsqueeze(1)
        # get approx 2D bbox for front of pred object
        lefx2 = torch.mean(torch.cat((output[:,0].unsqueeze(1),output[:,4].unsqueeze(1)),1),1).unsqueeze(1)
        rigx2 = torch.mean(torch.cat((output[:,1].unsqueeze(1),output[:,5].unsqueeze(1)),1),1).unsqueeze(1)
        topy2 = torch.mean(torch.cat((output[:,13].unsqueeze(1),output[:,12].unsqueeze(1)),1),1).unsqueeze(1)
        boty2 = torch.mean(torch.cat((output[:,9].unsqueeze(1),output[:,8].unsqueeze(1)),1),1).unsqueeze(1)
        # get approx 2D bbox for bottom of pred object (front of object is considered top)
        lefx3 = torch.mean(torch.cat((output[:,0].unsqueeze(1),output[:,3].unsqueeze(1)),1),1).unsqueeze(1)
        rigx3 = torch.mean(torch.cat((output[:,1].unsqueeze(1),output[:,2].unsqueeze(1)),1),1).unsqueeze(1)
        topy3 = torch.mean(torch.cat((output[:,8].unsqueeze(1),output[:,9].unsqueeze(1)),1),1).unsqueeze(1)
        boty3 = torch.mean(torch.cat((output[:,10].unsqueeze(1),output[:,11].unsqueeze(1)),1),1).unsqueeze(1)
        
        flat_out1   = torch.cat((lefx,topy,rigx,boty),1)
        flat_out2  = torch.cat((lefx2,topy2,rigx2,boty2),1)
        flat_out3  = torch.cat((lefx3,topy3,rigx3,boty3),1)
        
        #concat front, back and bottom
        flat_out = torch.cat((flat_out1,flat_out2,flat_out3),0)
        
        
        # get approx 2D bbox for back of pred object
        lefx4 = torch.mean(torch.cat((target[:,3].unsqueeze(1),target[:,7].unsqueeze(1)),1),1).unsqueeze(1)
        rigx4 = torch.mean(torch.cat((target[:,6].unsqueeze(1),target[:,2].unsqueeze(1)),1),1).unsqueeze(1)
        topy4 = torch.mean(torch.cat((target[:,14].unsqueeze(1),target[:,15].unsqueeze(1)),1),1).unsqueeze(1)
        boty4 = torch.mean(torch.cat((target[:,10].unsqueeze(1),target[:,11].unsqueeze(1)),1),1).unsqueeze(1)
        # get approx 2D bbox for front of pred object
        lefx5 = torch.mean(torch.cat((target[:,0].unsqueeze(1),target[:,4].unsqueeze(1)),1),1).unsqueeze(1)
        rigx5 = torch.mean(torch.cat((target[:,1].unsqueeze(1),target[:,5].unsqueeze(1)),1),1).unsqueeze(1)
        topy5 = torch.mean(torch.cat((target[:,13].unsqueeze(1),target[:,12].unsqueeze(1)),1),1).unsqueeze(1)
        boty5 = torch.mean(torch.cat((target[:,9].unsqueeze(1),target[:,8].unsqueeze(1)),1),1).unsqueeze(1)
        # get approx 2D bbox for bottom of pred object (front of object is considered top)
        lefx6 = torch.mean(torch.cat((target[:,0].unsqueeze(1),target[:,3].unsqueeze(1)),1),1).unsqueeze(1)
        rigx6 = torch.mean(torch.cat((target[:,1].unsqueeze(1),target[:,2].unsqueeze(1)),1),1).unsqueeze(1)
        topy6 = torch.mean(torch.cat((target[:,8].unsqueeze(1),target[:,9].unsqueeze(1)),1),1).unsqueeze(1)
        boty6 = torch.mean(torch.cat((target[:,10].unsqueeze(1),target[:,11].unsqueeze(1)),1),1).unsqueeze(1)
        
        flat_targ1   = torch.cat((lefx4,topy4,rigx4,boty4),1)
        flat_targ2  = torch.cat((lefx5,topy5,rigx5,boty5),1)
        flat_targ3  = torch.cat((lefx6,topy6,rigx6,boty6),1)
        
        #concat front, back and bottom
        flat_targ = torch.cat((flat_targ1,flat_targ2,flat_targ3),0)

        flip_box_loss = Flip_Box_Loss()
        
        return flip_box_loss(flat_out,flat_targ)


        

#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # define start epoch for consistent labeling if checkpoint is reloaded
    checkpoint_file =  "trial4_pre_init.pt"
    start_epoch = 0
    num_epochs = 150
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()    
    
    directory = "/media/worklab/data_HDD/cv_data/Synthetic_car_model_ims"
        
    train_data = Synthetic_Dataset_3D(directory,max_scaling = 1.5,mode = "train")
    test_data = Synthetic_Dataset_3D(directory, mode = "test")

    if False: # test the datasets
        idx = random.randint(0,len(train_data))
        X, (cls,bbox_2d,bbox_3d,calib) = train_data[idx]
        train_data.show(idx,plot_3D = True)
        test_data.show(idx,plot_3D = True)
    
    # create training params
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 4}

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
    reg_criterion = Flat_Loss()
    reg_criterion2 = nn.MSELoss()
    
    # all parameters are being optimized, not just fc layer
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum = 0.9)    
    # Decay LR by a factor of 0.5 every epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    

    # if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)
        #model,_,_ = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
        print("Checkpoint loaded.")
            
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    
    
    if True:    
    # train model
        print("Beginning training on {}.".format(device))
        model = train_model(model, reg_criterion,reg_criterion2, optimizer, 
                            exp_lr_scheduler, dataloaders,datasizes,
                            num_epochs, start_epoch)
    
    plot_batch(model,next(iter(trainloader)))