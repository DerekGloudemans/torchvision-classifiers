
# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division


# torch and specific torch packages for convenience
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch import multiprocessing

# for convenient data loading, image representation and dataset management
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.ndimage import affine_transform
import cv2

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
wer = 2

"""### 3. Download the pretrained torchvision model and (if necessary) reload checkpoint

#### Define Model 
The model used for regression is built upon the Inception V3 base provided by Pytorch as a pretrained model. On top of this, a 3 layer, fully connected regression head is added, replacing the original classifier of Inception V3. The model outputs 11 parameters that fully define a 2D projection of a rectangular prism.

* Edit: Actually we're using VGG for now but eventually will switch.
"""

class VGGBoxNet(nn.Module):
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate the CNN layer and the regression layer
        and assign them as member variables.
        """
        super(VGGBoxNet, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        self.vgg = models.vgg19(pretrained=True)
        
        
        
        # get size of some layers
        start_num = 25088
        mid_num = int(np.sqrt(start_num))
        reg_out_num = 9 # bounding box coords
        
        # define regressor
        self.vgg.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.BatchNorm1d(mid_num),
                          nn.ReLU(),
                          nn.Linear(mid_num,reg_out_num,bias = True),
                          nn.BatchNorm1d(reg_out_num),
                          nn.Sigmoid()
                          )
        
        for layer in self.vgg.classifier:
            try:
                layer.weight.data = torch.randn(layer.weight.data.shape)
                layer.bias.data = torch.randn(layer.bias.data.shape)

            except:
                pass
            
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        reg_out = self.vgg(x)

        return reg_out

"""#### Load model"""

model = VGGBoxNet()   
model.cuda()

"""## Load Datasets (we'll move definitions to a separate .py file later)

### 1. Define Dataset Class
"""

class Kitti_3D_Object_Dataset(data.Dataset):
    """
    Defines dataset and transforms for training data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self,directory, max_scaling = 2,mode = "training"):

        self.im_dir = os.path.join(directory,"images")
        self.max_scaling = max_scaling
        self.mode = mode
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
            
        
        # use first 80% of data if training, last 20% if testing
        # note that it is assumed classes are more or less randomly distributed by index
        num = len(self.images)
        if self.mode == "training":
            self.images = self.images[:int(num*0.8)]
            self.labels = self.labels[:int(num*0.8)]
        elif self.mode == "testing":
            self.images = self.images[int(num*0.8):]
            self.labels = self.labels[int(num*0.8):]
        else:
          raise ValueError("Invalid mode for dataset.")
        
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
       
        bbox_3d[0,:] = (bbox_3d[0,:]+im.size[0]*(wer-1)/2)/(im.size[0]*wer)
        bbox_3d[1,:] = (bbox_3d[1,:]+im.size[1]*(wer-1)/2)/(im.size[1]*wer)
        
        # generate box parameters based on corner points
        # x,y,w,h,l,rot,sin_orientation,cos_orientation,flip
        
        avg_x_front = np.average(bbox_3d[0,[0,1,4,5]])
        avg_y_front = np.average(bbox_3d[1,[0,1,4,5]])
        avg_x_back = np.average(bbox_3d[0,[2,3,6,7]])
        avg_y_back = np.average(bbox_3d[1,[2,3,6,7]])
         
        x_width_front =  ((bbox_3d[0,1]+bbox_3d[0,5]) - (bbox_3d[0,0]+bbox_3d[0,4]))/4.0
        y_height_front = ((bbox_3d[1,1]+bbox_3d[1,0]) - (bbox_3d[1,5]+bbox_3d[1,4]))/4.0
        x_width_back =   ((bbox_3d[0,2]+bbox_3d[0,6]) - (bbox_3d[0,7]+bbox_3d[0,3]))/4.0
        y_height_back =   ((bbox_3d[1,2]+bbox_3d[1,3]) - (bbox_3d[1,7]+bbox_3d[1,6]))/4.0
        #l_ratio = (x_width_back/x_width_front + y_height_back/y_height_front)/2.0 -0.5
        l_ratio = (y_height_back/y_height_front) -0.5 # only use height since width is more susceptible to large swings when narrow

        w = np.abs((x_width_front)/2)#  + x_width_back)/4 don't fully know why it's divided by 2 here
        h = (y_height_front)/2# + y_height_back)/4
        rot = 0.5
        l = np.sqrt((avg_x_front-avg_x_back)**2 + (avg_y_front-avg_y_back)**2)
        flip = int(np.abs(y["alpha"]) >= np.pi/2)
        
        sin_o = (((avg_y_back-avg_y_front)/l)+1)/2 # shift from [-1,1] to [0,1]
        cos_o = (((avg_x_back-avg_x_front)/l)+1)/2 
        box_params = np.array([avg_x_front,avg_y_front,w,h,l,rot,sin_o,cos_o,l_ratio])
        angle_params = np.array([(np.sin(y['alpha'])+1)/2,(np.cos(y['alpha'])+1)/2])
        
        # tensorize
        bbox_2d = torch.from_numpy(bbox_2d).float()
        bbox_3d = torch.from_numpy(np.reshape(bbox_3d,16)).float()
        box_params = torch.from_numpy(box_params).float()
        angle_params = torch.from_numpy(angle_params).float()

        # clamp to prevent really large anomalous values
        bbox_3d = torch.clamp(bbox_3d,min = 0.0)
        bbox_3d = torch.clamp(bbox_3d,max = 1.0)
        
        box_params = torch.clamp(box_params,min = 0.0)
        box_params = torch.clamp(box_params,max = 1.0)
        
        
        calib = torch.from_numpy(calib).float()
        cls = torch.LongTensor([cls])
        # y is a tuple of four tensors: cls,2dbbox, 3dbbox, and camera calibration matrix
        y = (cls,bbox_2d,bbox_3d,calib,box_params,angle_params)
        
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
        rotation = (random.random()-0.5) * 0#20.0 #angle
        
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

"""### 2. Load Datasets
Using the pytorch dataset and dataloader utilities for multiprocessing
"""

# define datasets
train_data = Kitti_3D_Object_Dataset("/media/worklab/data_HDD/cv_data/KITTI/3D_object_parsed",mode = "training")
test_data = Kitti_3D_Object_Dataset("/media/worklab/data_HDD/cv_data/KITTI/3D_object_parsed",mode = "testing")

# plot an example
#idx = random.randint(0,len(train_data))
#train_data.show(idx,plot_3D = True)

# set multiprocessing mode
try:
    torch.multiprocessing.set_start_method('spawn')    
except:
    pass
  
  
# dataloader params
params = {'batch_size': 32,
          'shuffle': True,
          'pin_memory': False,
          'num_workers': 0}

trainloader = data.DataLoader(train_data, **params)
testloader = data.DataLoader(test_data, **params)

"""## Define some utility functions
Again, these are outta here in the long term once I create an associated util.py file
"""

def plot_batch(model,batch,device = torch.device("cuda:0")):
    """
    Evaluate a batch of inputs and plot the result
    """
    model.eval()
    correct_labels = batch[1][2].data.cpu().numpy()
    correct_classes = batch[1][0].data.cpu().numpy()
    true_param_corners = tensor_prism(batch[1][4].to(device)).data.cpu().numpy()
    batch = batch[0].to(device)
    reg_out = model(batch)
    reg_out = tensor_prism(reg_out)
    batch = batch.data.cpu().numpy()
    bboxes = reg_out.data.cpu().numpy()
    
    #bboxes = true_param_corners
    # define figure subplot grid
    batch_size = len(reg_out)
    row_size = min(batch_size,8)
    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True,figsize = (20,10))
    
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        im =  batch[i].transpose((1,2,0))
        bbox = bboxes[i].reshape(2,-1)
        # transform bbox coords back into im pixel coords
        bbox = np.round(bbox* 224*wer - 224*(wer-1)/2)
        
        
        bbox2 = correct_labels[i].reshape(2,-1)
        bbox2 = np.round(bbox2* 224*wer - 224*(wer-1)/2)

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
       
        # plot bboxes        
        if True:
            new_im = im.copy()
            coords = np.transpose(bbox).astype(int)
            coords2 = np.transpose(bbox2).astype(int)
            #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
            edge_array= np.array([[0,1,0,1,1,0,0,0],
                                  [1,0,1,0,0,1,0,0],
                                  [0,1,0,1,0,0,1,1],
                                  [1,0,1,0,0,0,1,1],
                                  [1,0,0,0,0,1,0,1],
                                  [0,1,0,0,1,0,1,0],
                                  [0,0,1,0,0,1,0,1],
                                  [0,0,0,1,1,0,1,0]])
            #edge_array[i%8,:] = 1
        
            # plot lines between indicated corner points
            for i2 in range(0,8):
                for j2 in range(0,8):
                    if edge_array[i2,j2] == 1:
                        cv2.line(new_im,(coords[i2,0],coords[i2,1]),(coords[j2,0],coords[j2,1]),(0,5,30),1)
                        if True:
                            cv2.line(new_im,(coords2[i2,0],coords2[i2,1]),(coords2[j2,0],coords2[j2,1]),(10,255,0),1)
 
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
    plt.pause(5.0)
    fig.show()
    plt.pause(5.0)
    torch.cuda.empty_cache()

def load_model(checkpoint_file,model,optimizer):
    """
    Reloads a checkpoint, loading the model and optimizer state_dicts and 
    setting the start epoch
    """
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model,optimizer,epoch

def train_model(model, criterion, optimizer, scheduler, 
                dataloaders,datasizes, device, 
                num_epochs=5, start_epoch = 0, show_output_every = False,save_every = 5):
    """
    Alternates between a training step and a validation step at each epoch. 
    Validation results are reported but don't impact model weights
    """
    start_time = time.time()
    
    # for storing best model info
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    best_epoch = 0
    
    
    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            epoch_start_time = time.time()
            
            if phase == 'train':
                if epoch > 0:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            # for storing total epoch losses
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                
                # send X and y to GPU
                inputs = inputs.to(device)
                reg_target1 = labels[2].to(device)
                reg_target2 = labels[4].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # update gradients if in train
                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(False):

                    # forward pass
                    reg_outputs = model(inputs)

                    # get loss
                    reg_loss = criterion(reg_outputs,reg_target2,reg_target1) 
                   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        reg_loss.backward()
                        #temp = (model.vgg.classifier[-2].weight.grad).cpu().data.numpy()
                        optimizer.step()

                # statistics
                running_loss += (reg_loss.item()) * inputs.size(0)               
    
                # verbose update
                count += 1
                if count % 100 == 0 or (epoch == 0 and count % 10 == 0):
                    print("on minibatch {} -- loss: {}".format(count,reg_loss))
                    
            torch.cuda.empty_cache()
            epoch_loss = running_loss / datasizes[phase]            

            print('Epoch {} Loss: {:.4f} in {:.1f} sec.'.format(
                phase, epoch_loss, time.time() - epoch_start_time))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                del best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

        if show_output_every:
            plot_batch(model,next(iter(testloader)))
        
        torch.cuda.empty_cache()     
        
        # early stopping
        STOP = False
        if best_epoch + 5 < epoch: # 5 epochs since improvement
            STOP = True
        
        if epoch % save_every == 0 or STOP:
            # save checkpoint
            PATH = "trial5_1_checkpoint_{}.pt".format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                }, PATH)
        if STOP:
          break
            
            
            
       

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def tensor_prism(params,FLIP = False):
    """
    returns an N x 16 tensor of prism corner coords maintaining backpropogation abilities
    params - N x 11 tensor of parameters that define a rectangular prism in 
             2-dimensional space [x,y,w,h,l,front_angle,sin_o,cos_o,flip,l_ratio]
             x,y - int - center coordinates of front of box
             w,h - int - width and avg height of front of box
             l - int - distance between center of front of box and center of back of box
             front_angle - float in [0,1] - defines rotation of front rectangle in range -90,90
             sin_o,cos_o - float in [-1,1] - uniquely and continuously define 
            orinetation angle theta 
            w_shrink, ... - float in [0,1] control the vanishing rate along each axis
    """
    device = params.get_device()
    num_inputs = params.shape[0]

    # create new tensor to hold output values
    points = torch.zeros((num_inputs,2,8),requires_grad = False).to(device)

    # define front rectangle w and h
    points[:,0,[0,3]] = torch.index_select(params,1,torch.tensor([2]).to(device)).repeat(1,2)
    points[:,0,[1,2]] = -1*torch.index_select(params,1,torch.tensor([2]).to(device)).repeat(1,2)
    points[:,1,[0,1]] = torch.index_select(params,1,torch.tensor([3]).to(device)).repeat(1,2)
    points[:,1,[2,3]] = -1*torch.index_select(params,1,torch.tensor([3]).to(device)).repeat(1,2)
    # rotate front rectangle
    rot = ((torch.index_select(params,1,torch.tensor([5]).to(device))-0.5) * np.pi)
    rot_mat = torch.cat((torch.cos(rot),-torch.sin(rot),torch.sin(rot),torch.cos(rot)),1).view(num_inputs,2,2)
    points = torch.matmul(rot_mat,points) + points

    # shift front rectangle to location
    points[:,0,:4] = points[:,0,:4] + torch.index_select(params,1,torch.tensor([0]).to(device)).repeat(1,4)
    points[:,1,:4] = points[:,1,:4] + torch.index_select(params,1,torch.tensor([1]).to(device)).repeat(1,4)
    
    # get projection angle and distance of rear of box
    orientation_angle = torch.atan2((torch.index_select(params,1,torch.tensor([6]).to(device))*2)-1, \
                                    (torch.index_select(params,1,torch.tensor([7]).to(device))*2)-1)
    orient = orientation_angle + rot
    x_shift = torch.cos(orient) * torch.index_select(params,1,torch.tensor([4]).to(device))
    y_shift = torch.sin(orient) * torch.index_select(params,1,torch.tensor([4]).to(device))

    # get rear of box
    points[:,0,4:8] = points[:,0,0:4] + x_shift.repeat(1,4)
    points[:,1,4:8] = points[:,1,0:4] + y_shift.repeat(1,4)

#    # apply length ratio(if less than 1, shrinks back face)
    l_ratio = (params[:,8] + 0.5).unsqueeze(1).unsqueeze(1).repeat(1,2,4)
    l_points = torch.tensor([4,5,6,7]).to(device)
    l_avg = (torch.mean(torch.index_select(points,2,l_points),axis = 2).view(num_inputs,2,1)).repeat(1,1,4)
    l_diff =  torch.index_select(points,2,l_points) - l_avg
    points[:,:,l_points] = l_avg + torch.mul(l_diff,l_ratio)
    
    # map easy working space into correct space
    #in  fbr fbl ftl ftr rbr rbl rtl rtr
    #out fbr fbl rbl rbr ftr ftl rtl rtr

    # flip so that indices are correct in examples where front is behind back
    #flip = torch.index_select(params,1,torch.tensor([XX]).to(device)).unsqueeze(1).repeat(1,2,8)
    if FLIP:
        indices = torch.tensor([1,0,4,5,2,3,7,6]).to(device)
    else:
        indices = torch.tensor([0,1,5,4,3,2,6,7]).to(device)
    #flip_indices = torch.tensor([1,0,4,5,2,3,7,6]).to(device)
    #points = torch.mul(torch.index_select(points,2,indices),(1.0-flip)) + \
    #         torch.mul(torch.index_select(points,2,flip_indices),flip)
    
    points = torch.index_select(points,2,indices)
    return points

"""## Define Loss Functions

### 1. nn.MSELoss() on box corners
"""

class Prism_MSE_Loss(nn.Module):        
    def __init__(self):
        super(Prism_MSE_Loss,self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,preds,target):
        """ Compute the prism corner coords and calculate 
            MSE loss compared to target corner coords"""
        
        # get bbox coords from params
        outputs = tensor_prism(preds)
        target = target.view(outputs.shape)
        
        diff = outputs-target
        mul = torch.mul(diff,diff)
        mul_sum = torch.sum(mul,[1,2])
        sqrt = torch.sqrt(mul_sum)
        
        flip_outputs = tensor_prism(preds,FLIP = True)
        diff_flip = flip_outputs - target
        mul_flip = torch.mul(diff_flip,diff_flip)
        mul_sum_flip = torch.sum(mul_flip,[1,2])
        sqrt_flip = torch.sqrt(mul_sum_flip)
        
        min_loss,_ = torch.min(torch.cat((sqrt_flip.unsqueeze(1),sqrt.unsqueeze(1)),1),1)
        return torch.mean(min_loss)
    
"""### 3. Loss to constrain sin<sup>2</sup> + cos<sup>2</sup> = 1"""

class Pythagorean_Loss(nn.Module):        
    def __init__(self):
        super(Pythagorean_Loss,self).__init__()
        self.mse = nn.MSELoss()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
    def forward(self,preds):
        """ To learn that sin^2 + cos^2 = 1"""
        
        out = torch.mul(preds[:,6]*2-1,preds[:,6]*2-1) + torch.mul(preds[:,7]*2-1,preds[:,7]*2-1).float()
        targ = torch.ones(out.shape,requires_grad = False).float().to(self.device)

        return self.mse(out,targ)

"""### 4. Combination Loss"""

class Combination_Loss(nn.Module):        
    def __init__(self,pythagorean=0,param=0,prism = 0):
        super(Combination_Loss,self).__init__()
        self.pyth = pythagorean
        self.param = param
        self.prism = prism
        self.pythagorean_loss = Pythagorean_Loss()
        self.param_loss = Param_MSE_Loss()
        self.prism_loss = Prism_MSE_Loss()
        
    def forward(self,inputs,param_target,point_target):
        """ Combine constituent losses"""
        
        return self.pyth  * self.pythagorean_loss(inputs)    + \
               self.param * self.param_loss(inputs, param_target) + \
               self.prism * self.prism_loss(inputs,point_target)

"""### 5. MSE loss on box params"""

class Param_MSE_Loss(nn.Module):        
    def __init__(self):
        super(Param_MSE_Loss,self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,preds,target):
        """ Compute the prism corner coords and calculate 
            MSE loss compared to target corner coords"""
        
        # get bbox coords from params
       
        return self.mse(preds,target)

"""## Main Training Loop

### 1. Input training parameters
"""

save_every = 1
num_epochs = 50
learning_rate_init = 0.001

checkpoint_file = None
show_output_every = True
restart_from_epoch_0 = True

criterion = Combination_Loss(param = 0.5,prism = 0.4, pythagorean = 0.1)

"""### 2. Prep for running"""

start_epoch = 0

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.empty_cache()

# all parameters are being optimized, not just fc layer
optimizer = optim.SGD(model.parameters(), lr=learning_rate_init,momentum = 0.9)    

# Decay LR by a factor of 0.8 every epoch
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


# if checkpoint specified, load model and optimizer weights from checkpoint
if checkpoint_file != None:
    if restart_from_epoch_0:
        model,_,_ = load_model(checkpoint_file, model, optimizer) 
    else:
        model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)

        
        
# group dataloaders
dataloaders = {"train":trainloader, "val": testloader}
datasizes = {"train": len(train_data), "val": len(test_data)}

args =   {"model": model,
          "criterion": criterion,
          "optimizer": optimizer,
          "scheduler": exp_lr_scheduler,
          "dataloaders":dataloaders,
          "datasizes": datasizes,
          "device": device,
          "num_epochs": num_epochs,
          "start_epoch": start_epoch,
          "show_output_every":show_output_every,
          "save_every": save_every
         }

"""### 3. Train"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

# train model
torch.cuda.empty_cache()
print("Beginning training on {}.".format(device))
model = train_model(**args)

batch = next(iter(trainloader))
#plot_batch(model,batch)
labels = batch[1]
params = labels[4]
preds = model(batch[0].to(device))
preds2 = preds.data.cpu().numpy()
test = Pythagorean_Loss()
loss = test(params.to(device))
loss2 = test(preds.to(device))
loss3 = criterion(preds.to(device),params.to(device))
loss4 = criterion(params.to(device),params.to(device))
