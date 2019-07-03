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
        
        # transform, normalize and convert to tensor
        im,y = self.random_affine_crop(im,y,imsize = 224, tighten = 0.05)
        X = self.transforms(im)
        
        # normalize y wrt image size and convert to tensor
        y[0] = float(y[0])/im.size[0]
        y[1] = float(y[1])/im.size[1]
        y[2] = float(y[2])/im.size[0]
        y[3] = float(y[3])/im.size[1]
        y = torch.from_numpy(y).float()
        
        return X, y
    
    def random_affine_crop(self,im,y,imsize = 224,tighten = 0.05):
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
        
        # only transform coordinates for positive examples (negatives are [0,0,0,0,0])
        # clockwise from top left corner
        if y[4] == 1:
            
            # image transformation matrix
            shear = math.radians(-shear)
            rotation = math.radians(-rotation)
            M = np.array([[scale*np.cos(rotation),-scale*np.sin(rotation+shear)], 
                          [scale*np.sin(rotation), scale*np.cos(rotation+shear)]])
            
            
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
            if tighten != 0:
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
        
        return im,y
    
    def show(self, index):
        #Generates one sample of data
        file_name = self.file_list[index]
        
        im = Image.open(file_name).convert('RGB')
        y = self.labels[index,:]
        
        # transform, normalize and convert to tensor
        im,y = self.random_affine_crop(im,y,imsize = 224, tighten = 0.05)
        im_array = np.array(im)
        new_im = cv2.rectangle(im_array,(y[0],y[1]),(y[2],y[3]),(10,230,160),2)
        plt.imshow(new_im)
    

def load_bbox_mat():
    """
    Used to parse raw annotation files into numpy arrays.
    """
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


class Test_Dataset(data.Dataset):
    """
    Defines dataset and transforms for training data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self, positives_path,negatives_path):

        # use os module to get a list of positive and negative training examples
        # note that shuffling is essential because examples are in order
        pos_list =  [positives_path+'/test/'+f for f in os.listdir(positives_path+ '/test/')]
        pos_list.sort()
        neg_list =  [negatives_path+'/test/'+f for f in os.listdir(negatives_path+ '/test/')]
        neg_list.sort() # in case not all files were correctly downloaded; in this case, the .tar file didn't download completely
        self.file_list = pos_list + neg_list

        # load labels (first 4 are bbox coors, then class (1 for positive, 0 for negative)
        pos_labels = np.load(positives_path+'/labels/test_bboxes.npy')
        pos_labels = pos_labels[:len(pos_list)]
        neg_labels = np.load(negatives_path+'/labels/test_bboxes.npy')
        self.labels = np.concatenate((pos_labels,neg_labels),0)
        
        self.transforms = transforms.Compose([\
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
        
        # transform, normalize and convert to tensor
        im,y = self.scale_crop(im,y,imsize = 224)
        X = self.transforms(im)
        
        # normalize y wrt image size and convert to tensor
        y[0] = float(y[0])/im.size[0]
        y[1] = float(y[1])/im.size[1]
        y[2] = float(y[2])/im.size[0]
        y[3] = float(y[3])/im.size[1]
        y = torch.from_numpy(y).float()
        
        return X, y
    
    
    def scale_crop(self,im,y,imsize = 224):
        """
        center-crop image and adjust labels accordingly
        """
    
        #define parameters for random transform
        # verfify that scale will at least accomodate crop size
        scale = imsize / max(im.size)
        
        # transform matrix
        im = transforms.functional.affine(im,0,(0,0),scale,0)
        (xsize,ysize) = im.size
        
        # only transform coordinates for positive examples (negatives are [0,0,0,0,0])
        # clockwise from top left corner
        if y[4] == 1:
    
            # add 5th point corresponding to image center
            corners = np.array([[y[0],y[1]],[y[2],y[1]],[y[2],y[3]],[y[0],y[3]],[int(xsize/2),int(ysize/2)]])
            new_corners = corners * scale
            
            # realign with axes
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
            
            
        # crop at image center
        crop_x = xsize/2 -imsize/2
        crop_y = ysize/2 -imsize/2
        im = transforms.functional.crop(im,crop_y,crop_x,imsize,imsize)
        
        # transform bbox points into cropped coords
        if y[4] == 1:
            y[0] = y[0] - crop_x
            y[1] = y[1] - crop_y
            y[2] = y[2] - crop_x
            y[3] = y[3] - crop_y
        
        return im,y
    
    def show(self, index):
        #Generates one sample of data
        file_name = self.file_list[index]
        
        im = Image.open(file_name).convert('RGB')
        y = self.labels[index,:]
        
        # transform, normalize and convert to tensor
        im,y = self.scale_crop(im,y,imsize = 224)
        im_array = np.array(im)
        new_im = cv2.rectangle(im_array,(y[0],y[1]),(y[2],y[3]),(20,190,210),2)
        plt.imshow(new_im)


def make_model():
    """
    Loads pretrained torchvision model and redefines fc layer for car regression
    """
    # uses about 1 GiB of GPU memory
    model = models.vgg19(pretrained = True)
    #model = models.resnet50(pretrained = True)
    in_feat_num = model.classifier[3].in_features
    mid_feat_num = int(np.sqrt(in_feat_num))
    out_feat_num = 5
    
    # redefine the last two layers of the classifier for car classification
    model.classifier[3] = nn.Linear(in_feat_num,mid_feat_num)
    model.classifier[6] = nn.Linear(mid_feat_num, out_feat_num)
    
    return model


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


def train_model(model, criterion, optimizer, scheduler, dataloaders,dataset_sizes, num_epochs=5, start_epoch = 0):
    """
    Alternates between a training step and a validation step at each epoch. 
    Validation results are reported but don't impact model weights
    """
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            print("1")
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                print("2")
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    print(3)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                print("4")
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # here we need to define a function that checks the bbox iou with correct 
                # still wrong - must be across whole batch
                pred = outputs.data()
                actual = labels.data()
                correct,_ = score_pred(pred,actual)
                running_corrects += correct
                print("5")
                # verbose update
                count += 1
                if count % 10 == 0:
                    print("on minibatch {}".format(count))
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                del best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        # save checkpoint
        PATH = "checkpoint_{}.pt".format(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def score_pred(pred,actual):
    """
    returns two scoring metrics - classification correctness and bbox iou (for positive examples only)
    pred,actual - 1x5 numpy arrays (minx, miny, maxx, maxy, class) for each
    correct - int {0,1}
    bbox_acc - float in range [0,1]
    """
    
    # get class from regression value
    actual[4] = np.round(actual[4])
    pred[4] = np.round(pred[4])
    
    # for negative examples
    if actual[4] == 0:
        if pred[4] == 0:
            correct = 1
            bbox_acc = 1
        else:
            correct = 0
            bbox_acc = 0
        
    else:
        if pred[4] == 0:
            correct = 0
            bbox_acc = 0
        else:
            # get intersection bounds
            minx = max(actual[0],pred[0])
            miny = max(actual[1],pred[1])
            maxx = min(actual[2],pred[2])
            maxy = min(actual[3],pred[3])
            intersection = (maxx-minx)*(maxy-miny)
            
            a1 = (actual[2]-actual[0]) * (actual[3]-actual[1])
            a2 = (pred[2]-pred[0]) * (pred[3] - pred[1])
            union = a1+a2-intersection
            
            correct = 1
            bbox_acc = intersection/union
            
    return correct,bbox_acc


#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    
    # for repeatability
    random.seed = 0
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    
    # create training params
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 0}
    num_epochs = 3
    
    checkpoint_file = None
    
    # create dataloaders
    try:
        print("Checked dataloaders.")
        trainloader
        testloader
    except NameError:   
        pos_path = "/media/worklab/data_HDD/cv_data/images/data_stanford_cars"
        neg_path = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader"
        train_data = Train_Dataset(pos_path,neg_path)
        test_data = Test_Dataset(pos_path,neg_path)
        trainloader = data.DataLoader(train_data, **params)
        testloader = data.DataLoader(test_data, **params)
        print("Got dataloaders.")
    
    # load model
    try:
        model
        print("Checked model.")
    except NameError:
        
        # define CNN model
        model = make_model()
        model = model.to(device)
        print("Got model.")
        
        # define loss function
        criterion = nn.MSELoss()
        # all parameters are being optimized, not just fc layer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        
        # define start epoch for consistent labeling if checkpoint is reloaded
        start_epoch = 0
    
        # if checkpoint specified, load model and optimizer weights from checkpoint
        if checkpoint_file != None:
            model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)
            print("Checkpoint loaded.")
            
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    
    
    if True:    
    # train model
        print("Beginning training.")
        model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders,datasizes,
                               num_epochs, start_epoch)