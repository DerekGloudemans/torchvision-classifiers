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
    def __init__(self, positives_path,negatives_path, max_scaling = 1.5):

        self.max_scaling = max_scaling
        
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
        # pad to 5 times image size because bbox may fall outside of visible coordinates
        y[0] = (y[0]+im.size[0]*2)/(im.size[0]*4)
        y[1] = (y[1]+im.size[1]*2)/(im.size[1]*4)
        y[2] = (y[2]+im.size[0]*2)/(im.size[0]*4)
        y[3] = (y[3]+im.size[1]*2)/(im.size[1]*4)
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
        scale = min(self.max_scaling,max(random.gauss(0.5,1),imsize/min(im.size))) # verfify that scale will at least accomodate crop size
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
        y = y.astype(int)
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
        y[0] = (y[0]+im.size[0]*2)/(im.size[0]*4)
        y[1] = (y[1]+im.size[1]*2)/(im.size[1]*4)
        y[2] = (y[2]+im.size[0]*2)/(im.size[0]*4)
        y[3] = (y[3]+im.size[1]*2)/(im.size[1]*4)
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
        
        return im,y.astype(float)
    
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


class SplitNet(nn.Module):
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SplitNet, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        self.vgg = models.vgg19(pretrained=True)
        del self.vgg.classifier[2:]

        # get size of some layers
        start_num = self.vgg.classifier[0].out_features
        mid_num = int(np.sqrt(start_num))
        cls_out_num = 2 # car or non-car (for now)
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
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out
   

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
        for phase in ['train', 'val']:
            if phase == 'train':
                if True: #disable for Adam
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
                reg_target = labels[:,:4].to(device)
                cls_target = labels[:,4].long().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    cls_outputs, reg_outputs = model(inputs)
                    
                    # make copy of reg_outputs and zero if target is 0
                    # so that bboxes are only learned for positive examples
                    temp = cls_target.unsqueeze(1)
                    temp2 = torch.cat((temp,temp,temp,temp),1).float()
                    reg_outputs_mod = torch.mul(reg_outputs,temp2)
                    
                    # note that the classification loss is done using class-wise probs rather 
                    # than a single class label?
                    cls_loss = cls_criterion(cls_outputs,cls_target)
                    reg_loss = reg_criterion(reg_outputs_mod,reg_target)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        reg_loss.backward(retain_graph = True)
                        cls_loss.backward()
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
                actual = labels.numpy()
                
                
                correct,bbox_acc = score_pred(cls_pred,reg_pred,actual)
                running_corrects += correct
    
                # verbose update
                count += 1
                if count % 100 == 0:
                    print("on minibatch {} -- correct: {} -- avg bbox iou: {} ".format(count,correct,bbox_acc))
                    
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
        
        if epoch % 3 == 0:
            # save checkpoint
            PATH = "checkpoint_{}.pt".format(epoch)
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


def score_pred(cls_preds,reg_preds,actuals):
    """
    returns two scoring metrics - classification correctness and bbox iou (for positive examples only)
    preds,actuals - Bx5 numpy arrays (minx, miny, maxx, maxy, class) for each
    where B = batch size
    correct - int {0,1}
    bbox_acc - float in range [0,1]
    """
    correct_sum = 0
    bbox_accs = 0
    box_count = 0
    for i, cls_pred in enumerate(cls_preds):
        actual = actuals[i]
        bbox_pred = reg_preds[i,:]
        
        # get class from regression value
        actual[4] = np.round(actual[4])
        
        # for negative examples
        if actual[4] == 0:
            if cls_pred == 0:
                correct = 1
                bbox_acc = 0
            else:
                correct = 0
                bbox_acc = 0
            
        else:
            if cls_pred == 0:
                correct = 0
                bbox_acc = 0
            else:
                correct = 1
                box_count = box_count + 1
                # get intersection bounds
                minx = max(actual[0],bbox_pred[0])
                miny = max(actual[1],bbox_pred[1])
                maxx = min(actual[2],bbox_pred[2])
                maxy = min(actual[3],bbox_pred[3])
                if minx > maxx or miny > maxy:
                    bbox_acc = 0
                else:  
                    intersection = (maxx-minx)*(maxy-miny)
                    a1 = (actual[2]-actual[0]) * (actual[3]-actual[1])
                    a2 = (bbox_pred[2]-bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
                    union = a1+a2-intersection
                    bbox_acc = intersection/union
                        
        bbox_accs = bbox_accs + bbox_acc
        correct_sum = correct_sum + correct
    
    if box_count > 0:
        bbox_accs = bbox_accs/float(box_count) # only consider examples where bbox was predicted for positive
    else:
        bbox_accs = 0
    correct_percent = correct_sum/float(len(cls_preds))
    return correct_percent,bbox_accs


def plot_batch(model,loader):
    batch,labels = next(iter(loader))
    batch = batch.to(device)
    
        
    cls_out, reg_out = model(batch)
    _, cls_out = torch.max(cls_out,1)
     
    batch = batch.data.cpu().numpy()
    bboxes = reg_out.data.cpu().numpy()
    preds = cls_out.data.cpu().numpy()
    actuals = labels.data.cpu().numpy()
    
    # define figure subplot grid
    batch_size = loader.batch_size
    fig, axs = plt.subplots((batch_size+7)//8, 8, constrained_layout=True)
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        im =  batch[i].transpose((1,2,0))
        pred = preds[i]
        bbox = bboxes[i]
        actual = actuals[i]
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = std * im + mean
        im = np.clip(im, 0, 1)
        
        if np.round(pred) == 1:
            label = "pred: car"
        else:
            label = "pred: non-car"
        
        # transform bbox coords back into im pixel coords
        bbox = (bbox* 224*4 - 224*2).astype(int)
        actual = (actual *224*4 - 224*2).astype(int)
        # plot bboxes
        im = cv2.rectangle(im,(actual[0],actual[1]),(actual[2],actual[3]),(0.9,0.2,0.2),2)
        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)

        
        axs[i//8,i%8].imshow(im)
        axs[i//8,i%8].set_title(label)
        axs[i//8,i%8].set_xticks([])
        axs[i//8,i%8].set_yticks([])
        plt.pause(.0001)


#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    
    # for repeatability
    random.seed = 0
    
#    del model, train_data,test_data,trainloader,testloader
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    
    # create training params
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 0}
    num_epochs = 50
    
    checkpoint_file =  'checkpoints/7-11-2019/checkpoint_27.pt'
    
    # create dataloaders
    try:
        trainloader
        testloader
        print("Checked dataloaders.")
        
    except NameError:   
        pos_path = "/media/worklab/data_HDD/cv_data/images/data_stanford_cars"
        neg_path = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader"
        train_data = Train_Dataset(pos_path,neg_path,max_scaling = 0.5)
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
        model = SplitNet()
        model = model.to(device)
        print("Got model.")
        
        # define loss functions
        reg_criterion = nn.MSELoss()
        cls_criterion = nn.CrossEntropyLoss()
        
        # all parameters are being optimized, not just fc layer
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = optim.SGD(model.parameters(), lr=0.001,momentum = 0.9)
        
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # define start epoch for consistent labeling if checkpoint is reloaded
        start_epoch = 27
    
        # if checkpoint specified, load model and optimizer weights from checkpoint
        if checkpoint_file != None:
            model,_,start_epoch = load_model(checkpoint_file, model, optimizer)
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
    #plot_batch(model,testloader)
