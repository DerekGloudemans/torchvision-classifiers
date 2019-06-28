"""
In this file are defined dataset objects for training and validating a model
for classification of images as cars/not cars. This model will be built upon
a pretrained pytorch classifier such as vgg-19 or ResNet (pending tests)

Extensions will then include:
    additional transforms for data augmentation, including occlusion
    further classification into vehicle class (car, truck, minivan, suv, etc.)
    further classification by make and model
    
Again, much of the code was informed by:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    
Datasets used in this classifier include:
    Stanford Cars
        @inproceedings{KrauseStarkDengFei-Fei_3DRR2013,
          title = {3D Object Representations for Fine-Grained Categorization},
          booktitle = {4th International IEEE Workshop on  3D Representation and Recognition (3dRR-13)},
          year = {2013},
          address = {Sydney, Australia},
          author = {Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
        }
    Imagenet - for base model training
       
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

# always good to have
import time
import os
import numpy as np    
import _pickle as pickle
import random
import copy

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
        neg_list =  [negatives_path+'/train/'+f for f in os.listdir(negatives_path+ '/train/')]
        self.file_list = pos_list + neg_list

        # create labels (1 for positive, 0 for negative)
        pos_labels = [1 for i in range(len(pos_list))]
        neg_labels = [0 for i in range(len(neg_list))]
        self.labels = pos_labels + neg_labels
        
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
        X = self.transforms(im)
        y = self.labels[index]
        return X, y

class Test_Dataset(data.Dataset):
    """
    Defines dataset and transforms for testing data. The positive images and 
    negative images are stored in two different directories
    """
    def __init__(self, positives_path,negatives_path):

        # use os module to get a list of positive and negative training examples
        # note that shuffling is essential because examples are in order
        pos_list =  [positives_path+'/test/'+f for f in os.listdir(positives_path+'/test/')]
        neg_list =  [negatives_path+'/test/'+f for f in os.listdir(negatives_path+'/test/')]
        self.file_list = pos_list + neg_list

        # create labels (1 for positive, 0 for negative)
        pos_labels = [1 for i in range(len(pos_list))]
        neg_labels = [0 for i in range(len(neg_list))]
        self.labels = pos_labels + neg_labels
        
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
        file_name = self.file_list[index]
        im = Image.open(file_name).convert('RGB')
        X = self.transforms(im)
        y = self.labels[index]
        return X, y
    

def make_model():
    """
    Loads pretrained torchvision model and redefines fc layer for car classification
    """
    # uses about 1 GiB of GPU memory
    model = models.vgg19(pretrained = True)
    #model = models.resnet50(pretrained = True)
    in_feat_num = model.classifier[3].in_features
    mid_feat_num = int(np.sqrt(in_feat_num))
    out_feat_num = 2
    
    # redefine the last two layers of the classifier for car classification
    model.classifier[3] = nn.Linear(in_feat_num,mid_feat_num)
    model.classifier[6] = nn.Linear(mid_feat_num, out_feat_num)
    
    return model

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

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # verbose update
                count += 1
                if count % 100 == 0:
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
    
def show_output():
    pass


def flatten_image_directory():
    """
    I used this once to combine images from all of the files loaded by the imagenet
    into a single superdirectory
    """
    from shutil import copyfile

    train_directory = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader/train"
    test_directory = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader/test"
    base_directory = "/media/worklab/data_HDD/cv_data/images/imagenet_images"
    sub_dir_list = os.listdir(base_directory)
    for subdir in sub_dir_list:
        file_list = os.listdir(os.path.join(base_directory,subdir))
        count = 0
        for file in file_list:
            if count < 10:
                copyfile(os.path.join(base_directory,subdir,file),os.path.join(train_directory,file))
                count +=1
            else:
                copyfile(os.path.join(base_directory,subdir,file),os.path.join(test_directory,file))
        print("Subdirectory copied.")


#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        print("If multiprocessing context wasn't already set, error")
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    
    # for repeatability
    random.seed = 1
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    
    # create training params
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 6}
    num_epochs = 50
    
    checkpoint_file = "checkpoint_5.pt"
    
    # create dataloaders
    pos_path = "/media/worklab/data_HDD/cv_data/images/data_stanford_cars"
    neg_path = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader"
    train_data = Train_Dataset(pos_path,neg_path)
    test_data = Test_Dataset(pos_path,neg_path)
    trainloader = data.DataLoader(train_data, **params)
    testloader = data.DataLoader(test_data, **params)
    print("Dataloaders created.")
    
    # define CNN model
    model = make_model()
    model = model.to(device)
    print("Model created.")
    
    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    # define start epoch for consistent labeling if checkpoint is reloaded
    start_epoch = 0
    
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    
    # if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)
    
    # train model
    print("Beginning training.")
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders,datasizes,
                           num_epochs, start_epoch)
