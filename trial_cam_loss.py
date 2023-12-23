import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, pickle, shutil, random, PIL
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader,random_split,Dataset, ConcatDataset ,SubsetRandomSampler 
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import v2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from training_utils import *
from classification_models import *
from focal_loss_with_smoothing import *

from torchcam.methods import SmoothGradCAMpp, LayerCAM, GradCAM,CAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, resize
from torch.nn.functional import softmax, interpolate

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DFNAME = 'MvNM_rn_34_camLoss'
device = torch.device('cuda:0')
# criterion1 = FocalLossWithSmoothing(num_classes = 2, gamma=0.5, lb_smooth = 0.0)
criterion2 =  nn.L1Loss()
criterion1 = nn.CrossEntropyLoss()

modelname = 'MvNM_rn_34_camLoss'
n_epochs = 100
batch_size = 4


train_dir = '/home/aminul/CVL/cs791_project/mass_vs_non_mass/train/imgs/'
test_dir = '/home/aminul/CVL/cs791_project/mass_vs_non_mass/test/imgs/'
val_dir = '/home/aminul/CVL/cs791_project/mass_vs_non_mass/val/imgs/'

size = (512,384)
train_set_whole = ImageFolder(train_dir,transform = transforms.Compose([
    v2.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.3, hue=0.1),
    v2.RandomChannelPermutation(),
    v2.RandomAdjustSharpness(2,0.5),
    v2.RandomAutocontrast(0.5),
    v2.Resize(size),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomRotation(30),
    v2.ToTensor(),
    # v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]))

val_set = ImageFolder(val_dir,transform = transforms.Compose([
    v2.Resize(size),
    v2.ToTensor(),
    # v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]))

test_set = ImageFolder(test_dir,transform = transforms.Compose([
    v2.Resize(size),
    v2.ToTensor(),
    # v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]))


# train_set, valid_set = random_split(train_set_whole,[int(len(train_set_whole)*0.9), int(len(train_set_whole)*0.1)],
#                                   generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_set_whole, batch_size=batch_size, shuffle=True, num_workers = 4)
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers = 4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 4)
test_loader_2 = DataLoader(test_set, batch_size=1, shuffle=False)

model = ResNet34().to(device)

optim = torch.optim.Adam(model.parameters(),lr=0.05, weight_decay=1e-4)

def cam_loss_train(model, train_loader, criterion1, criterion2, optim, device, epoch):
    model.train()
    train_loss, total_correct, total = 0,0,0
    # cam_extractor1 = GradCAM(model,target_layer=['layer4'])
    
    for i,(images,labels) in enumerate(tqdm(train_loader)):     
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()

        outputs, ac_map = model(images)
        loss = criterion1(outputs, labels)

        # model.eval()
        
        with LayerCAM(model,target_layer=['layer4']) as extractor:
            # outputs2, _ = model(images)
            # print( outputs2.argmax().item(),outputs2.argmax(),predicted2, outputs2)
            _,predicted = torch.max(outputs2.data,1)
            l = [x for x in predicted]
            cams = extractor(l, outputs2)
            cam = cams.pop()
            min_val = ac_map.min(-1)[0].min(-1)[0]
            max_val = ac_map.max(-1)[0].max(-1)[0]
            ac_map = (ac_map-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
            cam_loss = criterion2(cam,ac_map)
        
        # model.train()
        final_loss = loss + 0.5*cam_loss
        
        final_loss.backward()
            
        optim.step()
        # train_scheduler.step()
        
        train_loss += loss.item() * images.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epoch+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))
    
    return train_loss/len(train_loader), total_correct*100/total


def cam_loss_test(model,test_loader,criterion,optim,modelname,device,epochs):
    model.eval()
    global best_acc
    test_loss,total_correct, total = 0,0,0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = total_correct*100/total
        print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))  

        if acc > best_acc:
            print('Saving Best model...')
            state = {
                        'model':model.state_dict(),
                        'acc':acc,
                        'epoch':epochs,
                }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point+modelname+'model.pth.tar')
            best_acc = acc

    return test_loss/len(test_loader),acc

def cam_best_test(model,test_loader,criterion,optim,device,epoch):
    model.eval()
    global best_acc
    test_loss,total_correct, total = 0,0,0
    y,y_pred = [],[]
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            y.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

        acc = total_correct*100/total
        print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epoch+1,test_loss/len(test_loader),acc))  

    return test_loss/len(test_loader),acc,y,y_pred



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)

history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

for epoch in range(n_epochs):
    
    train_loss, train_acc = cam_loss_train(model, train_loader, criterion1, criterion2, optim, device, epoch)
    # train_loss, train_acc = train_model(model,train_loader,criterion,optim,None,device,epoch)
    # cam_extractor1._hooks_enabled = False
    valid_loss, valid_acc = cam_loss_test(model,valid_loader,criterion1,optim,modelname,device,epoch)
    
    scheduler.step(valid_loss)
    
    # test(model,test_loader,criterion,optim,filename,modelname,device,epoch)
            
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)
    history['train_acc'].append(train_acc)
    history['valid_acc'].append(valid_acc)

    
with open('./storage/' + DFNAME + '.pkl', 'wb') as f:
    pickle.dump(history, f) 








































