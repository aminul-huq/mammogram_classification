import torch,os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os, cv2
from os import path
import copy
import torch.nn.functional as F
from torchcam.methods import SmoothGradCAMpp, LayerCAM, GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, resize
from torch.nn.functional import softmax, interpolate

best_acc = 0
best_loss = 10000000


def train_model(model,train_loader,criterion,optim,train_scheduler,device,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(train_loader)):     
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs,sl_map,ac_map = model(images)
        
        c2 = nn.L1Loss()
        
        loss2 = c2(sl_map,ac_map)
        
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
            
        final_loss = loss #+ 0.5*loss2
        
        final_loss.backward()
        optim.step()
        # train_scheduler.step()
        
        train_loss += final_loss.item() * images.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

    return train_loss/len(train_loader), total_correct*100/total

def test_model(model,test_loader, criterion, optim,modelname,device,epochs):
    model.eval()
    global best_acc
    test_loss,total_correct, total = 0,0,0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs,_,_ = model(images)
            
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


def best_test(model,test_loader,criterion,optim,device,epochs):
    model.eval()
    test_loss,total_correct, total = 0,0,0
    y,y_pred, y_prob, y_true = [],[],[],[]
    
    
    for i,(images,labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs,_,_ = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            
            prob = F.softmax(outputs, dim =1)
            top_prob, top_class = prob.topk(1, dim=1)
            
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            
            y_prob.append(top_prob.cpu())
            y_true.append(labels.cpu())
            
    acc = total_correct*100/total
    print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))    
    return test_loss/len(test_loader),acc,y ,y_pred #, y_prob, y_true





