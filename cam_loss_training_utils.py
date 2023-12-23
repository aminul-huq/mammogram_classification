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
import copy

best_acc = 0
best_loss = 10000000


def cam_loss_train_model(model,train_loader,criterion1,criterion2,optim,train_scheduler,device,epochs):
    model.train()
    train_loss,cam_loss_per_epoch, total_correct, total = 0,0,0,0
    alpha = 20
    
    for i,(images,labels) in enumerate(tqdm(train_loader)):     
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs,sl_map,ac_map = model(images)
        
        loss = criterion1(outputs, labels)

        # model.eval()
        
        cam_loss = 0
        if epochs > 15:
                new_model = copy.deepcopy(model)
                with GradCAM(new_model,target_layer=['rn']) as extractor:
                    outputs2, _, _ = new_model(images)
                    # print( outputs2.argmax().item(),outputs2.argmax(),predicted2, outputs2)
                    _,predicted = torch.max(outputs2.data,1)
                    l = [x for x in predicted]
                    cams = extractor(l, outputs2)
                    cam = cams.pop()
                    
                    ac_map = ac_map.unsqueeze(dim=1)
                    
                    min_val = ac_map.min(-1)[0].min(-1)[0]
                    max_val = ac_map.max(-1)[0].max(-1)[0]
                    ac_map = (ac_map-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
                    
                    
                    cam = cam.unsqueeze(dim=1)
                    
                    min_val = cam.min(-1)[0].min(-1)[0]
                    max_val = cam.max(-1)[0].max(-1)[0]
                    cam = (cam-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
                    
                
                cam_loss = criterion2(cam,ac_map)/(ac_map.shape[2] * ac_map.shape[3])
                
        
        # loss2 = criterion2(sl_map,ac_map)
        
        # print(loss)
        # print(cam_loss)
        final_loss =  loss  + alpha * cam_loss
        
        final_loss.backward()
        optim.step()
        # train_scheduler.step()
        if epochs > 15:
            cam_loss_per_epoch += cam_loss.item() * images.size(0)
        
        train_loss += final_loss.item() * images.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

    return train_loss/len(train_loader), cam_loss_per_epoch/len(train_loader), total_correct*100/total

def cam_loss_test_model(model,test_loader, criterion, optim,modelname,device,epochs):
    model.eval()
    global best_acc
    test_loss,cam_loss_per_epoch,total_correct, total = 0,0,0,0
    
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


def cam_loss_best_test(model,test_loader,criterion,optim,device,epochs):
    model.eval()
    test_loss,total_correct, total = 0,0,0
    y,y_pred = [],[]
    for i,(images,labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs,_,_ = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            
    acc = total_correct*100/total
    print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))    
    return test_loss/len(test_loader),acc,y,y_pred





