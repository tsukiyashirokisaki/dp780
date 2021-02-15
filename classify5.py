#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import pi, matmul,sqrt,dot,array,zeros,cos,sin,pi,arccos
from torch.utils.data import DataLoader
from func import OR,heatplot,misorientation,mat2plot,imgshow,ipfread,negsample
from Class import Data,Cluster,Dataset
import torch.nn.functional as F
import matplotlib.patches as patches


# In[2]:


def create_dataset(root="data/train/",feature=["Orient","MAD"]):
    corner=torch.load("data/corner.pkl")
    bef=[]
    target=[]
    source=[]
    h=w=50
    for date in os.listdir(root):
        path=root+date+"/before/"
        data=Data(path)
        data.data["Orient"]=data.data["Orient"].reshape(data.h,data.w,-1)
        for (i,j) in corner[date]:
            app=[]
            for ele in feature:
                if ele=="Orient":
                    app.append(data.data[ele][i:i+h,j:j+w])
                else:
                    app.append(data.data[ele][i:i+h,j:j+w,np.newaxis])
            bef.append(np.concatenate(app,axis=2))
            source.append([date,(i,j)])
            target.append(0)
        for (i,j) in negsample(corner[date]):
            app=[]
            for ele in feature:
                if ele=="Orient":
                    app.append(data.data[ele][i:i+h,j:j+w])
                else:
                    app.append(data.data[ele][i:i+h,j:j+w,np.newaxis])
            source.append([date,(i,j)])
            bef.append(np.concatenate(app,axis=2))
            target.append(1)
    bef=np.transpose(np.array(bef),(0,3,1,2))
    target=np.array(target)
    return Dataset(bef,target,source)
train=create_dataset("data/train/",feature=["Orient","BC","BS"])
test=create_dataset("data/test/",feature=["Orient","BC","BS"])


# In[3]:


class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.batchnorm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU() # activation
        self.maxpool = nn.MaxPool2d(kernel_size=2) 
        self.cnn1 = nn.Conv2d(in_channels=in_channel, out_channels=12, kernel_size=3, stride=1, padding=0) 
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=3, stride=1, padding=1) 
        self.cnn3 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=3, stride=1, padding=0) 
        self.fc1 = nn.Linear(24 * 5 * 5, 2) 
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        # Convolution 1 50
        out = self.batchnorm(x)
        out = self.cnn1(x) # 48
        out = self.relu(out)
        out = self.maxpool(out) #24
        out = self.cnn2(out) #24
        out = self.relu(out) 
        out = self.maxpool(out) #12
        out = self.cnn3(out) #10
        out = self.relu(out)
        out = self.maxpool(out) #5
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.softmax(out)
        return out
class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()
        self.batchnorm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU() # activation
        self.maxpool = nn.MaxPool2d(kernel_size=2) 
        self.cnn1 = nn.Conv2d(in_channels=in_channel, out_channels=12, kernel_size=3, stride=1, padding=0) 
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=3, stride=1, padding=0) 
        self.cnn3 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=3, stride=1, padding=0) 
        self.fc1 = nn.Linear(24 * 9 * 9, 2) 
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        # Convolution 1 50
        out = self.batchnorm(x)
        out = self.cnn1(x) # 48
        out = self.relu(out)
        out = self.maxpool(out) #24
        out = self.cnn2(out) #22
        out = self.relu(out) 
        out = self.maxpool(out) #11
        out = self.cnn3(out) #9
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.softmax(out)
        return out


# In[4]:


train_loader= DataLoader(train, batch_size=64, shuffle=True,  num_workers=0, drop_last=True )
test_loader= DataLoader(test, batch_size=64, shuffle=True,  num_workers=0,  drop_last=False )


# In[10]:


for __ in range(10):
    in_channel=train[0][0].shape[0]
    model = CNN5()
    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    max_acc = 0.7
    epoch = 200
    for ep in range(epoch):
        for batch_ndx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            X,Y,_= sample
            output=model(X)
            loss = Loss(output,Y)
            loss.backward()
            optimizer.step()
    #         print("train ",ep,loss.item())
        model.eval()
        cum=0
        loss=0
        for batch_ndx, sample in enumerate(test_loader):
            X,Y,_= sample
            output=model(X)
            loss+=Loss(output,Y)
            predict = torch.max(output, 1)[1]
            cum+=np.sum((Y == predict).cpu().numpy())
        acc=cum/len(test)
        if acc>max_acc:
            max_acc=acc
            min_loss = loss.item()
            print(ep,min_loss)
            print("acc= ",cum,"/",len(test))
            torch.save(model,"model/%s_obcbs_%.3f.pkl"%(type(model).__name__,acc))
        
    