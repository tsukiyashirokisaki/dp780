#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import pi, matmul,sqrt,dot,array,zeros,cos,sin,pi,arccos
from torch.utils.data import DataLoader
from func import OR,heatplot,misorientation,ipfread,negsample
from Class import Data,Cluster,Dataset,CNN50a,CNN50b,CNN49a,CNN49b
import torch.nn.functional as F
import matplotlib.patches as patches
feature=sys.argv[1].split("_")

# In[2]:

pic_size=500
sam_size=49
def create_dataset(root="data/train/",feature=["Orient","MAD"],pic_size=500,sam_size=50):
    corner=torch.load("data/corner.pkl")
    bef=[]
    target=[]
    source=[]
    h=w=sam_size
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
        for (i,j) in negsample(corner[date],pic_size,sam_size):
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
train=create_dataset("data/train/",feature=feature,pic_size=pic_size,sam_size=sam_size)
test=create_dataset("data/test/",feature=feature,pic_size=pic_size,sam_size=sam_size)


train_loader= DataLoader(train, batch_size=64, shuffle=True,  num_workers=0, drop_last=True )
test_loader= DataLoader(test, batch_size=64, shuffle=True,  num_workers=0,  drop_last=False )


# In[10]:

for modeltype in [CNN49a,CNN49b]:
    in_channel=train[0][0].shape[0]
    model=modeltype(in_channel)
    for __ in range(3):
        
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
                torch.save(model,"model/%.3f_%s_%s.pkl"%(acc,type(model).__name__,"_".join(feature)))
            
        