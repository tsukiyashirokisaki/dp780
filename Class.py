import sys
import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, matmul,sqrt,dot,array,zeros,cos,sin,pi,arccos
from func import OR,heatplot,misorientation
import pandas as pd
import torch.nn as nn

class Data:
    def __init__(self,path=0,crop=False):
        self.inv=np.load("unit_matrix_inv.npy")
        if crop:
            d0,h0,w0,h,w=crop
            self.w,self.h=w,h
            self.data=dict()
            for ele in ["Orient","Phase","MAD","BC","BS","Bands","Error"]:
                self.data[ele]=d0.data[ele][h0:h0+h,w0:w0+w]
            return 
        for ele in os.listdir(path):
            if ".ctf" in ele:
                fn=path+ele
                break
        df=pd.read_csv(fn,skiprows=14,sep="\t")
        dic=dict()
        for col in df.columns:
            dic[col]=col.replace(" ","")
        df=df.rename(columns=dic).dropna(axis=1)
        if "" in df.columns:
            df=df.drop([""],axis=1)
        file=open(fn,"r")
        txt=file.read().split("\n")
        w,h=int(txt[4].split("\t")[1]),int(txt[5].split("\t")[1])
        orient=np.zeros([h,w,3,3])
        data=df[['Euler1', 'Euler2', 'Euler3']].values.reshape(h,w,3)
        for i in range(h):
            for j in range(w):
                orient[i,j]=OR(data[i,j,:3])
        self.w,self.h=w,h
        self.data=dict()
        self.data["Orient"]=orient
        for ele in ["Phase","MAD","BC","BS","Bands","Error"]:
            self.data[ele]=self.set(df,ele)
        self.data["Euler"]=data
    def set(self,df,attr):
        return df[attr].values.reshape(self.h,self.w)
    def get(self,attr):
        return self.data[attr]
    def attr(self):
        return self.data.keys()
class Cluster:
    def __init__(self,index,h,w):
        self.pixels={(h,w)}
        self.neigh=set()
        self.num=1
        self.index=index
    def add(self,h,w):
        self.pixels.add((h,w))
        self.num+=1
    def addn(self,n):
        self.neigh.add(n)
    def removen(self,n):
        self.neigh.remove(n)
    def merge(self,ss,id2clus):
        self.pixels=self.pixels.union(id2clus[ss].pixels)
        self.num+=id2clus[ss].num
        id2clus[ss].neigh.remove(self.index)
        self.neigh.remove(id2clus[ss].index)
        for ele in id2clus[ss].neigh:
            id2clus[ele].addn(self.index)
            id2clus[ele].removen(id2clus[ss].index)        
        self.neigh=self.neigh.union(id2clus[ss].neigh)
        del id2clus[ss]
class Dataset(torch.utils.data.Dataset):
    def __init__(self,bef,target,source):
        self.bef=bef
        self.source=source
        self.target=target
    def __getitem__(self,index):
        X=self.bef[index]
        Y=self.target[index]
        source=self.source[index]
        return torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.long),source
    def __len__(self):
        return len(self.bef)
class CNN50a(nn.Module):
    def __init__(self,in_channel):
        super(CNN50a, self).__init__()
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
class CNN50b(nn.Module):
    def __init__(self,in_channel):
        super(CNN50b, self).__init__()
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
class CNN49a(nn.Module):
    def __init__(self,in_channel):
        super(CNN49a, self).__init__()
        self.batchnorm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU() # activation
        self.maxpool = nn.MaxPool2d(kernel_size=2) 
        self.cnn1 = nn.Conv2d(in_channels=in_channel, out_channels=12, kernel_size=2, stride=1, padding=0) 
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=3, stride=1, padding=0) 
        self.cnn3 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=3, stride=1, padding=0) 
        self.fc1 = nn.Linear(24 * 9 * 9, 2) 
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        # Convolution 1 49
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
class CNN49b(nn.Module):
    def __init__(self,in_channel):
        super(CNN49b, self).__init__()
        self.batchnorm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU() # activation
        self.maxpool = nn.MaxPool2d(kernel_size=2) 
        self.cnn1 = nn.Conv2d(in_channels=in_channel, out_channels=12, kernel_size=4, stride=1, padding=0) 
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=4, stride=1, padding=0) 
        self.cnn3 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=3, stride=1, padding=0) 
        self.fc1 = nn.Linear(24 * 4 * 4, 2) 
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        # Convolution 1 49
        out = self.batchnorm(x)
        out = self.cnn1(x) # 46
        out = self.relu(out)
        out = self.maxpool(out) #23
        out = self.cnn2(out) #20
        out = self.relu(out) 
        out = self.maxpool(out) #10
        out = self.cnn3(out) #8
        out = self.relu(out)
        out = self.maxpool(out) #4
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.softmax(out)
        return out