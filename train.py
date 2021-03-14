#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from func import OR,heatplot,misorientation,ipfread,negsample,properties
from func2 import create_dataset
from Class import Data,Cluster,Dataset,CNN49ag,CNN49bg
import torch.nn.functional as F
import matplotlib.patches as patches
feature=sys.argv[1].split("_")
pic_size=500
sam_size=49
# prop_dic=properties(header)
prop_dic=torch.load("data/properties.pkl")
train=create_dataset("train",feature=feature,prop_dic=prop_dic,pic_size=pic_size,sam_size=sam_size)
val=create_dataset("val",feature=feature,prop_dic=prop_dic,pic_size=pic_size,sam_size=sam_size)

comp=len(train[0][1])
in_channel=train[0][0].shape[0]
train_loader= DataLoader(train, batch_size=64, shuffle=True,  num_workers=0, drop_last=True )
val_loader= DataLoader(val, batch_size=64, shuffle=True,  num_workers=0,  drop_last=False )
Loss = nn.CrossEntropyLoss()

for modeltype in [CNN49ag,CNN49bg]:
    for __ in range(3):
    	model=modeltype(in_channel,comp)
    	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    	max_acc = 0.7
    	epoch = 200
    	for ep in range(epoch):
    		for batch_ndx, sample in enumerate(train_loader):
	    		optimizer.zero_grad()
	    		X,P,Y,_= sample
	    		output=model(X,P)
	    		loss = Loss(output,Y)
	    		loss.backward()
	    		optimizer.step()
			
    		model.eval()
    		cum=0
    		loss=0
    		for batch_ndx, sample in enumerate(val_loader):
    			X,P,Y,_= sample
    			output=model(X,P)
    			loss+=Loss(output,Y)
    			predict = torch.max(output, 1)[1]
    			cum+=np.sum((Y == predict).cpu().numpy())
    		acc=cum/len(val)
    		if acc>max_acc:
    			max_acc=acc
    			min_loss = loss.item()
    			print(ep,min_loss)
    			print("acc= ",cum,"/",len(val))
    			torch.save(model,"model/%.3f_%s_%s.pkl"%(acc,type(model).__name__,"_".join(feature)))



