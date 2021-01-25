import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from numpy import pi, matmul,sqrt,dot,array,zeros,cos,sin,pi,arccos
from func import OR,heatplot,misorientation,mat2plot, match
import sys
import os
import cv2
import pandas as pd
class Exp:
    def __init__(self,bef,aft):
        self.data=[bef,aft]
    def shift(self,arr):
        return (arr-0.5)*2.
    def shiftb(self,arr):
        return arr/2.+0.5
    def match(self,attr,func):
        bef=self.data[0];aft=self.data[1]
        befarr=self.shift(bef.get(attr))
        aftarr=self.shift(aft.get(attr))
        h=bef.h
        w=bef.w
        max_val=0
        for i in range(aft.h-bef.h):        
            for j in range(aft.w-bef.w):
                val=np.sum(func(aftarr[i:i+h,j:j+w],befarr))
                if val>max_val:
                    max_val=val
                    param=[i,j]
        self.i=param[0]
        self.j=param[1]
    def getmatch(self,attr,aft=1):
        return self.mod(self.data[aft].get(attr),aft)
    def mod(self,arr,aft=1):
        if aft!=1:
            return arr
        else:
            return arr[self.i:self.i+self.data[0].h,self.j:self.j+self.data[0].w]
class Data:
    def __init__(self,path=0,crop=False):
        self.inv=np.load("unit_matrix_inv.npy")
        if crop:
            d0,h0,w0,h,w=crop
            self.w,self.h=w,h
            self.data=dict()
            for ele in ["orient","Phase","MAD","BC","BS","Bands","Error"]:
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
        self.data["orient"]=orient
        for ele in ["Phase","MAD","BC","BS","Bands","Error"]:
            self.data[ele]=self.set(df,ele)


    def set(self,df,attr):
        return df[attr].values.reshape(self.h,self.w)
    def get(self,attr):
        return self.data[attr]
    

    def attr(self):
        return self.data.keys()
        
    def bclassmap(self,t=1):
        h=self.h;w=self.w;bc=self.get("BC");inv=self.inv;se=None;orient=self.orient
        id2clus=dict()
        class_map=np.zeros([h,w]).astype("int") # c-> class #
        class_map[0,0]=c=1
        cluster=Cluster(c,0,0,self.orient,None,self.get("BC"))
        id2clus[class_map[0,0]]=cluster
        for i in range(h):#h
            for j in range(w):#w
                if i==0 :
                    if j!=w-1:
                        if misorientation(orient[i,j],orient[i,j+1],inv)<t:
                            class_map[i,j+1]=class_map[i,j]
                            id2clus[class_map[i,j+1]].add(i,j+1,orient,se,bc)
                        else:
                            c+=1
                            class_map[i,j+1]=c
                            cluster=Cluster(c,i,j+1,orient,se,bc)
                            id2clus[class_map[i,j+1]]=cluster
                            id2clus[class_map[i,j+1]].addn(class_map[i,j])
                            id2clus[class_map[i,j]].addn(class_map[i,j+1])
                else:
                    if j==0:
                        if misorientation(orient[i,j],orient[i-1,j],inv)<t:
                            class_map[i,j]=class_map[i-1,j]
                            id2clus[class_map[i,j]].add(i,j,orient,se,bc)
                        else:
                            c+=1
                            class_map[i,j]=c
                            cluster=Cluster(c,i,j,orient,se,bc)
                            id2clus[class_map[i,j]]=cluster
                            id2clus[class_map[i-1,j]].addn(class_map[i,j])
                            id2clus[class_map[i,j]].addn(class_map[i-1,j])
                    else:
                        n1=misorientation(orient[i,j],orient[i-1,j],inv)
                        n2=misorientation(orient[i,j],orient[i,j-1],inv)
                        if n1<n2 and n1<t:
                            class_map[i,j]=class_map[i-1,j]
                            id2clus[class_map[i,j]].add(i,j,orient,se,bc)
                            if class_map[i,j]!=class_map[i,j-1]:
                                id2clus[class_map[i,j]].addn(class_map[i,j-1])
                                id2clus[class_map[i,j-1]].addn(class_map[i,j])
                        elif n2<t:
                            class_map[i,j]=class_map[i,j-1]
                            id2clus[class_map[i,j]].add(i,j,orient,se,bc)
                            if class_map[i,j]!=class_map[i-1,j]:
                                id2clus[class_map[i,j]].addn(class_map[i-1,j])
                                id2clus[class_map[i-1,j]].addn(class_map[i,j])
                        else:
                            c+=1
                            class_map[i,j]=c
                            cluster=Cluster(c,i,j,orient,se,bc)
                            id2clus[class_map[i,j]]=cluster
                            if class_map[i,j]!=class_map[i-1,j]:
                                id2clus[class_map[i,j]].addn(class_map[i-1,j])
                                id2clus[class_map[i-1,j]].addn(class_map[i,j])
                            if class_map[i,j]!=class_map[i,j-1]:
                                id2clus[class_map[i,j]].addn(class_map[i,j-1])
                                id2clus[class_map[i,j-1]].addn(class_map[i,j])
        self.id2clus=id2clus
        self.class_map=class_map

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