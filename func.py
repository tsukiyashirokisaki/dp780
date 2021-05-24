import os
import cv2
import random
import numpy as np
import numpy.ma as ma
import torch
import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi, matmul,sqrt,dot,array,zeros,cos,sin,pi,arccos,trace
from seaborn import heatmap
from matplotlib.colors import ListedColormap
def OR(angle):
    a,b,c=angle
    a=a/180*pi
    b=b/180*pi
    c=c/180*pi
    # a-> 1st(z); b->second(x'); c->third(z'')
    A=array([[cos(a),sin(a),0],[-sin(a),cos(a),0],[0,0,1]])
    B=array([[1,0,0],[0,cos(b),sin(b)],[0,-sin(b),cos(b)]])
    C=array([[cos(c),sin(c),0],[-sin(c),cos(c),0],[0,0,1]])
    return matmul(matmul(C,B),A)
def heatplot(mat,title=0,cbar=False):    
    plt.figure()
    plot=heatmap(mat,xticklabels=0, yticklabels=0,cmap="gray",cbar=cbar,square=1)
    plot=plot.get_figure()
    if title!=0:
        plot.savefig(title, dpi=600)
def preprocess_unit_matrix():
    file=open("unit_matrix.txt")
    unit_matrix=[]
    ite=0
    txt=file.read().split("\n")
    while "" in txt:
        txt.remove("")
    for line in txt:
        ele=line.split(" ")
        while "" in ele:
            ele.remove("")
        #print(ele)
        if ite==0:
            app=np.empty([3,3])
        app[ite]=np.array(ele)
        ite+=1
        if ite==3:
            unit_matrix.append(app)
            ite=0
            #print(app)
            #print("\n")
    unit_matrix=np.array(unit_matrix)
    unit_matrix_inv=[]
    for i in range(len(unit_matrix)):
        unit_matrix_inv.append(np.linalg.inv(unit_matrix[i]))
    unit_matrix_inv=np.array(unit_matrix_inv)
    return unit_matrix,unit_matrix_inv
def misorientation(M1,M2,unit_matrix_inv):
    middle=matmul(M1,np.linalg.inv(M2))
    cal=matmul(middle,unit_matrix_inv)
    cosval=-1
    for i in range(24):
        cosval=max((trace(cal[:3,3*i:3*i+3])-1)/2,cosval)
    return arccos(cosval)/pi*float(180)
def classplot(mat,show=1):
    label_colours = np.random.randint(255,size=(1000,3))
    img=np.array([label_colours[ c % 1000 ] for c in mat.astype("int")]).astype("uint8")
    plt.imshow(img)
    if show:
        plt.show()
    return img
def ipfplot(im,xyz=2,show=1):
	img=im[:,:,3*xyz:3*xyz+3]
	if show:
		plt.imshow(img)
	return img

def neigh(i,j):
    if i>=h or i<0 or j>=w or j<0:
        return None
    ret=[]
    if i==h-1 or i==0:
        if j==w-1 or j==0:
            return None
        else:
            ret.append([i,j+1])
            ret.append([i,j-1])
            if i==0:
                ret.append([1,j])
            else:
                ret.append([i-1,j])
    elif j==w-1 or j==0:
        ret.append([i+1,j])
        ret.append([i-1,j])
        if j==w-1:
            ret.append([i,j-1])
        else:
            ret.append([i,1])
    else:
        ret.append([i,j+1])
        ret.append([i,j-1])
        ret.append([i+1,j])
        ret.append([i-1,j])
    return ret


def ipfread(path):
    im=[0,0,0]
    for ele in os.listdir(path):
        if "IPF X" in ele:
            im[0]=cv2.imread(path+"/"+ele)
        elif  "IPF Y" in ele:
            im[1]=cv2.imread(path+"/"+ele)
        elif  "IPF Z" in ele:
            im[2]=cv2.imread(path+"/"+ele)
    im=np.concatenate(im,axis=2)
    return im

def find_neigh(coord,h,w):
    i,j=coord
    neigh=set()
    if i!=h-1:
        neigh.add((i+1,j))
    if i!=0:
        neigh.add((i-1,j))
    if j!=w-1:
        neigh.add((i,j+1))
    if j!=0:
        neigh.add((i,j-1))
    return neigh

def negsample(corner,pic_size=500,sam_size=50):
    mat=np.zeros([pic_size,pic_size])
    for (i,j) in corner:
        for k in range(max(0,i-sam_size+1),min(pic_size,i+sam_size)):
            for l in range(max(0,j-sam_size+1),min(pic_size,j+sam_size)):
                mat[k,l]=1
    cand = []
    for i in range(pic_size-sam_size+1):
        for j in range(pic_size-sam_size+1):
            if not mat[i][j]:
                cand.append([i,j])
    return random.sample(cand,len(corner))

def properties(header=['ND面硬度(Hv)', 'TD面硬度(Hv)', 'UTS strain(%)', 'UTS stress(MPa)','Total Elongation(%)','Ys(Mpa)', 'C2', 'Si1', 'Mn3', 'P1','Cu5', 'Cr', 'Ti4', 'Al7']):
    df=pd.read_excel("data/properties.xlsx",sheet_name="summary")
    names=df["Title"].values
    data=df[header].values
    dic=dict()
    for i in range(data.shape[1]):
        data[:,i]=(data[:,i]-np.mean(data[:,i]))/np.std(data[:,i])
    for i,name in enumerate(names):
        dic[str(name)]=data[i,:].tolist()
    torch.save(dic,"data/properties.pkl")
    return dic
def quaternion(angle):
    a,b,c=angle
    a=a/180*pi
    b=b/180*pi
    c=c/180*pi
    # a-> 1st(z); b->second(x'); c->third(z'')
    return np.array([cos(a/2)*cos(b/2)*cos(c/2)-sin(a/2)*cos(b/2)*sin(c/2),
            sin(a/2)*sin(b/2)*sin(c/2)+cos(a/2)*sin(b/2)*cos(c/2),
            -cos(a/2)*sin(b/2)*sin(c/2)+sin(a/2)*sin(b/2)*cos(c/2),
            cos(a/2)*cos(b/2)*sin(c/2)+sin(a/2)*cos(b/2)*cos(c/2)
           ])
