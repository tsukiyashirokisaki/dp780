import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from numpy import pi, matmul,sqrt,dot,array,zeros,cos,sin,pi,arccos,trace
from seaborn import heatmap
from matplotlib.colors import ListedColormap
import os
import cv2
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
def mat2plot(mat):
    label_colours = np.random.randint(255,size=(100,3))
    img=np.array([label_colours[ c % 100 ] for c in mat.astype("int")]).astype("uint8")
    plt.imshow(img)
    plt.show()
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
def calpoint(p1,h0,w0,img2):
    h,w,_=img2.shape
    min_l=2e9
    t=50
    for k in range(max(h0-t,0),min(h0+t,h)):
        for l in range(max(w0-t,0),min(w0+t,w)):
            loss=L2(p1,img2[k,l])
            if loss<min_l:
                min_l=loss
                ind=[k,l]
    return ind
def imgshow(im):    
    plt.imshow(im[:,:,6:])
    plt.show()
def match(func,befarr,aftarr,h0,w0):
    h=befarr.shape[0]
    w=befarr.shape[1]
    min_val=1e9
    t=150
    for i in range(max(0,h0-t),min(h0+t,aftarr.shape[0]-befarr.shape[0])):        
        for j in range(max(0,w0-t),min(w0+t,aftarr.shape[1]-befarr.shape[1])):
            val=np.sum(func(aftarr[i:i+h,j:j+w],befarr))
#             print(val)    
            if val<min_val:
                min_val=val
                param=[i,j]
    return param,min_val
#     for i in range(aftarr.shape[0]-befarr.shape[0]):        
#         for j in range(aftarr.shape[1]-befarr.shape[1]):
#             val=np.sum(func(aftarr[i:i+h,j:j+w],befarr))
# #             print(val)    
#             if val<min_val:
#                 min_val=val
#                 param=[i,j]
#     return param,min_val
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
def L2(a,b):
    return (a-b)**2
def L1(a,b):
    return abs(a-b)
def f(num):
    return num+1
def calmisorientation(orient,i,j,h,w,inv,t):
    if i==0:
        if j!=w-1:
            if misorientation(orient[i,j],orient[i,j+1],inv)>t:
                return [[i,j+1],[i,j]]
    else:
        if j==0:
            if misorientation(orient[i,j],orient[i-1,j],inv)>t:
                return [[i,j-1],[i,j]]
        else:
            n1=misorientation(orient[i,j],orient[i-1,j],inv)
            n2=misorientation(orient[i,j],orient[i,j-1],inv)
            ret=[]
            if n1>t:
                ret.append([i,j])
                ret.append([i-1,j])
            if n2>t:
                ret.append([i,j])
                ret.append([i,j-1])
            return ret
