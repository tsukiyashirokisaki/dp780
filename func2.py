import torch
import os
import numpy as np
from Class import Data,Dataset
from func import negsample
def create_dataset(root="data/train/",feature=["Orient","MAD"],use_corner=1,pic_size=500,sam_size=50):
    corner=torch.load("data/corner.pkl")
    bef=[]
    target=[]
    source=[]
    paths=[]
    h=w=sam_size
    for date in sorted(os.listdir(root)):
        path=root+date+"/before/"
        paths.append(path)
        data=Data(path)
        data.data["Orient"]=data.data["Orient"].reshape(data.h,data.w,-1)
        if not use_corner:
            app=[]
            for ele in feature:
                if ele=="Orient":
                    app.append(data.data[ele])
                else:
                    app.append(data.data[ele][:,:,np.newaxis])
            bef.append(np.concatenate(app,axis=2))
            continue

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
    if not use_corner:
        return torch.tensor(bef,dtype=torch.float32),paths
    target=np.array(target)
    return Dataset(bef,target,source)
