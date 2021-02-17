import torch
import os
import numpy as np
from Class import Data,Dataset
from func import negsample
def create_dataset(root="data/train/",feature=["Orient","MAD"],use_corner=1):
    corner=torch.load("data/corner.pkl")
    bef=[]
    target=[]
    source=[]
    h=w=50
    for date in os.listdir(root):
        path=root+date+"/before/"
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
    if not use_corner:
        return torch.tensor(bef,dtype=torch.float32)
    target=np.array(target)
    return Dataset(bef,target,source)
