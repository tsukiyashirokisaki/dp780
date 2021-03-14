import torch
import os
import numpy as np
from Class import Data,Dataset
from func import negsample,quaternion
def create_dataset(subset="train",steels=["590","780","980"],feature=["Orient","MAD"],prop_dic={},pic_size=500,sam_size=50,prediction=0):
    bef=[]
    target=[]
    source=[]
    props=[]
    h=w=sam_size
    for steel in steels:
        corner=torch.load("data/%s_corner.pkl"%(steel))
        root="data/%s/%s/"%(steel,subset)
        for date in sorted(os.listdir(root)):
            path="%s%s/before/"%(root,date)
            print(path)
            data=Data(path)
            data.data["Orient"]=data.data["Orient"].reshape(data.h,data.w,-1)
            data.data["Quaternion"]=data.data["Quaternion"].reshape(data.h,data.w,-1)
            if prediction:
                app=[]
                for ele in feature:
                    if ele=="Orient" or ele=="Quaternion":
                        app.append(data.data[ele][:pic_size,:pic_size])
                    else:   
                        app.append(data.data[ele][:pic_size,:pic_size,np.newaxis])
                bef.append(np.concatenate(app,axis=2))
                source.append(path)
                props.append(prop_dic[steel])
                    
                    
            else:
                for (i,j) in corner[date]:
                    app=[]
                    for ele in feature:
                        if ele=="Orient" or ele=="Quaternion":
                            app.append(data.data[ele][i:i+h,j:j+w])
                        else:   
                            app.append(data.data[ele][i:i+h,j:j+w,np.newaxis])
                    bef.append(np.concatenate(app,axis=2))
                    props.append(prop_dic[steel])
                    source.append([path,(i,j)])
                    target.append(0)
                for (i,j) in negsample(corner[date],pic_size,sam_size):
                    app=[]
                    for ele in feature:
                        if ele=="Orient" or ele=="Quaternion":
                            app.append(data.data[ele][i:i+h,j:j+w])
                        else:
                            app.append(data.data[ele][i:i+h,j:j+w,np.newaxis])
                    source.append([date,(i,j)])
                    bef.append(np.concatenate(app,axis=2))
                    props.append(prop_dic[steel])
                    target.append(1)
    bef=np.transpose(np.array(bef),(0,3,1,2))
    props=np.array(props)
    target=np.array(target)
    if prediction:
        return  torch.tensor(bef,dtype=torch.float32),torch.tensor(props,dtype=torch.float32),source
    return Dataset(bef,props,target,source)