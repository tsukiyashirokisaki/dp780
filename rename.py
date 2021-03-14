import os
import torch
path="model/"
for name in os.listdir(path):
  model=torch.load(path+name)
  torch.save(model,"model_2/"+name,_use_new_zipfile_serialization=False)
