import os
import torch
path="model/CNN_49g_780_980/"
for name in os.listdir(path):
  print(path+name)
  model=torch.load(path+name)
  torch.save(model,"model_2/"+name,_use_new_zipfile_serialization=False)
