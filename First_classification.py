#################################################################################################################################################################################

##### Copyright C 2024, Arthur Sarron. All Right Reserved.
##### 2024 is the year in which the work was first published
##### Arthur Sarron is the individual or institution that owns the copyright

#################################################################################################################################################################################

##### Importation of librarires ######

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

from difflib import SequenceMatcher

##### Importing Data, conversion and reading ######
MyData=pd.read_csv('E:\\Postdoc\\Formation_libre\\Fiddles_IA_2024\\Données\\breast-cancer.csv')
MyData.pop('id')


Feats=list(MyData.keys())
#print(Feats)
print(MyData.shape)
print(MyData.isnull().sum().sum())

####### Research of similarity intercolumn #### to develop
def similar(a,b):
    threshold = 0.8
    return (SequenceMatcher(None, a, b).ratio()>threshold)
print(similar('radius_mean','area_mean'))
##### Conversion Data to numpy #####

MyData_numpy=MyData.to_numpy()
MyData_numpy_md=MyData_numpy
#print (MyData)

##### Conversion to tensorflow

##MyData_pytorch=torch.tensor(MyData_numpy,dtype=torch.float64)