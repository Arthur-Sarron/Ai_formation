#################################################################################################################################################################################

##### Copyright C 2024, Arthur Sarron. All Right Reserved.
##### 2024 is the year in which the work was first published
##### Arthur Sarron is the individual or institution that owns the copyright

#################################################################################################################################################################################

##### Importation of librarires ######

import numpy as np
import matplotlib.pyplot as plt
import pandas

import torch

##### Importing Data, conversion and reading ######

MyData=pandas.read_csv('E:\Postdoc\Formation_libre\Fiddles_IA_2024\Données\S5_mathematique_notation.csv')
###print(MyData)
###data_head=list(MyData.columns) 
###print(data_head)
MyData.pop('Eleve') #### Remember the default separatoir is a comma not semi columns the Student will be an "header"

Matieres=list(MyData.keys()) #### print the field
print(Matieres)

Notes=MyData.to_numpy() ##### conversion of the mark in numpy format and print them
print(Notes)

Notes_pt=torch.tensor(Notes,dtype=torch.float64) ### conversion the Notes to pytorch tensor format using a double-precision floating-point format occupying a6' bits in computer mem

###### Definitions of the acivation function and density layer (including its parameters) ####

act_func_relu=torch.nn.ReLU() #### Definition of the activation ReLU (0;max)

lin_filter=torch.nn.Linear(7,2, bias=True) #### Definition of the dense layer and its parameter 7 is the input demension to 2 output dimension

lin_filter.weight.data=torch.tensor([[0.2,0.2,0.2,0,0,0.2,0.2],[0.0,0.0,0.0,0.5,0.5,0.0,0.0]],dtype=torch.float64) ### defintion of the weights in 2 vectors (here will be science in the first and LV for the second dimensions), type 

lin_filter.bias.data=torch.tensor([-10,-10],dtype=torch.float64) ### Defintion of the Bias for the 2 dimension

#### Data processing ####

filtered_data=act_func_relu(lin_filter(Notes_pt[:,:])) #### Data processing Notes_pt will be processing in the first filter (lin_filter) and its results will be used in the activation function(act_func_relu)

#### Printing Results ####
print(filtered_data)