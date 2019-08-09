# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:21:35 2017

@author: Elham
"""

import pickle
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

pyplot.close("all")


# %% Image Reading

PatchFile_benign = "//192.168.106.134/Projects/BACH_challenge/Pickle Files [224x244x3]/i_benign.pickle"
PatchFile_insitu = "//192.168.106.134/Projects/BACH_challenge/Pickle Files [224x244x3]/i_insitu.pickle"
PatchFile_invasive = "//192.168.106.134/Projects/BACH_challenge/Pickle Files [224x244x3]/i_invasive.pickle"
PatchFile_normal = "//192.168.106.134/Projects/BACH_challenge/Pickle Files [224x244x3]/i_normal.pickle"

ResultsPath = "//192.168.106.134/Projects/BACH_challenge/TrainValidateTestSamples_Pickle Files/"

with open ( PatchFile_benign, 'rb' ) as PKFile:
     PatchFile_benign = pickle.load(PKFile) 
     
with open ( PatchFile_insitu, 'rb' ) as PKFile:
     PatchFile_insitu = pickle.load(PKFile) 

with open ( PatchFile_invasive, 'rb' ) as PKFile:
     PatchFile_invasive = pickle.load(PKFile) 
     
with open ( PatchFile_normal, 'rb' ) as PKFile:
     PatchFile_normal = pickle.load(PKFile)      


benign_train, benign_test = train_test_split(PatchFile_benign, test_size=0.2, train_size=0.8, random_state = 10)
benign_train, benign_validate = train_test_split(benign_train, test_size=0.125, train_size=0.875, random_state = 10)


insitu_train, insitu_test = train_test_split(PatchFile_insitu, test_size=0.2, train_size=0.8, random_state = 10)
insitu_train, insitu_validate = train_test_split(insitu_train, test_size=0.125, train_size=0.875, random_state = 10)


invasive_train, invasive_test = train_test_split(PatchFile_invasive, test_size=0.2, train_size=0.8, random_state = 10)
invasive_train, invasive_validate = train_test_split(invasive_train, test_size=0.125, train_size=0.875, random_state = 10)


normal_train, normal_test = train_test_split(PatchFile_normal, test_size=0.2, train_size=0.8, random_state = 10)
normal_train, normal_validate = train_test_split(normal_train, test_size=0.125, train_size=0.875, random_state = 10)


# %%     
with open(ResultsPath + 'benign_train' + '.pkl', 'wb') as PKFile:
     pickle.dump(benign_train, PKFile)
     
with open(ResultsPath + 'benign_test' + '.pkl', 'wb') as PKFile:
     pickle.dump(benign_test, PKFile)
     
with open(ResultsPath + 'benign_validate' + '.pkl', 'wb') as PKFile:
     pickle.dump(benign_validate, PKFile)  
     

# %%     
with open(ResultsPath + 'insitu_train' + '.pkl', 'wb') as PKFile:
     pickle.dump(insitu_train, PKFile)
     
with open(ResultsPath + 'insitu_test' + '.pkl', 'wb') as PKFile:
     pickle.dump(insitu_test, PKFile)
     
with open(ResultsPath + 'insitu_validate' + '.pkl', 'wb') as PKFile:
     pickle.dump(insitu_validate, PKFile)  
     
 
# %%    
with open(ResultsPath + 'invasive_train' + '.pkl', 'wb') as PKFile:
     pickle.dump(invasive_train, PKFile)
     
with open(ResultsPath + 'invasive_test' + '.pkl', 'wb') as PKFile:
     pickle.dump(invasive_test, PKFile)
     
with open(ResultsPath + 'invasive_validate' + '.pkl', 'wb') as PKFile:
     pickle.dump(invasive_validate, PKFile)  
     

# %%
with open(ResultsPath + 'normal_train' + '.pkl', 'wb') as PKFile:
     pickle.dump(normal_train, PKFile)
     
with open(ResultsPath + 'normal_test' + '.pkl', 'wb') as PKFile:
     pickle.dump(normal_test, PKFile)
     
with open(ResultsPath + 'normal_validate' + '.pkl', 'wb') as PKFile:
     pickle.dump(normal_validate, PKFile)       