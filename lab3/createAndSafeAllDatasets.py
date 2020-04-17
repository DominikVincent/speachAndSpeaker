import warnings
warnings.simplefilter("ignore")
import time
import random
import sys
import importlib
import editdistance
import numpy as np
from prondict import *
from lab3_proto import *
from lab3_tools import *
from lab1_proto import *
from lab1_tools import *
from lab2_proto import *
from lab2_tools import *
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD
from keras.callbacks.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15, 4)
np.set_printoptions(precision=3)

def accousticContextLmfcc(dset):    
    for j in range(len(dset)):
        length = dset[j]['lmfcc'].shape[0]
        allFeatures = np.zeros((0,91))
        for i in range(len(dset[j]['lmfcc'])):
            currentfeatures = []
            if i>=3 and i < length - 3:
                currentfeatures = dset[j]['lmfcc'][i-3:i+4]
                currentfeatures = currentfeatures.flatten()
            elif i<=3:
                first = dset[j]['lmfcc'][:4-i]
                first = np.flip(first)

                currentfeatures = np.append(first, dset[j]['lmfcc'][1:i+4])
            else:
                second = dset[j]['lmfcc'][-(3-(length-i-1))-1:-1]
                second = np.flip(second)

                currentfeatures = np.append(dset[j]['lmfcc'][(i-3):], second)

            #print(currentfeatures.shape)
            allFeatures = np.vstack((allFeatures, currentfeatures))
        
        #print(dset[j]['lmfcc'].shape," - ", allFeatures.shape)
        dset[j]['lmfcc'] = allFeatures
    return dset

def accousticContextMspec(dset):  
    for j in range(len(dset)):
        length = dset[j]['mspec'].shape[0]
        allFeatures = np.zeros((0,280))
        for i in range(len(dset[j]['mspec'])):
            currentfeatures = []
            if i>=3 and i < length - 3:
                currentfeatures = dset[j]['mspec'][i-3:i+4]
                currentfeatures = currentfeatures.flatten()
            elif i<=3:
                first = dset[j]['mspec'][:4-i]
                first = np.flip(first)

                currentfeatures = np.append(first, dset[j]['mspec'][1:i+4])
            else:
                second = dset[j]['mspec'][-(3-(length-i-1))-1:-1]
                second = np.flip(second)

                currentfeatures = np.append(dset[j]['mspec'][(i-3):], second)

            #print(currentfeatures.shape)
            allFeatures = np.vstack((allFeatures, currentfeatures))
        
        #print(dset[j]['lmfcc'].shape," - ", allFeatures.shape)
        dset[j]['mspec'] = allFeatures
    
    print("accousticContextMspec done")
    return dset

def standardizationWholeTypeLmfcc(dset, numberOfStates, scaler=None):
    dFrame = pd.DataFrame(dset.tolist())
    feat = dFrame['lmfcc'].values
    targets = dFrame['targets'].values
    feat    = np.concatenate( feat,    axis=0 )
    targets = np.concatenate( targets, axis=0 )
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(feat)
    featTransformed = scaler.transform(feat)
    return np.array([featTransformed.astype('float32'), np_utils.to_categorical(targets, numberOfStates)[np.newaxis]]), scaler

def standardizationWholeTypeMspec(dset, numberOfStates, scaler=None):
    dFrame = pd.DataFrame(dset.tolist())
    feat = dFrame['mspec'].values
    targets = dFrame['targets'].values
    feat    = np.concatenate( feat,    axis=0 )
    targets = np.concatenate( targets, axis=0 )
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(feat)
    featTransformed = scaler.transform(feat)
    return np.array([featTransformed.astype('float32'), np_utils.to_categorical(targets, numberOfStates)[np.newaxis]]), scaler



phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
stateList.sort()
stateList = np.array(stateList)
numOfStates = stateList.shape[0]

scalers = {}
for i in ["train", "vali", "test"]:
    for accousticContext in ["Mspec", "Lmfcc", "MspecNone", "LmfccNone"]:
        print(i," - ", accousticContext)
        if i == "train":
            setCurrent = np.load("trainset.npz", allow_pickle = True)['trainset']
        elif i == "vali":
            setCurrent = np.load("valiset.npz", allow_pickle = True)['valiset']
        elif i == "test":
            setCurrent  = np.load("testdata.npz", allow_pickle = True)['testdata']
        if accousticContext == "Mspec":
            setCurrent = accousticContextMspec(setCurrent)
        elif accousticContext == "Lmfcc":
            setCurrent = accousticContextLmfcc(setCurrent)
        elif accousticContext == "MspecNone":
            pass
        elif accousticContext == "LmfccNone":
            pass
        
        standardizationWholeTypeLmfcc
        if i == "train":
            if accousticContext[:5] == "Lmfcc":
                setCurrent, scaler = standardizationWholeTypeLmfcc(setCurrent, numOfStates)
                scalers[accousticContext] = scaler
                np.savez_compressed("set"+i+"_"+accousticContext+".npz", setCurrent=setCurrent)
            else:
                setCurrent, scaler = standardizationWholeTypeMspec(setCurrent, numOfStates)
                scalers[accousticContext] = scaler
                np.savez_compressed("set"+i+"_"+accousticContext+".npz", setCurrent=setCurrent)
        else:
            if accousticContext[:5] == "Lmfcc":
                setCurrent, scaler = standardizationWholeTypeLmfcc(setCurrent, numOfStates, scalers[accousticContext])
                np.savez_compressed("set"+i+"_"+accousticContext+".npz", setCurrent=setCurrent)
            else:
                setCurrent, scaler = standardizationWholeTypeMspec(setCurrent, numOfStates, scalers[accousticContext])
                np.savez_compressed("set"+i+"_"+accousticContext+".npz", setCurrent=setCurrent)
        print("done")

    