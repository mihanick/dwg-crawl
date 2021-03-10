'''
Contains tests for input and output of models and also for loss functions
'''

#https://code.visualstudio.com/docs/python/testing
import torch
from torch import nn
from model import DimRnn
from dataset import DwgDataset, EntityDataset
import pandas as pd

def test_dataset():
    d = pd.read_pickle('test_dataset_cluster_labeled.pickle')

    for x,y in EntityDataset(d):
        print(x.shape,y)
