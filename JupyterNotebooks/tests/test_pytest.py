'''
Contains tests for input and output of models and also for loss functions
'''

#https://code.visualstudio.com/docs/python/testing
import torch
from torch import nn
from model import MyRnn
from dataset import DwgDataset, EntityDataset
from train import calculate_accuracy, calculate_accuracy2, count_relevant
import pandas as pd

def test_dataset():
    d = pd.read_pickle('test_dataset_cluster_labeled.pickle')

    for x,y in EntityDataset(d):
        print(x.shape,y)

def test_model():
    dwg_dataset = DwgDataset(pickle_file='test_dataset_cluster_labeled.pickle', batch_size=4)
    loader = dwg_dataset.train_loader
    model = MyRnn(loader.ent_features, loader.dim_features, 16)
    for i, (x, y) in enumerate(loader):
        out = model(x)
        print(out)
        
def _check_loss(loss_layer, print_loss=False):
    loss = loss_layer
    _a = [torch.randn([3, 6]), torch.randn([2, 6])]
    _b = [torch.randn([4, 6]), torch.randn([1, 6])]

    loss_v = loss(_a, _b)

    print('check random sets')
    print(loss_v)

    loss_v.backward()
    if print_loss:
        print(loss_v)

    acc = calculate_accuracy(_a, _b)
    print(acc)

    print("check empty sets")
    _a = [torch.randn([0, 6]), torch.randn([2, 6])]
    _b = [torch.randn([4, 6]), torch.randn([0, 6])]

    loss_v = loss(_a, _b)
    print(loss_v)

    # RuntimeError: leaf variable has been moved into the graph interior
    # https://stackoverflow.com/questions/64272718/understanding-the-reason-behind-runtimeerror-leaf-variable-has-been-moved-into

    loss_v.backward()
    acc = calculate_accuracy(_a, _b)
    print(acc)

