'''
Contains functions and modules to calculate chamfer distance
'''
import numpy as np
import numpy.linalg as LA
from sklearn.neighbors import KDTree
import torch
from torch import nn
from model import NNModulePrototype

# https://stackoverflow.com/questions/47060685/chamfer-distance-between-two-point-clouds-in-tensorflow

def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point, 1)),
                    (-1, num_features))
    distances = LA.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    batch_size, _num_point, _num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist

def chamfer_distance_sklearn(array1, array2):
    batch_size, num_point = array1.shape[:2]
    dist = 0
    for i in range(batch_size):
        tree1 = KDTree(array1[i], leaf_size=num_point+1)
        tree2 = KDTree(array2[i], leaf_size=num_point+1)
        distances1, _ = tree1.query(array2[i])
        distances2, _ = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist

def my_chamfer_distance(out, y):
    a = out.unsqueeze(0).cpu().detach().numpy()
    b = y.unsqueeze(0).cpu().detach().numpy()
    return chamfer_distance_sklearn(a, b)

class ChamferDistance(NNModulePrototype):
    def forward(self, x, target):
        d = my_chamfer_distance(x, target)
        result = torch.as_tensor(d, device=self.device)
        result.requires_grad = True
        return result
