import math
import numpy as np
import numpy.linalg as LA
from sklearn.neighbors import KDTree
import torch
from torch import nn

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
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist


def chamfer_distance_sklearn(array1,array2):
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
    loss = torch.zeros(1, dtype = torch.float32, requires_grad = True)

    for i in range(len(y)):
        # we need extra dimension for batch
        xxx = torch.unsqueeze(out[i], dim = 0)
        yyy = torch.unsqueeze(y[i],dim = 0)    
        # print(yyy.shape)
        # print(xxx.shape)    

        if xxx.shape[1] == 0 and yyy.shape[1] == 0:
            # loss += 0
            continue
        if xxx.shape[1] == 0:
            loss = loss + math.inf
            continue
        if yyy.shape[1] == 0:
            loss = loss + math.inf
            continue
        # https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308
        curr_loss = chamfer_distance_sklearn(xxx.detach().numpy(), yyy.detach().numpy()).sum()
        loss = loss + curr_loss
    return loss

class MyChamferDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return my_chamfer_distance(input, target)