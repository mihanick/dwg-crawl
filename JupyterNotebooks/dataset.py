# https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, SubsetRandomSampler

from sklearn import preprocessing

class EntityDataset(Dataset):
    def __init__(self, pandasData):
        self.data = pandasData
        self.groupped = pandasData.groupby('FileId')
        self.keys = list(self.groupped.groups.keys() )
        
        self.x_columns = [
            #'ClassName', 
            'StartPoint.X', 'StartPoint.Y', 'StartPoint.Z',
            'EndPoint.X', 'EndPoint.Y', 'EndPoint.Z',
            'Position.X', 'Position.Y', 'Position.Z']

        self.y_columns = [
            #'ClassName', 
            'XLine1Point.X', 'XLine1Point.Y','XLine1Point.Z', 
            'XLine2Point.X', 'XLine2Point.Y', 'XLine2Point.Z']

        self.ent_features = len(self.x_columns)
        self.dim_features = len(self.y_columns)

        self.min_max_scaler = preprocessing.MinMaxScaler()

    def __len__(self):
        return len(self.keys)

    def _normalize_dataset(self, data):
        '''
        Normalizes the values in dataset using sklean preprocessing
        from sklearn import preprocessing
        https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
        '''

        # handle empty sets
        if len(data) == 0:
            return data

        x = data.values
        x_scaled = self.min_max_scaler.fit_transform(data)

        return pd.DataFrame(x_scaled)

    def __getitem__(self, index):
        # https://stackoverflow.com/questions/45147100/pandas-drop-columns-with-all-nans
        
        x = self.groupped.get_group(self.keys[index])[self.x_columns]
        x = x.dropna( how ='all')
        x = x.fillna(0.0)
        x = self._normalize_dataset(x)

        y = self.groupped.get_group(self.keys[index])[self.y_columns]
        y = y.dropna(how = 'all')
        y = y.fillna(0.0)
        y = self._normalize_dataset(y)
        
        #print(self.keys[index])
        # if not y.empty:
        #    print(x, y)
        
        return torch.FloatTensor(np.array(x.values)), torch.FloatTensor(np.array(y.values))

class DwgDataset:
    def __init__(self, pickle_file, batch_size=1):

        test_data = pd.read_pickle(pickle_file)

        self.entities = EntityDataset(test_data)

        data_len = len(self.entities)

        validation_fraction = 0.2
        test_fraction       = 0.3

        val_split  = int(np.floor(validation_fraction * data_len))
        test_split = int(np.floor(test_fraction * data_len))
        indices = list(range(data_len))
        np.random.seed(228)

        np.random.shuffle(indices)

        val_indices   = indices[:val_split]
        test_indices  = indices[val_split:test_split]
        train_indices = indices[test_split:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler   = SubsetRandomSampler(val_indices)
        test_sampler  = SubsetRandomSampler(test_indices)

        # https://stackoverflow.com/questions/64586575/adding-class-objects-to-pytorch-dataloader-batch-must-contain-tensors
        def custom_collate(sample):
            x = []
            y = []

            for (xx, yy) in sample:
                #if torch.count_nonzero(xx) > 0:
                    # print(xx)
                x.append(xx)
                #if torch.count_nonzero(yy) > 0:
                y.append(yy)
            #print(len(x), len(y))
            return x, y

        self.train_loader = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = train_sampler, collate_fn=custom_collate, drop_last=True)
        self.val_loader   = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = val_sampler, collate_fn=custom_collate, drop_last=True)
        self.test_loader  = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = test_sampler, collate_fn=custom_collate)

        self.max_seq_length = 0
        self.MaxSeqLength(self.train_loader)
        self.MaxSeqLength(self.val_loader)
        self.MaxSeqLength(self.test_loader)

    def MaxSeqLength(self, loader):
        for (x, y) in loader:
            for xx in x: 
                l = xx.shape[0] 
                if l > self.max_seq_length:
                    self.max_seq_length = l

        