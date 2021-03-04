# https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, SubsetRandomSampler
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from sklearn import preprocessing


class EntityDataset(Dataset):
    def __init__(self, pandasData):
        self.x_columns = [
            'StartPoint.X', 'StartPoint.Y', 'StartPoint.Z',
            'EndPoint.X', 'EndPoint.Y', 'EndPoint.Z']

        self.y_columns = [
            'XLine1Point.X', 'XLine1Point.Y','XLine1Point.Z', 
            'XLine2Point.X', 'XLine2Point.Y', 'XLine2Point.Z']

        self.ent_features = len(self.x_columns)
        self.dim_features = len(self.y_columns)

        df = pandasData.dropna(how='all', subset=(self.x_columns + self.y_columns))
        
        # drop unclustered data
        df = df.drop(df[df["label"] == -1].index)
        #print(df)
        #data = self.min_max_scaler.fit_transform(data)
    
        groupped = df.groupby(['FileId', 'label'])
        keys = list(groupped.groups.keys())
        
        self.data = []
        y_cache = None
        
        for key in keys:
            _group = groupped.get_group(key)
            x = _group[self.x_columns]
            x = x.dropna(how ='all')
            x = x.fillna(0.0)
            x = torch.FloatTensor(x.values)
            
            if y_cache is None:
                _y = _group[self.y_columns]
                _y = _y.dropna(how = 'all')
                _y = _y.fillna(0.0)
                y_cache = list(_y.to_numpy())

            y = None
            if len(y_cache) > 0:
                #pop
                for _y in y_cache:
                    y = _y.reshape(1, self.dim_features)
                    y = torch.FloatTensor(y)
                    self.data.append((x,y))
            else:
                y = torch.zeros((1, self.dim_features), dtype=torch.float32)
                y_cache = None
            
            self.data.append((x, y))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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
