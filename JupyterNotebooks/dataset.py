'''
Loading data from pickle pandas table
'''
# https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas
import pandas as pd
import numpy as np

import torch

from torch.utils.data import Dataset, SubsetRandomSampler
from processing import scale_ds

class EntityDataset(Dataset):
    def __init__(self, pandasData, limit_seq_len=None):
        self.x_columns = [
            'StartPoint.X', 'StartPoint.Y', 
            'EndPoint.X', 'EndPoint.Y']

        self.y_columns = [
            'XLine1Point.X', 'XLine1Point.Y',
            'XLine2Point.X', 'XLine2Point.Y']

        self.stroke_columns = ['dx', 'dy', 'is_dim']
        # dx, dy, is_dim, eos
        self.stroke_features = len(self.stroke_columns) + 1

        df = pandasData.dropna(how='all', subset=(self.x_columns + self.y_columns))

        # drop unclustered data, 
        # i.e. outlier points that has not been cluster labeled
        # i.e label==-1
        df = df.drop(df[df["label"] == -1].index)

        groupped = df.groupby(['FileId', 'label'])
        keys = list(groupped.groups.keys())
        
        self.groupped = groupped
        self.data_frame = df
        
        calculated_data = []

        self.max_seq_length = 0

        for key in keys:
            _group = groupped.get_group(key)
            
            group_df = _group[self.x_columns + self.y_columns]
            group_df, _ = scale_ds(group_df)

            # reformat dataset to one table [dx, dy, 'is_dim']
            # is_dim = 1 for dimensions
            group_df['is_dim'] = 0.0
            group_df['dx'] = group_df['EndPoint.X'] - group_df['StartPoint.X']
            group_df['dy'] = group_df['EndPoint.Y'] - group_df['StartPoint.Y']
            
            # https://www.datasciencelearner.com/how-to-merge-two-columns-in-pandas/
            group_df['is_dim'] = np.where(group_df['EndPoint.X'].isna(), 1, 0)
            group_df['dx'] = np.where(group_df['EndPoint.X'].isna(), group_df['XLine1Point.X'] - group_df['XLine2Point.X'], group_df['dx'])
            group_df['dy'] = np.where(group_df['EndPoint.X'].isna(), group_df['XLine1Point.Y'] - group_df['XLine2Point.Y'], group_df['dy'])
            
            seq_len = len(group_df)
            if limit_seq_len is not None:
                if seq_len > limit_seq_len:
                    continue

            if self.max_seq_length < seq_len:
                self.max_seq_length = seq_len

            d = group_df[self.stroke_columns]
            calculated_data.append(d.values)

        #if scale is None:
        #    scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in calculated_data]))
        #self.scale = scale

        # pad all data to max_seq_len
        self.data = torch.zeros(len(calculated_data), self.max_seq_length + 2, self.stroke_features, dtype=torch.float)
        self.mask = torch.zeros(len(calculated_data), self.max_seq_length + 1)

        for i, seq in enumerate(calculated_data):
            seq = torch.from_numpy(seq)
            seq_len = len(seq)

            # dx, dy
            self.data[i,1:seq_len + 1, :2] = seq[:,:2] #/ scale

            #is_dim
            self.data[i,1:seq_len + 1, 2] = seq[:,2]

            #end of sequence
            self.data[i, :seq_len + 1, 3] = 1

            self.mask[i, :seq_len +1] = 1

        # start of sequence
        self.data[:, 0, 2] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.mask[index]

class DwgDataset:
    def __init__(self, pickle_file, batch_size=128, limit_seq_len=10000):
        self.batch_size = batch_size

        test_data = pd.read_pickle(pickle_file)

        self.entities = EntityDataset(test_data, limit_seq_len=limit_seq_len)

        data_len = len(self.entities)

        validation_fraction = 0.2
        test_fraction       = 0.3

        val_split  = int(np.floor(validation_fraction * data_len))
        test_split = int(np.floor(test_fraction * data_len))
        indices = list(range(data_len))
        np.random.seed(42)

        np.random.shuffle(indices)

        val_indices   = indices[:val_split]
        test_indices  = indices[val_split:test_split]
        train_indices = indices[test_split:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler   = SubsetRandomSampler(val_indices)
        test_sampler  = SubsetRandomSampler(test_indices)

        self.train_loader = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = train_sampler)
        self.val_loader   = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = val_sampler)
        self.test_loader  = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = test_sampler)       
