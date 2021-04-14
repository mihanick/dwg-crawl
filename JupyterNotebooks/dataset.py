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
    def __init__(self, pandasData, limit_seq_len=None, min_seq_length=20):
        self.x_columns = [
            'StartPoint.X', 'StartPoint.Y', 
            'EndPoint.X', 'EndPoint.Y']

        self.y_columns = [
            'XLine1Point.X', 'XLine1Point.Y',
            'XLine2Point.X', 'XLine2Point.Y']

        self.min_seq_length = min_seq_length

        self.stroke_columns = ['dx', 'dy', 'is_pen', 'is_dim']
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

        max_seq_length = 0

        for key in keys:
            _group = groupped.get_group(key)
            
            group_df = _group[self.x_columns + self.y_columns]
            group_df, _ = scale_ds(group_df)

            seq_len = len(group_df)

            if limit_seq_len is not None:
                if seq_len > limit_seq_len:
                    continue
                
            #skip small groups
            if seq_len < self.min_seq_length:
                continue

            # reformat dataset to one table [dx, dy, 'is_pen', 'is_dim']
            # is_pen = 1 for dims and lines, is_pen = 0 for pen transition
            # is_dim = 1 for dimensions

            # https://www.datasciencelearner.com/how-to-merge-two-columns-in-pandas/
            group_df['is_dim'] = np.where(group_df['EndPoint.X'].isna(), 1, 0)
            group_df['StartPoint.X'] = np.where(group_df['is_dim'] == 1, group_df['XLine1Point.X'], group_df['StartPoint.X'])
            group_df['StartPoint.Y'] = np.where(group_df['is_dim'] == 1, group_df['XLine1Point.Y'], group_df['StartPoint.Y'])
            group_df['EndPoint.X'] = np.where(group_df['is_dim'] == 1, group_df['XLine2Point.X'], group_df['EndPoint.X'])
            group_df['EndPoint.Y'] = np.where(group_df['is_dim'] == 1, group_df['XLine2Point.Y'], group_df['EndPoint.Y'])
            

            # calculated numpy data
            # for each line of group_df
            # we will add 2 lines in npd:
            # one for pen movement from end of (prevx, prevy) to current x1,x2 (is_pen=0)
            # and one for pen drawing from x1,y1 to x2,y2
            # is_dim will be written to numpy data as well
            npd = np.zeros((seq_len*2, 4), dtype=np.float)

            prevx = 0
            prevy = 0

            # TODO: More optimization needed, like npd[:]
            i = 0
            for  row  in group_df[['StartPoint.X', 'StartPoint.Y', 'EndPoint.X', 'EndPoint.Y', 'is_dim']].values:
                x1 = row[0]
                y1 = row[1]
                x2 = row[2]
                y2 = row[3]

                npd[2*i + 1, 3] = row[4]

                npd[2*i + 1, 0] = x2 - x1
                npd[2*i + 1, 1] = y2 - y1
                npd[2*i + 1, 2] = 1

                npd[2*i, 0] = x1 - prevx
                npd[2*i, 1] = y1 - prevy
                npd[2*i, 2] = 0

                prevx = x2
                prevy = y2
                
                #print (row)
                # print (npd[2*i:2*i+2])

                i+=1
            
            #print(group_df)
            #print(npd)
            
            if max_seq_length < npd.shape[0]:
                max_seq_length = npd.shape[0]

            calculated_data.append(npd)

        self.max_seq_length = max_seq_length + 2 # add sos and eos

        # pad all data to max_seq_len
        self.data = torch.zeros(len(calculated_data), self.max_seq_length, self.stroke_features, dtype=torch.float)
        self.mask = torch.zeros(len(calculated_data), self.max_seq_length - 1)

        for i, seq in enumerate(calculated_data):
            seq = torch.from_numpy(seq)
            seq_len = len(seq)

            # dx, dy
            self.data[i, 1:seq_len + 1, :2] = seq[:, :2] #/ scale

            #is_pen
            self.data[i, 1:seq_len + 1, 2] = seq[:, 2]

            #is_dim
            #self.data[i, 1:seq_len + 1, 3] = seq[:, 3]
            #is_lift
            self.data[i, 1:seq_len + 1, 3] = 1 -  seq[:, 2]

            #end of sequence
            self.data[i, seq_len + 1:, 4] = 1

            self.mask[i, :seq_len + 1] = 1

        # start of sequence
        self.data[:, 0, 2] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.mask[index]

class DwgDataset:
    def __init__(self, pickle_file, batch_size=128, limit_seq_len=10000, min_seq_length=20):
        self.batch_size = batch_size

        test_data = pd.read_pickle(pickle_file)

        self.entities = EntityDataset(test_data, limit_seq_len=limit_seq_len, min_seq_length=min_seq_length)

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
