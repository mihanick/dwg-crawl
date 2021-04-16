'''
Main program for model training and evaluation in python console
version 4. For SketchRNN and straight sequence of entities
'''
# https://code.visualstudio.com/docs/remote/wsl-tutorial

import torch
from torch import nn
from sketch_rnn import Trainer, DecoderRNN, EncoderRNN
from dataset import DwgDataset

def run(batch_size=32, pickle_file='test_dataset_cluster_labeled.pickle', lr=0.008, epochs=5, train_verbose=True, limit_seq_len=200):
    dwg_dataset = DwgDataset(pickle_file=pickle_file, batch_size=batch_size, limit_seq_len=limit_seq_len)
    trainer = Trainer(
        dwg_dataset,
        lr=lr,
        train_verbose=train_verbose)

    
    train_ls = []
    train_lp = []
    train_lkl = []

    val_ls = []
    val_lp = []
    val_lkl = []
    
    for epoch in range(epochs):
        train_ls_, train_lp_, train_lkl_, val_ls_, val_lp_, val_lkl_ = trainer.train_epoch(epoch)
        train_ls.append(train_ls_)
        train_lp.append(train_lp_)
        train_lkl.append(train_lkl_)
        val_ls.append(val_ls_)
        val_lp.append(val_lp_)
        val_lkl.append(val_lkl_)

    # Calculate test accuracy
    test_ls, test_lp, test_lkl = trainer.CalculateLoaderAccuracy(trainer.test_loader)
    print('Test losses ls:{:1.4f} lp:{:1.4f} kl:{:1.4f}'.format(test_ls, test_lp, test_lkl))

    return train_ls, train_lp, train_lkl, val_ls, val_lp, val_lkl
