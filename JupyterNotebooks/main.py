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

    train_kl_losses = []
    train_rl_losses = []
    val_kl_losses   = []
    val_rl_losses   = []
    
    for epoch in range(epochs):
        test_rl, train_kl, val_rl, val_kl = trainer.train_epoch(epoch)
        train_kl_losses.append(train_kl)
        train_rl_losses.append(test_rl)
        val_kl_losses.append(val_kl)
        val_rl_losses.append(val_rl)

    # Calculate test accuracy
    train_kl, test_rl = trainer.CalculateLoaderAccuracy(trainer.test_loader)
    print('Test losses rl:{:1.4f} kl:{:1.4f}'.format(test_rl, train_kl))

    return train_rl_losses, train_kl_losses, val_rl_losses, val_kl_losses, test_rl, train_kl
