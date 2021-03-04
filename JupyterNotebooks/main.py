'''
Main program for model training and evaluation in python console
version 3. For Rnn and samples groupped by clusters
'''
# https://code.visualstudio.com/docs/remote/wsl-tutorial
import time
import torch
from torch import nn

import numpy as np

# https://stackoverflow.com/questions/20309456/call-a-function-from-another-file
from dataset import DwgDataset
from model import MyRnn
from train import calculate_accuracy2, count_relevant, CalculateLoaderAccuracy
from chamfer_distance_loss import ChamferDistance2

batch_size = 4

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

dwg_dataset = DwgDataset(pickle_file='test_dataset_cluster_labeled.pickle', batch_size=batch_size)

train_loader = dwg_dataset.train_loader
val_loader   = dwg_dataset.val_loader
test_loader  = dwg_dataset.test_loader

ent_features = dwg_dataset.entities.ent_features
dim_features = dwg_dataset.entities.dim_features

loss = ChamferDistance2()

model = MyRnn(ent_features, dim_features, 16, device)
model.to(device)

lr = 0.001
epochs = 5
opt = torch.optim.Adam(model.parameters(), lr=lr)

start = time.time()

loss_history   = []
train_history  = []
val_history    = []
train_accuracy = 0.0

for epoch in range(epochs):
    torch.cuda.empty_cache()
    
    model.train()

    loss_accumulated = 0

    for i, (x, y) in enumerate(train_loader):

        opt.zero_grad()
        out = model(x)

        loss_value = loss(out, y)

        loss_value.backward()
        opt.step()

        loss_accumulated += loss_value

        train_accuracy = calculate_accuracy2(decoded_p, y_padded)
        with torch.no_grad():
            print('[{:4.0f}-{:4.0f} @ {:5.1f} sec] Loss: {:5.4f} Train acc: {:1.2f} in: {} out: {} target: {}'
                    .format(
                        epoch,
                        i,
                        time.time() - start,
                        float(loss_value),
                        train_accuracy,
                        count_relevant(x_p),
                        count_relevant(decoded_p),
                        count_relevant(y_padded)
                    ))
            
    train_history.append(float(train_accuracy))
    loss_history.append(float(loss_accumulated))
    
    # validation
    val_accuracy = CalculateLoaderAccuracy(model, val_loader, input_padder, output_padder)
    print('Epoch [{}] validation accuracy: {:1.3f}'.format(epoch, val_accuracy))
    val_history.append(float(val_accuracy))


# Calculate test accuracy
mean_test_accuracy = CalculateLoaderAccuracy(model, test_loader, input_padder, output_padder)
print('Accuracy on testing: {}'.format(mean_test_accuracy))
