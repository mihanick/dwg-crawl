'''
Main program for model training and evaluation in python console
'''
# https://code.visualstudio.com/docs/remote/wsl-tutorial
import time
import torch
from torch import nn

import numpy as np

# https://stackoverflow.com/questions/20309456/call-a-function-from-another-file
from dataset import DwgDataset
from model import UglyModel, Padder
from train import calculate_accuracy2, count_relevant

batch_size = 2

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

dwg_dataset = DwgDataset(pickle_file='test_dataset.pickle', batch_size=batch_size)

train_loader = dwg_dataset.train_loader
val_loader   = dwg_dataset.val_loader
test_loader  = dwg_dataset.test_loader

ent_features = dwg_dataset.entities.ent_features
dim_features = dwg_dataset.entities.dim_features

max_seq_length = dwg_dataset.max_seq_length

loss = nn.MSELoss()

model = UglyModel(ent_features=ent_features, dim_features=dim_features, max_seq_length=max_seq_length, enforced_device=device)

lr = 0.1
epochs = 3
opt = torch.optim.Adam(model.parameters(), lr=lr)


start = time.time()

loss_history   = []
train_history  = []
val_history    = []
train_accuracy = 0.0

padder = Padder(max_seq_length)

for epoch in range(epochs):
    torch.cuda.empty_cache()
    
    model.train()

    loss_accumulated = 0

    for i, (x, y) in enumerate(train_loader):

        opt.zero_grad()
        decoded = model(padder(x))
        y_padded = padder(y)
        loss_value = loss(decoded, y_padded)

        loss_value.backward()
        opt.step()

        loss_accumulated += loss_value

        train_accuracy = calculate_accuracy2(decoded, y_padded)
        with torch.no_grad():
            print('[{:4.0f}-{:4.0f} @ {:5.1f} sec] Loss: {:5.1f} Train acc: {:1.2f} in: {} out: {} target: {}'
                    .format(
                        epoch,
                        i,
                        time.time() - start,
                        float(loss_value),
                        train_accuracy,
                        count_relevant(padder(x)),
                        count_relevant(decoded),
                        count_relevant(y_padded)
                    ))
            
    train_history.append(float(train_accuracy))
    loss_history.append(float(loss_accumulated))
    
    # validation
    with torch.no_grad():
        model.eval()

        val_accuracies = []
        for _, (x, y) in enumerate(val_loader):
            dim_predictions = model(padder(x))

            val_acc = calculate_accuracy2(dim_predictions, padder(y))
            val_accuracies.append(val_acc)
        val_accuracy = np.mean(val_accuracies)
        print('Epoch [{}] validation error: {:1.3f}'.format(epoch, val_accuracy))
        val_history.append(float(val_accuracy))

    # show memory consumption
    # https://stackoverflow.com/questions/110259/which-python-memory-profiler-is-recommended
    #print("--------------------memory consumption:")
    #heap = hpy()
    #print(heap.heap())
    #print("--------------------------------------:")


# Calculate test accuracy

test_accuracies = []
for (x,y) in test_loader:
    with torch.no_grad():
        prediction = model(padder(x))
        accuracy = calculate_accuracy2(prediction, padder(y))
        test_accuracies.append(accuracy)
        
mean_test_accuracy = np.mean(test_accuracies)
print('Accuracy on testing: {}'.format(mean_test_accuracy))