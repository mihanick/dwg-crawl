'''
Main program for model training and evaluation in python console
'''
# https://code.visualstudio.com/docs/remote/wsl-tutorial
import torch
import pandas as pd
import numpy as np

# https://stackoverflow.com/questions/20309456/call-a-function-from-another-file
from train import train_model
from dataset import DwgDataset
from chamfer_distance_loss import MyChamferDistance
from model import RnnDecoder, RnnEncoder
from train import calculate_accuracy

batch_size = 1

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

dwg_dataset = DwgDataset(pickle_file='test_dataset.pickle', batch_size=batch_size)

train_loader = dwg_dataset.train_loader
val_loader   = dwg_dataset.val_loader
test_loader  = dwg_dataset.test_loader

ent_features = dwg_dataset.entities.ent_features
dim_features = dwg_dataset.entities.dim_features

rnn_encoder = RnnEncoder(ent_features, 512, enforced_device=device).to(device)
rnn_decoder = RnnDecoder(512, dim_features, enforced_device=device).to(device)

loss = MyChamferDistance()

lr = 0.1
epochs = 30
decoder_optimizer = torch.optim.Adam(rnn_decoder.parameters(), lr=lr)
encoder_optimizer = torch.optim.Adam(rnn_encoder.parameters(), lr=0.5*lr)


train_model(
    encoder      = rnn_encoder, 
    decoder      = rnn_decoder, 
    train_loader = train_loader,
    val_loader   = val_loader,
    loss         = loss,
    decoder_opt  = decoder_optimizer,
    encoder_opt  = encoder_optimizer,
    epochs       = epochs)

# print validation results
rnn_encoder.eval()
rnn_decoder.eval()

i = 0
for (x, y) in iter(val_loader):
    outs, learned = rnn_encoder(x)
    decoded = rnn_decoder(outs, learned)
    
    yyy = []
    for yy in y:
        yyy.append(yy.shape[0])
    ppp = []
    for dd in decoded:
        ppp.append(dd.shape[0])
    
    print('actual:', yyy)
    print('predicted:', ppp)
    
    lv = loss(decoded, y)
    print('loss:', lv)

    acc = calculate_accuracy(decoded, y)
    print('accuracy:', acc)

    i += 1
    print(i, '------------------------------')

# Somewhat visualize last predictions 
ii = pd.DataFrame(x[0].cpu().detach().numpy())
print(ii.head())
yy = pd.DataFrame(y[0].cpu().detach().numpy())
print(len(yy))
print(yy.head())
pp = pd.DataFrame(decoded[0].cpu().detach().numpy())
print(len(pp))
print(pp.head())

# Calculate test accuracy

test_accuracies = []
for (x,y) in test_loader:
    with torch.no_grad():
        out, hidden = rnn_encoder(x)
        prediction = rnn_decoder(out, hidden)
        accuracy = calculate_accuracy(prediction, y)
        test_accuracies.append(accuracy)
        
mean_test_accuracy = np.mean(test_accuracies)
print('Accuracy on testing: {0:2.3f}'.format(mean_test_accuracy))