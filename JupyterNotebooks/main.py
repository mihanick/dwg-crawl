'''
Main program for model training and evaluation in python console
version 3. For Rnn and samples groupped by clusters
'''
# https://code.visualstudio.com/docs/remote/wsl-tutorial
import time
import torch
from torch import nn

# https://stackoverflow.com/questions/20309456/call-a-function-from-another-file
from dataset import DwgDataset
from model import DimTransformer
from accuracy import calculate_accuracy, CalculateLoaderAccuracy
from chamfer_distance_loss import ChamferDistance

def run(batch_size=4, pickle_file='test_dataset_cluster_labeled.pickle', lr=0.001, epochs=3):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    dwg_dataset = DwgDataset(pickle_file=pickle_file, batch_size=batch_size)

    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader
    test_loader  = dwg_dataset.test_loader

    ent_features = dwg_dataset.entities.ent_features
    dim_features = dwg_dataset.entities.dim_features

    loss = nn.MSELoss()
    # loss = ChamferDistance(device)

    model = DimTransformer(ent_features=ent_features, dim_features=dim_features, hidden_size=1024, enforced_device=device)
    model.to(device)

    lr = lr
    epochs = epochs
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
            out = model(x, y)

            loss_value = loss(out, y)

            loss_value.backward()
            opt.step()

            loss_accumulated += loss_value

            train_accuracy = calculate_accuracy(out, y)

            with torch.no_grad():
                print('[{:4.0f}-{:4.0f} @ {:5.1f} sec] Loss: {:5.5f} Chamfer distance: {:1.4f}'
                        .format(
                            epoch,
                            i,
                            time.time() - start,
                            float(loss_value),
                            train_accuracy
                        ))
                
        train_history.append(float(train_accuracy))
        loss_history.append(float(loss_accumulated))
        
        # validation
        val_accuracy = CalculateLoaderAccuracy(model, val_loader)
        print('Epoch [{}] validation chamfer distance: {:1.4f}'.format(epoch, val_accuracy))
        val_history.append(float(val_accuracy))

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save model
        torch.save(model.state_dict(), 'DimTransformerTrained.model')

    # Calculate test accuracy
    mean_test_accuracy = CalculateLoaderAccuracy(model, test_loader)
    print('Testing chamfer distance: {:1.4f}'.format(mean_test_accuracy))
    
    return train_history, loss_history, val_history
