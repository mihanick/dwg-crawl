'''
Train functions to run from console or jupyter
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

def calculate_accuracy(dim_outputs, y):
    '''
    For now we will calculate accuracy by model guessing the right amount of dimensions produced
    '''

    accuracies = []

    # y, dim_outputs: list len=batch_size of tensors with shape (dim_count, dim_features)
    for batch_no in range(len(y)):
        actual_dim_number = y[batch_no].shape[0]
        predicted_dim_number = dim_outputs[batch_no].shape[0]
        acc = 0
        if actual_dim_number == 0 and predicted_dim_number == 0:
            acc = 0
        else:

            acc = torch.abs(torch.mean(y[batch_no], dim=0) - torch.mean(dim_outputs[batch_no], dim=0)).detach().cpu().numpy()
        # print('expected: {0} predicted: {1}'.format(actual_dim_number, predicted_dim_number))
        accuracies.append(acc)

    return accuracies

def plot_history(loss_history, train_history, val_history):
    '''
    Plots learning history in jupyter
    '''
    plt.ylabel('Accuracy @ epoch')

    train, = plt.plot(train_history)
    train.set_label("train")

    validation, = plt.plot(val_history)
    validation.set_label("validation")

    loss, = plt.plot(loss_history)
    loss.set_label("loss")

    plt.legend()
    plt.show()

def train_model(encoder, decoder, train_loader, val_loader, loss, decoder_opt, encoder_opt, epochs):
    '''
    trains model and outputs loss, train and validation history
    '''

    start = time.time()

    loss_history   = []
    train_history  = []
    val_history    = []
    train_accuracy = 0.0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        loss_accumulated = 0

        for i, (x, y) in enumerate(train_loader):

            decoder_opt.zero_grad()
            encoder_opt.zero_grad()

            outs_num, learned_hidden_representation = encoder(x)

            decoded = decoder(outs_num, learned_hidden_representation)
            loss_value = loss(decoded, y)

            loss_value.backward()
            decoder_opt.step()
            encoder_opt.step()

            loss_accumulated += loss_value

            train_accuracy = np.mean(calculate_accuracy(decoded, y))

            print('[{0}-{1} @ {2:.1f} sec] Log10 Loss: {3:2f} Train err: {4:2.1f}%'. format(
                epoch,
                i,
                time.time() - start,
                np.log10(float(loss_value)),
                (1 - train_accuracy) * 100
            ))

        train_history.append(train_accuracy)
        loss_history.append(float(loss_accumulated))

        # validation
        encoder.eval()
        decoder.eval()

        val_accuracies = []
        for _, (x, y) in enumerate(val_loader):
            outs, hiddens = encoder(x)
            dim_predictions = decoder(outs, hiddens)

            val_acc = np.mean(calculate_accuracy(dim_predictions, y))
            val_accuracies.append(val_acc)
        val_accurcy = np.mean(val_accuracies)
        print('Epoch validation accuracy: {0:2.3f}'.format(val_accurcy))
        val_history.append(val_accurcy)

    return loss_history, train_history, val_history
