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
 
    def calculate_volume(set):
        size = set.shape[0]
        features = set.shape[1]

        min_coords = np.zeros((1, features))
        max_coords = np.zeros((1, features))

        if size != 0:
            min_coords = np.min(set.detach().cpu().numpy(), axis=0)
            max_coords = np.max(set.detach().cpu().numpy(), axis=0)

        # if size is 1 difference will be 0 and volume will be 0

        
        diffs = max_coords - min_coords
        return np.sum(diffs**2)

    accuracies = []
    # y, dim_outputs: list len=batch_size of tensors with shape (dim_count, dim_features)
    for batch_no in range(len(y)):
        vol_actual = calculate_volume(y[batch_no])
        vol_predicted = calculate_volume(dim_outputs[batch_no])

        acc = 0
        if vol_actual == vol_predicted:
            acc = 1
        elif vol_predicted == 0:
            acc = 0
        elif vol_actual != 0:
            acc = 1 - np.abs(vol_predicted -vol_actual/ vol_actual)
            
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

    loss, = plt.plot(np.log10(loss_history))
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
        torch.cuda.empty_cache()
        
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

            print('[{0}-{1} @ {2:.1f} sec] Loss: {3:2f} Train err: {4:2.1f}%'. format(
                epoch,
                i,
                time.time() - start,
                float(loss_value),
                100.0 * (1 - train_accuracy)
            ))
            
        train_history.append(float(train_accuracy))
        loss_history.append(float(loss_accumulated))
        
        

        # validation
        with torch.no_grad():
            encoder.eval()
            decoder.eval()

            val_accuracies = []
            for _, (x, y) in enumerate(val_loader):
                outs, hiddens = encoder(x)
                dim_predictions = decoder(outs, hiddens)

                val_acc = np.mean(calculate_accuracy(dim_predictions, y))
                val_accuracies.append(val_acc)
            val_accuracy = np.mean(val_accuracies)
            print('Epoch [{0}] validation error: {1:2.3f}%'.format(epoch, 100.0 * (1 - val_accuracy)))
            val_history.append(float(val_accuracy))
        
        # free memory
        del(loss_accumulated)
        # del(outs_num, learned_hidden_representation, decoded)

    return loss_history, train_history, val_history
