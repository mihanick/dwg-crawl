'''
Train functions to run from console or jupyter
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

def CalculateLoaderAccuracy(model, loader, input_padder, output_padder):
    with torch.no_grad():
        model.eval()

        val_accuracies = []
        for _, (x, y) in enumerate(loader):
            dim_predictions = model(input_padder(x))

            val_acc = calculate_accuracy(dim_predictions, y)
            val_accuracies.append(val_acc)
        val_accuracy = torch.mean(val_accuracies)
        return val_accuracy

def calculate_accuracy(dim_outputs, y):
    
    batch_size = y.shape[0]
    accuracies = torch.FloatTensor((batch_size,1))

    for i in range(batch_size):
        z = torch.zeros_like(y[i])
        if (y[i].eq(z) and dim_outputs[i].eq(z)):
            accuracies[i] = 1
        elif y[i].eq(z):
            accuracies[i] = 0
        else
            accuracies[i] = torch.mean(torch.abs(dim_outputs[i] - y[i])/torch.abs(y[i]))

    return torch.mean(accuracies)

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

            print('[{:4.0f}-{:4.0f} @ {:5.1f} sec] Loss: {:5.1f} Train acc: {:1.2f} in: {:5.0f} out: {:5.0f} target: {:5.0f}'
                  .format(
                        epoch,
                        i,
                        time.time() - start,
                        float(loss_value),
                        train_accuracy,
                        x[0].shape[0],
                        decoded[0].shape[0],
                        y[0].shape[0]
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
            print('Epoch [{}] validation error: {:1.3f}'.format(epoch, val_accuracy))
            val_history.append(float(val_accuracy))
        
    return loss_history, train_history, val_history
