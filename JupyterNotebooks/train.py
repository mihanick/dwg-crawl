'''
Train functions to run from console or jupyter
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

def count_relevant(y):
    '''
    counts number of non all-zeroes
    input y : tensor.shape(batch, max_seq_len, features)
    output tensor with shape(batch, 1)
    '''
    n = np.count_nonzero(y.detach().cpu().numpy(), axis=2)
    n = np.count_nonzero(n, axis=1)
    return n
    

def calculate_accuracy2(dim_outputs, y_padded):
    '''
    Calculates the value representing how accurate is prediction of dim_outputs 
    in comparison to given y

    dim_outputs is a torch.tensor.shape(batch, max_sequence_len, dim_features)
    y_padded is a torch.tensor.shape(batch, max_sequence_len, dim_features)
    '''

    y = count_relevant(y_padded)
    dim_out = count_relevant(dim_outputs)

    result = np.zeros_like(y)

    for i in range(len(y)):
        if y[i] != 0:
            result[i] = 1 - np.abs(dim_out[i] - y[i]) / y[i]
        elif dim_out[i] == 0:
            result[i] = 1
        else:
            result[i] = 0
    
    return np.mean(result)

def CalculateLoaderAccuracy(model, loader, input_padder, output_padder):
    with torch.no_grad():
        model.eval()

        val_accuracies = []
        for _, (x, y) in enumerate(loader):
            dim_predictions = model(input_padder(x))

            val_acc = calculate_accuracy2(output_padder(dim_predictions), output_padder(y))
            val_accuracies.append(val_acc)
        val_accuracy = np.mean(val_accuracies)
        return val_accuracy

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
        return 1 - np.sum(diffs**2)

    accuracies = []
    # y, dim_outputs: list len=batch_size of tensors with shape (dim_count, dim_features)
    for batch_no in range(len(y)):
        vol_actual = calculate_volume(y[batch_no])
        vol_predicted = calculate_volume(dim_outputs[batch_no])

        acc = 0
        if vol_actual != 0:
            acc = np.abs(vol_predicted - vol_actual / vol_actual)
            
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
