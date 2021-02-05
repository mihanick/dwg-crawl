import numpy as np
import matplotlib.pyplot as plt


import time

def plot_history(loss_history, train_history, val_history):
    plt.ylabel('Accuracy @ epoch')
    
    train, = plt.plot(train_history)
    train.set_label("train")
    validation, = plt.plot(val_history)
    validation.set_label("validation")

    plt.legend()
    plt.show()
    
def train_model(encoder, decoder, train_loader, val_loader, loss, decoder_opt,encoder_opt, epochs):
    start = time.time()
    
    loss_history = []
    train_history = []
    val_history  = []
    train_accuracy = 0.0
    val_accuracy = 0.0
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        loss_accumulated = 0
        
        i_step = 1
        for _, (x,y) in enumerate(train_loader):
            
            decoder_opt.zero_grad()
            encoder_opt.zero_grad()
            
            outs_num, learned = encoder(x)
            
            decoded = decoder(learned, outs_num)
            loss_value = loss(decoded, y)
            
            loss_value.backward()
            decoder_opt.step()
            encoder_opt.step()
            
            loss_accumulated += loss_value
            
            #val_accuracy = compute_accuracy(model, val_loader)
            print('[{0} @ {4:.1f} sec] Loss: {1:4f} Train err: {2:2.2f}% Val err: {3:2.2f}%'. format(
                epoch,
                loss_value.item(),
                (1 - train_accuracy) * 100,
                (1 - val_accuracy) * 100,
                time.time() - start
            ))
        
        # loss_history.append(float(loss_accum / i_step))
        # val_history.append(val_accuracy)
        # clear_output(wait=True)
        
        i_step += 1

    return loss_history, train_history, val_history        
