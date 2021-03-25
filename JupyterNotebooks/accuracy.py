import numpy as np
import torch
from chamfer_distance_loss import my_chamfer_distance

def CalculateLoaderAccuracy(model, loader):
    with torch.no_grad():
        model.eval()

        val_accuracies = []
        for _, (x, y) in enumerate(loader):
            dim_predictions = model(x, y)

            val_acc = calculate_accuracy(dim_predictions, y)
            val_accuracies.append(val_acc)
        val_accuracy = np.mean(val_accuracies)
        return val_accuracy

def calculate_accuracy(dim_outputs, y):
    l = my_chamfer_distance(dim_outputs, y)

    return l
