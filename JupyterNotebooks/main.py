# https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9

from __future__ import division
from tkinter import image_names

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from IPython import display

from models import CalculatePrediction
from utils.utils import PlotImageAndPrediction

def run(
        use_cuda=False,
        class_path="config/dwg.names",
        data_config_path="config/dwg.data",
        model_config_path="config/yolov3.cfg",
        weights_path="config/yolov3.weights",
        batch_size=4,
        epochs=20,
        checkpoint_interval=1,
        checkpoint_dir="checkpoints",
        n_cpu=0
    ):
    cuda = torch.cuda.is_available() and use_cuda

    os.makedirs("checkpoints", exist_ok=True)

    classes = load_classes(class_path)

    # Get data configuration
    data_config = parse_data_config(data_config_path)
    train_path = data_config["train"]

    # Get hyper parameters
    hyperparams = parse_model_config(model_config_path)[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])

    # Initiate model
    model = Darknet(model_config_path)
    model.load_weights(weights_path)
    #model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.train()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path, max_objects=87), batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            print(
                "[%d/%d]-[%d/%d] [Losses: x %.4f, y %.4f, w %.4f, h %.4f, conf %.4f, cls %.4f, total %.4f, recall: %.3f, precision: %.3f]"
                % (
                    epoch,
                    epochs,
                    batch_i,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )

            model.seen += imgs.size(0)
            
        file_names, imgs, targets = next(iter(dataloader))
        if cuda:
            imgs = imgs.cuda()
            targets = targets.cuda()
            
        dets = CalculatePrediction(model=model, batch_of_images=imgs)
        for i, _img in enumerate(imgs):
            det = dets[i]
            trg = targets[i]
            display.display(PlotImageAndPrediction(image=_img, target=trg, detections=det))

        if epoch % checkpoint_interval == 0:
            model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))
        
        
