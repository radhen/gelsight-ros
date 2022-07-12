#!/usr/bin/env python3

"""
Labels data for training a depth reconstruction model.
Can collect images using the 'record.py' script.

The dataset can only include the impression and rolling
of a single spherical object (like a marble). The first image
should also have no impression, for purposes of training from the
pixel-wise difference.

Directions:
- Press 'Y' to accept current circle label into the dataset
- Press 'N' to discard current image
- Press 'Q' to exit the program
- Click, drag and release to manually label a circle (replaces current label)

From a technical perspective, this script uses the known radius
of a spherical object to estimate the gradient at every contact
point. From the gradients, it can then generate a dataset in CSV
format which relates (R, G, B, x, y) -> (gx, gy). For inference,
you can then use poisson reconstruction to build the depth
map from gradients.

You can train a new model from the output dataset using the 'train.py' script.
"""

import csv
from csv import writer
import cv2
import gelsight_ros as gsr
import math
import numpy as np
import os
import rospy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = "cuda"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    rospy.init_node("train")

    # Retrieve path where images are saved
    if not rospy.has_param("~input_path"):
        rospy.signal_shutdown("No input path provided. Please set input_path/.")
    input_path = rospy.get_param("~input_path")

    # Retrieve path where dataset will be saved    
    if not rospy.has_param("~output_path"):
        rospy.signal_shutdown("No output path provided. Please set output_path/.")
    output_path = rospy.get_param("~output_path")
    if output_path[-1] == "/":
        output_path = output_path[:len(output_path)-1]

    if not os.path.exists(output_path):
        rospy.logwarn("Output folder doesn't exist, will create it.")
        os.makedirs(output_path)
        
        if not os.path.exists(output_path):
            rospy.signal_shutdown(f"Failed to create output folder: {output_path}")
    output_file = output_path + "/gelsight-depth-dataset.csv"

    # Create dataset
    batch_size = 64
    dataset = gsr.GelsightDepthDataset(input_path)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(trainset, batch_size=batch_size)
    test_dataloader = DataLoader(testset, batch_size=batch_size)

    # Initiate model and optimizer
    model = gsr.RGB2Grad().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train model
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")