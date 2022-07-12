#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F_


class RGB2Grad(nn.Module):
    """
    Learns the lookup table for RGB intensity to gelsight gradient.

    Architecture: 5 (R, G, B, x, y) -> 64 -> 64 -> 64 -> 2 (gx, gy)
    """

    dropout_p = 0.05

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = F_.relu(self.fc1(x))
        x = self.drop(x)
        x = F_.relu(self.fc2(x))
        x = self.drop(x)
        x = F_.relu(self.fc3(x))
        x = self.drop(x)
        return self.fc4(x)
        

class GelsightDepthDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.labels = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.labels.iloc[idx, 1:6].to_numpy()
        y = self.labels.iloc[idx, 6:].to_numpy()
        return X.astype(np.float32), y.astype(np.float32)
        # return R, G, B, x, y, gx, gy