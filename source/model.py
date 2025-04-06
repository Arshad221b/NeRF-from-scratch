import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NeRF(nn.Module): 
    def __init__(self): 
        super(NeRF, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(63, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):  
        x = self.layers(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs

    def forward(self, x):
        encoding = [x]
        for i in range(self.num_freqs):
            encoding.append(torch.sin(2.0 ** i * np.pi * x))
            encoding.append(torch.cos(2.0 ** i * np.pi * x))
        return torch.cat(encoding, dim=-1)

class NeRFModel(nn.Module):
    def __init__(self, num_freqs=10):
        super(NeRFModel, self).__init__()
        self.positional_encoding = PositionalEncoding(num_freqs)
        self.nerf = NeRF()

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.nerf(x)
        return x
