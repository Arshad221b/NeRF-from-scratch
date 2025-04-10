import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NeRF(nn.Module): 
    def __init__(self, pos_in_dims=63, dir_in_dims=27, hidden_size=256): 
        super(NeRF, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.layer1 = nn.Linear(pos_in_dims, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size + pos_in_dims, hidden_size)  
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size + pos_in_dims, hidden_size)  
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        self.layer7 = nn.Linear(hidden_size + pos_in_dims, hidden_size)  
        self.layer8 = nn.Linear(hidden_size, hidden_size)
        
        self.sigma_layer = nn.Linear(hidden_size, 1)

        self.dir_layer1 = nn.Linear(hidden_size + dir_in_dims, hidden_size//2)
        self.dir_layer2 = nn.Linear(hidden_size//2, hidden_size//2)
        self.rgb_layer = nn.Linear(hidden_size//2, 3)

    def forward(self, x, d):
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        h = F.relu(self.layer3(torch.cat([h, x], dim=-1)))
        h = F.relu(self.layer4(h))
        h = F.relu(self.layer5(torch.cat([h, x], dim=-1)))
        h = F.relu(self.layer6(h))
        h = F.relu(self.layer7(torch.cat([h, x], dim=-1)))
        h = F.relu(self.layer8(h))

        sigma = self.sigma_layer(h)

        dir_input = torch.cat([h, d], dim=-1)
        h = F.relu(self.dir_layer1(dir_input))
        h = F.relu(self.dir_layer2(h))
        rgb = torch.sigmoid(self.rgb_layer(h))
        
        return torch.cat([rgb, sigma], dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.funcs = [torch.sin, torch.cos]
        
        self.freqs = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        self.freqs = self.freqs * np.pi

    def forward(self, x):
        out = []
        if self.include_input:
            out.append(x)
            
        for freq in self.freqs:
            for func in self.funcs:
                out.append(func(x * freq))
        return torch.cat(out, dim=-1)

class NeRFModel(nn.Module):
    def __init__(self, pos_freqs=10, dir_freqs=4, hidden_size=256):
        super(NeRFModel, self).__init__()
        self.pos_encoder = PositionalEncoding(num_freqs=pos_freqs)
        self.dir_encoder = PositionalEncoding(num_freqs=dir_freqs)
        
        self.nerf = NeRF(pos_in_dims=3*(1 + 2*pos_freqs), 
                        dir_in_dims=3*(1 + 2*dir_freqs),
                        hidden_size=hidden_size)

    def forward(self, points, view_dirs):
        points_encoded = self.pos_encoder(points)
        dirs_encoded = self.dir_encoder(view_dirs)
        return self.nerf(points_encoded, dirs_encoded)

