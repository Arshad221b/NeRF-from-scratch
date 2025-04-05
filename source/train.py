from model import NeRFModel
from dataloaders import CustomDataloader
import torch
import torch.nn as nn    
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class trainModel: 
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeRFModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.dataloader = CustomDataloader(32, 'nerf_synthetic/chair/train', 'nerf_synthetic/chair/transforms_train.json')
        self.epochs = 100
        self.batch_size = 32
        self.loss_function = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.start_epoch = 0
        self.checkpoint_path = 'checkpoint.pth'

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            for i, data in enumerate(self.dataloader):
                points = data['points'].to(self.device)
                images = data['images'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(points)
                loss = self.loss_function(outputs, images)
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch [{epoch}/{self.epochs}], Step [{i}/{len(self.dataloader)}], Loss: {loss.item():.4f}')

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f'Model saved to {self.checkpoint_path}')
        print('Training complete.')
        # self.model.eval()   

trainModel = trainModel()
trainModel.train()