import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.model import NeRFModel
from source.dataloaders import CustomDataloader
import torch
import torch.nn as nn    
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class TrainModel: 
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeRFModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.batch_size = 1

        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        base_path = os.path.join(workspace_root, 'nerf_synthetic', 'chair')
        data_path = os.path.join(base_path, 'train')  # Path to images
        transforms_path = os.path.join(base_path, 'transforms_train.json')
        
        self.dataloader = CustomDataloader(self.batch_size, data_path, transforms_path)
        self.epochs = 1000
        
        self.loss_function = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.start_epoch = 0
        self.checkpoint_path = 'checkpoint.pth'

    #ChatGPT Generated Function
    def volume_rendering(self, outputs, z_vals, rays_d):
        rgb = outputs[..., :3]                  # [B, N, 3]
        sigma = outputs[..., 3]                 # [B, N]
        sigma = torch.clamp(sigma, min=0.0, max=10.0)
        # Compute distances between z samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [B, N-1]
        last_dist = 1e10 * torch.ones_like(dists[..., :1])  # [B, 1]
        dists = torch.cat([dists, last_dist], dim=-1)       # [B, N]

        # Scale by ray direction magnitude
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)  # [B, N]

        # Volume rendering steps
        alpha = 1.0 - torch.exp(-sigma * dists)  # [B, N]
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        weights = alpha * transmittance  # [B, N]

        # RGB + Depth
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [B, 3]
        depth_map = torch.sum(weights * z_vals, dim=-1)        # [B]

        return rgb_map, depth_map
    
    def train(self):
        try:
            torch.cuda.empty_cache()
            for epoch in range(self.start_epoch, self.epochs):
                for i, data in enumerate(self.dataloader):
                    try:
                        for key, value in data.items():
                            data[key] = value.to(self.device)

                        points = data['points'].to(self.device)
                        rays_d = data['rays_d'].to(self.device)
                        z_vals = data['z_vals'].to(self.device)
                        rgb_gt = data['rgb_gt'].to(self.device)
                        
                        batch_size = points.shape[0]
                        
                        if rays_d.shape[0] != batch_size:
                            rays_d = rays_d[:batch_size]
                        
                        if z_vals.shape[0] != batch_size:
                            z_vals = z_vals[:batch_size]
                        
                        if rgb_gt.shape[0] != batch_size:
                            rgb_gt = rgb_gt[:batch_size]
                        
                        if points.shape[1] != z_vals.shape[1]:
                            min_samples = min(points.shape[1], z_vals.shape[1])
                            points = points[:, :min_samples, :]
                            z_vals = z_vals[:, :min_samples]
                        
                        self.optimizer.zero_grad()
                        outputs = self.model(points)
                        
                        rgb_pred, depth_pred = self.volume_rendering(outputs, z_vals, rays_d)
                        loss = self.loss_function(rgb_pred, rgb_gt)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                        if i % 100 == 0:
                            print(f'Epoch [{epoch}/{self.epochs}], Step [{i}/{len(self.dataloader)}], Loss: {loss.item():.4f}')
                    except Exception as e:
                        print(f"Error in batch {i}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue

                self.scheduler.step()

                if (epoch + 1) % 10 == 0:
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(f'Model saved to {self.checkpoint_path}')
            print('Training complete.')
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
        # self.model.eval()   

if __name__ == "__main__":
    train_model = TrainModel()
    train_model.train()