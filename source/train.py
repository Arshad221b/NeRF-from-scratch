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
        self.model = NeRFModel(pos_freqs=10, dir_freqs=4, hidden_size=256).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=1e-4,  # Lower initial learning rate
                                  betas=(0.9, 0.999),
                                  eps=1e-8)
        
        self.batch_size = 1
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_path = os.path.join(workspace_root, 'nerf_synthetic', 'chair')
        data_path = os.path.join(base_path, 'train')
        transforms_path = os.path.join(base_path, 'transforms_train.json')
        
        self.dataloader = CustomDataloader(self.batch_size, data_path, transforms_path)
        self.epochs = 200

        self.mse_loss = nn.MSELoss()
        

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.running_loss = []
        self.window_size = 100

    def train(self):
        try:
            torch.cuda.empty_cache()
            
            for epoch in range(self.start_epoch, self.epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for i, data in enumerate(self.dataloader):
                    try:
                        points = data['points'].to(self.device)
                        rays_d = data['rays_d'].to(self.device)
                        z_vals = data['z_vals'].to(self.device)
                        rgb_gt = data['rgb_gt'].to(self.device)
                        
                        batch_size, n_rays, n_samples = points.shape[0], points.shape[1], points.shape[2]
                        chunk_size = 2048  # Reduced chunk size
                        rgb_pred = torch.zeros((batch_size, n_rays, 3), device=self.device)
                        
                        self.optimizer.zero_grad()
                        
                        for chunk_start in range(0, n_rays, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, n_rays)
                            

                            points_chunk = points[:, chunk_start:chunk_end].reshape(-1, n_samples, 3)
                            rays_d_chunk = rays_d[:, chunk_start:chunk_end].reshape(-1, 3)
                            z_vals_chunk = z_vals[:, chunk_start:chunk_end].reshape(-1, n_samples)
                            

                            rays_d_norm = F.normalize(rays_d_chunk, dim=-1)
 
                            points_flat = points_chunk.reshape(-1, 3)
                            dirs_flat = rays_d_norm.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3)
                            
                            outputs = self.model(points_flat, dirs_flat)
                            outputs = outputs.reshape(-1, n_samples, 4)
                            
                            rgb = outputs[..., :3]
                            sigma = outputs[..., 3]
                            
                            dists = z_vals_chunk[..., 1:] - z_vals_chunk[..., :-1]
                            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)
                            
                            alpha = 1 - torch.exp(-F.relu(sigma) * dists)
                            T = torch.cumprod(torch.cat([
                                torch.ones_like(alpha[..., :1]),
                                (1 - alpha + 1e-10)[..., :-1]
                            ], dim=-1), dim=-1)
                            
                            weights = alpha * T
                            rgb_chunk = (weights.unsqueeze(-1) * rgb).sum(dim=1)

                            rgb_pred[:, chunk_start:chunk_end] = rgb_chunk.reshape(batch_size, -1, 3)

                        loss = self.mse_loss(rgb_pred, rgb_gt)
                        loss.backward()
  
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                        
                        self.optimizer.step()

                        epoch_loss += loss.item()
                        batch_count += 1
                        
                        self.running_loss.append(loss.item())
                        if len(self.running_loss) > self.window_size:
                            self.running_loss.pop(0)
                        
                        if i % 10 == 0:
                            avg_loss = sum(self.running_loss) / len(self.running_loss)
                            print(f'Epoch [{epoch}/{self.epochs}], Step [{i}], Loss: {avg_loss:.6f}')
                        
                        # Clear memory
                        del points, rays_d, z_vals, rgb_gt, rgb_pred, outputs
                        torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error in batch {i}: {str(e)}")
                        continue

                self.scheduler.step()
                
                avg_loss = epoch_loss / batch_count

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_loss,
                    }, 'best_checkpoint.pth')

                if (epoch + 1) % 50 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_loss,
                    }, f'checkpoint_epoch_{epoch+1}.pth')
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            
if __name__ == "__main__":
    train_model = TrainModel()
    train_model.train()