import os
import sys

# Add the parent directory to the path so imports work correctly
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
        self.epochs = 2
        
        self.loss_function = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.start_epoch = 0
        self.checkpoint_path = 'checkpoint.pth'

    #ChatGPT Generated Function
    def volume_rendering(self, outputs, z_vals, dirs):
        
        rgb = outputs[..., :3]  # [N_rays, N_samples, 3]
        sigma = outputs[..., 3]  # [N_rays, N_samples]
        # # print(f"rgb shape: {rgb.shape}")
        # # print(f"sigma shape: {sigma.shape}")

        # Compute distances between adjacent samples
        dists = z_vals[:, 1:] - z_vals[:, :-1]  # [N_rays, N_samples - 1]
        # print(f"dists shape before unsqueeze: {dists.shape}")
        dists = dists.unsqueeze(-1)  # [N_rays, N_samples - 1, 1]
        # print(f"dists shape after unsqueeze: {dists.shape}")
        
        # Create padding with correct size [N_rays, 1, 1]
        padding = 1e10 * torch.ones_like(dists[:, :1, :])  # Using same device and dtype as dists
        # print(f"padding shape: {padding.shape}")
        dists = torch.cat([dists, padding], dim=1)  # [N_rays, N_samples, 1]
        # print(f"dists shape after cat: {dists.shape}")

        # Scale distances by ray direction magnitude
        # Compute ray direction norms with correct broadcasting
        ray_dirs_norm = torch.norm(dirs, dim=-1, keepdim=True)  # [N_rays, 1]
        # print(f"ray_dirs_norm shape before unsqueeze: {ray_dirs_norm.shape}")
        
        # Reshape ray_dirs_norm to match dists shape
        # The issue is that ray_dirs_norm has shape [N_rays, 1] but we need [N_rays, 1, 1]
        # and we need to make sure the N_rays dimension matches
        if ray_dirs_norm.shape[0] != dists.shape[0]:
            # If the number of rays doesn't match, we need to expand ray_dirs_norm
            ray_dirs_norm = ray_dirs_norm[:dists.shape[0]]
        
        ray_dirs_norm = ray_dirs_norm.unsqueeze(1)  # [N_rays, 1, 1]
        # print(f"ray_dirs_norm shape after unsqueeze: {ray_dirs_norm.shape}")
        
        # Make sure ray_dirs_norm has the right shape for broadcasting
        if ray_dirs_norm.shape[0] != dists.shape[0]:
            # If the number of rays still doesn't match, we need to expand ray_dirs_norm
            ray_dirs_norm = ray_dirs_norm.expand(dists.shape[0], 1, 1)
            # print(f"ray_dirs_norm shape after expand: {ray_dirs_norm.shape}")
        
        # The issue is that dists has shape [N_rays, N_samples, 1] but ray_dirs_norm has shape [N_rays, 1, 1]
        # We need to make sure the N_samples dimension is compatible
        if dists.shape[1] != ray_dirs_norm.shape[1]:
            # If the N_samples dimension doesn't match, we need to expand ray_dirs_norm
            ray_dirs_norm = ray_dirs_norm.expand(-1, dists.shape[1], -1)
            # print(f"ray_dirs_norm shape after final expand: {ray_dirs_norm.shape}")
        
        dists = dists * ray_dirs_norm  # [N_rays, N_samples, 1]
        # print(f"dists shape after multiplication: {dists.shape}")

        # Compute alpha values from densities
        alpha = 1.0 - torch.exp(-sigma.unsqueeze(-1) * dists)  # [N_rays, N_samples, 1]
        # print(f"alpha shape: {alpha.shape}")

        # Compute transmittance
        ones = torch.ones((alpha.shape[0], 1, 1), device=alpha.device)
        # print(f"ones shape: {ones.shape}")
        alpha_cat = torch.cat([ones, 1. - alpha + 1e-10], dim=1)
        # print(f"alpha_cat shape: {alpha_cat.shape}")
        accum_prod = torch.cumprod(alpha_cat, dim=1)
        # print(f"accum_prod shape: {accum_prod.shape}")
        T = accum_prod[:, :-1, :]  # [N_rays, N_samples, 1]
        # print(f"T shape: {T.shape}")

        # Compute final weights
        weights = T * alpha  # [N_rays, N_samples, 1]
        # print(f"weights shape: {weights.shape}")

        # Weighted sum of colors
        rgb_final = torch.sum(weights * rgb, dim=1)  # [N_rays, 3]
        # print(f"rgb_final shape: {rgb_final.shape}")

        return rgb_final

    def train(self):
        try:
            torch.cuda.empty_cache()
            for epoch in range(self.start_epoch, self.epochs):
                for i, data in enumerate(self.dataloader):
                    try:
                        for key, value in data.items():
                            if isinstance(value, torch.Tensor):
                                print(f"  {key}: {value.shape}")
                        
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
                        rgb_pred = self.volume_rendering(outputs, z_vals, rays_d)
                        loss = self.loss_function(rgb_pred, rgb_gt)
                        loss.backward()
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