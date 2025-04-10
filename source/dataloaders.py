import os 
import json 
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class LoadSyntheticDataset(Dataset): 
    def __init__(self, path_to_images, path_to_labels): 
        
        if not os.path.exists(path_to_images):
            raise FileNotFoundError(f"Images directory not found: {path_to_images}")
        
        if not os.path.exists(path_to_labels):
            raise FileNotFoundError(f"Labels file not found: {path_to_labels}")
            
        self.path_to_images = path_to_images
        all_files = os.listdir(path_to_images)
        self.images = [im for im in all_files if im.endswith('.png')]
        
        self.transform = transforms.ToTensor()
        
        try:
            with open(path_to_labels, 'r') as f: 
                self.labels = json.load(f)
            self.camera_angle_x = self.labels.get('camera_angle_x', None)
        except Exception as e:
            raise

    def get_origins_and_directions(self, frame, width, height): 
        origins = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        origins = origins[:3, 3]

        origins = origins.view(1, 3)
        origins = origins.repeat(width * height, 1)

        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            indexing='xy'
            )   
        
        focal = width / 2
        x = (i - width * 0.5) / focal 
        y = (j - height * 0.5) / focal
        z = -torch.ones_like(x) * 1

        directions = torch.stack((x, y, z), dim=-1)
        directions = directions.view(-1, 3)

        return origins, directions

    def sample_random_rays(self, rays_o, rays_d, N_rays):
        total_rays = rays_o.shape[0] 
        indices = torch.randint(0, total_rays, (N_rays,))  

        rays_o_sampled = rays_o[indices]  
        rays_d_sampled = rays_d[indices]  

        return rays_o_sampled, rays_d_sampled

    
    def get_rays_sampling(self, origins, directions, near, far, samples):
        z_vals = torch.linspace(near, far, steps=samples)  # [samples]
        z_vals = z_vals[None, :, None]  # [1, samples, 1]

        origins = origins[:, None, :]     # [N_rays, 1, 3]
        directions = directions[:, None, :]  # [N_rays, 1, 3]

        points = origins + directions * z_vals  # [N_rays, samples, 3]

        return points.float(), z_vals.squeeze(0)  # z_vals: [samples] → used for rendering

    def __len__(self): 
        return len(self.images)

    def __getitem__ (self, idx): 
        try:
            label = self.labels['frames'][idx]
            file_name = os.path.basename(label['file_path']) + '.png'
            img_path = os.path.join(self.path_to_images, file_name)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            image = Image.open(img_path).convert("RGB") 

            if self.transform: 
                image = self.transform(image)

            N_rays = 4096
            H, W = image.shape[1], image.shape[2]
            
            if self.camera_angle_x is not None:
                focal = W / (2 * np.tan(self.camera_angle_x / 2))
            else:
                focal = W / 2 

            i = torch.randint(0, W, (N_rays,))
            j = torch.randint(0, H, (N_rays,))
            
            rgb_gt = image[:, j, i].permute(1, 0)  # [N_rays, 3]

            x = (i.float() - W * 0.5) / focal
            y = (j.float() - H * 0.5) / focal
            z = -torch.ones_like(x)
            dirs = torch.stack([x, y, z], dim=-1)  # [N_rays, 3]

            c2w = torch.tensor(label['transform_matrix'], dtype=torch.float32)  # [4, 4]
            rays_d = (dirs @ c2w[:3, :3].T).float()  # Rotate ray directions
            rays_o = c2w[:3, 3].expand(rays_d.shape)  # [N_rays, 3]

            near, far = 2.0, 6.0
            t_vals = torch.linspace(0., 1., steps=64)
            z_vals = near * (1. - t_vals) + far * t_vals  # [64]
            z_vals = z_vals.expand(N_rays, -1)  # [N_rays, 64]

            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

            points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [N_rays, 64, 3]

            return {
                'points': points,
                'rays_d': rays_d,
                'rgb_gt': rgb_gt,
                'z_vals': z_vals
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

class CustomDataloader: 
    def __init__(self, batch_size, path_to_images=None, path_to_labels=None): 
        
        if path_to_images is None or path_to_labels is None:
            raise ValueError("Both path_to_images and path_to_labels must be provided")
            
        self.dataset = LoadSyntheticDataset(
                path_to_images=path_to_images, 
                path_to_labels=path_to_labels  
            )
        self.batch_size = batch_size
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def __iter__(self):
        return iter(self.loader)
        
    def __len__(self):
        return len(self.loader)

# dataset = LoadSyntheticDataset(
#     path_to_images= '/teamspace/studios/this_studio/nerf_synthetic/chair', 
#     path_to_labels= '/teamspace/studios/this_studio/nerf_synthetic/chair/transforms_train.json'
# )


# loader = DataLoader(dataset, batch_size = 4, shuffle= True)



# for points in loader: 
#     print(points.shape)
#     break
    # print(labels)
