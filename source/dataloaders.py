import os 
import json 
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LoadSyntheticDataset(Dataset): 
    def __init__(self, path_to_images, path_to_labels): 
        print(f"Initializing dataset with images path: {path_to_images}")
        print(f"Labels path: {path_to_labels}")
        
        if not os.path.exists(path_to_images):
            raise FileNotFoundError(f"Images directory not found: {path_to_images}")
        
        if not os.path.exists(path_to_labels):
            raise FileNotFoundError(f"Labels file not found: {path_to_labels}")
            
        self.path_to_images = path_to_images
        print(f"Contents of {path_to_images}:")
        all_files = os.listdir(path_to_images)
        print(f"All files: {all_files}")
        self.images = [im for im in all_files if im.endswith('.png')]
        print(f"Found {len(self.images)} PNG images: {self.images}")
        
        self.transform = transforms.ToTensor()
        
        try:
            with open(path_to_labels, 'r') as f: 
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels['frames'])} frames from labels file")
        except Exception as e:
            print(f"Error loading labels file: {str(e)}")
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
            # Get just the filename from the file_path, removing any directory components
            file_name = os.path.basename(label['file_path']) + '.png'
            img_path = os.path.join(self.path_to_images, file_name)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            image = Image.open(img_path).convert("RGB")

            if self.transform: 
                image = self.transform(image)
            print(f'Processing image {idx}: shape {image.shape}')

            N_rays = 1024
            H, W = image.shape[1], image.shape[2]
            origins, directions = self.get_origins_and_directions(label, W, H)
            origins, directions = self.sample_random_rays(origins, directions, N_rays)

            points, z_vals = self.get_rays_sampling(origins, directions, 2, 6, 64)
            image = image.reshape(H * W, 3)
            _, indices = torch.sort(torch.randperm(H*W)[:N_rays])  # reuse same indices
            rgb_gt = image[indices]  # [N_rays, 3]

            return {
                'points': points,
                'rays_d': directions,
                'rgb_gt': rgb_gt,
                'z_vals': z_vals.T.expand(origins.shape[0], -1)  
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

class CustomDataloader: 
    def __init__(self, batch_size, path_to_images=None, path_to_labels=None): 
        print(f"Initializing CustomDataloader with batch_size={batch_size}")
        print(f"path_to_images={path_to_images}")
        print(f"path_to_labels={path_to_labels}")
        
        if path_to_images is None or path_to_labels is None:
            raise ValueError("Both path_to_images and path_to_labels must be provided")
            
        self.dataset = LoadSyntheticDataset(
                path_to_images=path_to_images, 
                path_to_labels=path_to_labels  
            )
        self.batch_size = batch_size
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print(f"Created DataLoader with {len(self.loader)} batches")
    
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
