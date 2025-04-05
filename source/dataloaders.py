import os 
import json 
import torch
import torch.nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LoadSyntheticDataset(Dataset): 
    def __init__(self, path_to_images, path_to_labels): 
        self.path_to_images = path_to_images
        self.images = [im for im in os.listdir(path_to_images)]
        self.transform = transforms.ToTensor()
        with open(path_to_labels, 'r') as f: 
            self.labels = json.load(f)

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

    def get_rays_sampling(self, origins, directions, near, far, samples):
        z_vals = torch.linspace(near, far, steps=samples)  # [samples]
        z_vals = z_vals[None, :, None]  # [1, samples, 1]

        origins = origins[:, None, :]     # [N_rays, 1, 3]
        directions = directions[:, None, :]  # [N_rays, 1, 3]

        points = origins + directions * z_vals  # [N_rays, samples, 3]

        return points

    def __len__(self): 
        return len(self.images)

    def __getitem__ (self, idx): 
        img_path = os.path.join(self.path_to_images, self.images[idx])
        image = Image.open(img_path)
        label = self.labels['frames'][idx]

        if self.transform: 
            image = self.transform(image)
        origins, directions = self.get_origins_and_directions(label, 800, 800)
        points = self.get_rays_sampling(origins, directions, 2, 6, 64)
        return points


class CustomDataloader: 
    def __init__(self, batch_size, path_to_images=None, path_to_labels=None): 
        self.dataset =  LoadSyntheticDataset(
                path_to_images= path_to_images, 
                path_to_labels= path_to_labels  
            )
        self.batch_size = batch_size
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def __iter__(self):
        return iter(self.loader)
    def __len__(self):
        return len(self.loader)

# dataset = LoadSyntheticDataset(
#     path_to_images= 'nerf_synthetic/chair/train', 
#     path_to_labels= '/teamspace/studios/this_studio/nerf_synthetic/chair/transforms_train.json'
# )


# loader = DataLoader(dataset, batch_size = 4, shuffle= True)



# # for points in loader: 
#     print(points.shape)
#     break
    # print(labels)



