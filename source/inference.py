### ENTIRELY AI GENERATED CODE ###

import os
import torch
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from model import NeRFModel
from rederer import save_rendered_image
import torch.nn.functional as F

def load_checkpoint(model, checkpoint_path):
    """Load the latest checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
    else:
        raise ValueError("Checkpoint format not recognized")
    
    model.load_state_dict(state_dict)
    print("Successfully loaded checkpoint from:", checkpoint_path)
    return model

def get_rays(H, W, focal, c2w, device='cuda'):
    """Generate rays for the given camera parameters."""
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    i = i.float()
    j = j.float()
    
    x = (i - W * 0.5) / focal
    y = (j - H * 0.5) / focal
    z = -torch.ones_like(x)
    dirs = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
    
    # Rotate ray directions
    rays_d = (dirs @ c2w[:3, :3].T)  # [H, W, 3]
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # [H, W, 3]
    
    return rays_o, rays_d

def render_novel_view(model, c2w, H=400, W=400, focal=None, device='cuda'):
    """Render a novel view from the given camera pose."""
    model.eval()
    if focal is None:
        focal = W / 2  # Default focal length
        
    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, focal, c2w, device=device)  # [H, W, 3]
        
        # Flatten rays for processing
        rays_o = rays_o.reshape(-1, 3)  # [H*W, 3]
        rays_d = rays_d.reshape(-1, 3)  # [H*W, 3]
        
        # Sample points along rays
        near, far = 2.0, 6.0
        t_vals = torch.linspace(0., 1., steps=64, device=device)
        z_vals = near * (1. - t_vals) + far * t_vals  # [64]
        z_vals = z_vals.expand(rays_o.shape[0], -1)  # [H*W, 64]
        
        # Get sample points
        points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [H*W, 64, 3]
        points_flat = points.reshape(-1, 3)  # [H*W*64, 3]
        
        # Process in chunks to avoid OOM
        chunk_size = 32768
        outputs = []
        
        for i in range(0, points_flat.shape[0], chunk_size):
            points_chunk = points_flat[i:i+chunk_size]
            # The model will handle positional encoding internally
            output_chunk = model(points_chunk)  # [chunk_size, 5]
            outputs.append(output_chunk)
            
        outputs = torch.cat(outputs, 0)  # [H*W*64, 5]
        
        # Split into RGB and sigma
        rgb_map = torch.sigmoid(outputs[..., :3])  # [H*W*64, 3]
        sigma_map = outputs[..., 3:4]  # [H*W*64, 1]
        
        # Reshape back to [H*W, 64, 3] and [H*W, 64, 1]
        num_rays = H * W
        rgb_map = rgb_map.reshape(num_rays, 64, 3)  # [H*W, 64, 3]
        sigma_map = sigma_map.reshape(num_rays, 64, 1)  # [H*W, 64, 1]
        
        # Apply volume rendering equation
        delta_dists = z_vals[:, 1:] - z_vals[:, :-1]  # [H*W, 63]
        delta_dists = torch.cat([delta_dists, torch.ones_like(delta_dists[:, :1]) * 1e10], dim=-1)  # [H*W, 64]
        delta_dists = delta_dists.unsqueeze(-1)  # [H*W, 64, 1]
        
        # Alpha compositing
        alpha = 1.0 - torch.exp(-F.relu(sigma_map) * delta_dists)  # [H*W, 64, 1]
        T = torch.cumprod(
            torch.cat([torch.ones((num_rays, 1, 1), device=device), 1.0 - alpha + 1e-10], dim=1)[:, :-1, :],
            dim=1)  # [H*W, 64, 1]
        weights = alpha * T  # [H*W, 64, 1]
        
        # Compute final color
        rgb_final = (weights * rgb_map).sum(dim=1)  # [H*W, 3]
        
        # Reshape to image dimensions
        rgb_final = rgb_final.reshape(H, W, 3)
        
        return rgb_final

def create_360_degree_poses(num_frames=120):
    """Create camera poses for a 360-degree rotation around the object."""
    poses = []
    for th in np.linspace(0., 360., num_frames, endpoint=False):
        # Convert angle to radians
        theta = np.deg2rad(th)
        
        # Create rotation matrix
        c2w = np.array([
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 2],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        poses.append({'transform_matrix': c2w})
    return poses

def create_gif(image_folder, output_path, duration=0.1):
    """Create a GIF from a folder of images."""
    images = []
    # Sort files to ensure correct order
    files = sorted(os.listdir(image_folder))
    for filename in files:
        if filename.endswith('.png'):
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))
    
    # Save as GIF
    imageio.mimsave(output_path, images, duration=duration)
    print(f"GIF saved to {output_path}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = NeRFModel(num_freqs=10).to(device)
    
    # Load checkpoint
    checkpoint_path = 'checkpoint.pth'
    model = load_checkpoint(model, checkpoint_path)
    
    # Create output directory
    render_dir = 'rendered_views'
    os.makedirs(render_dir, exist_ok=True)
    
    # Generate 360-degree camera poses
    frames = create_360_degree_poses(num_frames=120)  # 120 frames for smooth rotation
    
    print("Rendering 360-degree views...")
    # Render novel views
    for idx, frame in enumerate(frames):
        print(f"Rendering view {idx + 1}/{len(frames)}")
        
        # Get camera-to-world transform and move to device
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)
        
        # Render
        rgb_map = render_novel_view(model, c2w, device=device)
        
        # Move to CPU for saving
        rgb_map = rgb_map.cpu()
        
        # Save rendered image
        output_path = os.path.join(render_dir, f'view_{idx:03d}.png')
        save_rendered_image(rgb_map, rgb_map.shape[1], rgb_map.shape[0], output_path)
    
    # Create GIF from rendered views
    print("Creating GIF from rendered views...")
    create_gif(render_dir, 'nerf_360_rotation.gif', duration=0.05)  # 0.05s per frame = 20fps
    print("Done! Check nerf_360_rotation.gif for the final animation.")

if __name__ == '__main__':
    main() 