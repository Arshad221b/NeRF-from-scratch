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
        focal = W / 2
        
    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, focal, c2w, device=device)
        
        # Flatten rays for processing
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        # Process rays in chunks to save memory
        chunk_size = 4096
        rgb_final = torch.zeros((H*W, 3), device=device)
        
        for chunk_start in range(0, rays_o.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, rays_o.shape[0])
            
            rays_o_chunk = rays_o[chunk_start:chunk_end]
            rays_d_chunk = rays_d[chunk_start:chunk_end]
            rays_d_chunk = F.normalize(rays_d_chunk, p=2, dim=-1)
            
            # Sampling strategy from training
            near, far = 2.0, 6.0
            t_vals = torch.linspace(0., 1., steps=64, device=device)
            z_vals = near * (1. - t_vals) + far * t_vals
            z_vals = z_vals.expand(rays_o_chunk.shape[0], -1)
            
            # Add stratified sampling
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
            
            # Get sample points
            points = rays_o_chunk[:, None, :] + rays_d_chunk[:, None, :] * z_vals[..., :, None]
            view_dirs = rays_d_chunk[:, None, :].expand_as(points)
            
            # Flatten points and directions
            points_flat = points.reshape(-1, 3)
            dirs_flat = view_dirs.reshape(-1, 3)
            
            # Process in smaller sub-chunks
            sub_chunk_size = 8192
            outputs_chunks = []
            
            for i in range(0, points_flat.shape[0], sub_chunk_size):
                points_sub = points_flat[i:i+sub_chunk_size]
                dirs_sub = dirs_flat[i:i+sub_chunk_size]
                outputs_sub = model(points_sub, dirs_sub)
                outputs_chunks.append(outputs_sub)
                torch.cuda.empty_cache()
            
            outputs = torch.cat(outputs_chunks, 0)
            outputs = outputs.reshape(points.shape[0], points.shape[1], 4)
            
            # Split outputs
            rgb = outputs[..., :3]  # [chunk_size, N_samples, 3]
            sigma = outputs[..., 3]  # [chunk_size, N_samples]
            
            # Use same density scaling as training
            sigma = sigma * 0.1
            
            # Compute distances
            dists = z_vals[..., 1:] - z_vals[..., :-1]  # [chunk_size, N_samples-1]
            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)  # [chunk_size, N_samples]
            
            # Compute alpha compositing
            alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)  # [chunk_size, N_samples]
            
            # Compute weights
            T = torch.ones_like(alpha[:, :1])  # [chunk_size, 1]
            weights = []
            
            for i in range(alpha.shape[1]):
                weights.append(T * alpha[:, i:i+1])
                T = T * (1.0 - alpha[:, i:i+1] + 1e-10)
            
            weights = torch.cat(weights, dim=1)  # [chunk_size, N_samples]
            
            # Compute final RGB
            rgb_chunk = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # [chunk_size, 3]
            rgb_final[chunk_start:chunk_end] = rgb_chunk
            
            # Clear memory
            del points, view_dirs, outputs, rgb, sigma, weights
            torch.cuda.empty_cache()
        
        # Reshape and post-process
        rgb_final = rgb_final.reshape(H, W, 3)
        rgb_final = torch.clamp(rgb_final, 0.0, 1.0)
        
        return rgb_final

def create_360_degree_poses(num_frames=120, radius=4.0, h=0.5):
    """Create camera poses for a 360-degree rotation around the object."""
    poses = []
    for th in np.linspace(0., 360., num_frames, endpoint=False):
        theta = np.deg2rad(th)
        
        # Spiral path
        phi = np.deg2rad(30.0)  # Tilt angle
        
        # Camera position
        x = radius * np.cos(theta) * np.cos(phi)
        y = h + radius * np.sin(phi)  # Slight elevation
        z = radius * np.sin(theta) * np.cos(phi)
        
        # Look-at point (slightly above origin)
        target = np.array([0, 0.2, 0])  # Look slightly above center
        eye = np.array([x, y, z])
        up = np.array([0, 1, 0])
        
        # Create camera-to-world matrix
        c2w = look_at(eye, target, up)
        c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
        
        poses.append({'transform_matrix': c2w})
    return poses

def look_at(eye, target, up):
    """Create a look-at matrix."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    rot = np.stack([right, up, -forward], axis=1)
    trans = eye
    
    return np.column_stack([rot, trans])

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

def load_test_poses(transforms_path):
    """Load test poses from transforms.json file."""
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    frames = []
    camera_angle_x = transforms.get('camera_angle_x', None)
    
    for frame in transforms.get('frames', []):
        frames.append({
            'transform_matrix': np.array(frame['transform_matrix'], dtype=np.float32),
            'file_path': frame.get('file_path', None)
        })
    
    return frames, camera_angle_x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = NeRFModel(pos_freqs=10, dir_freqs=4).to(device)
    
    # Load checkpoint
    checkpoint_path = '/teamspace/studios/this_studio/checkpoint_epoch_200.pth'
    model = load_checkpoint(model, checkpoint_path)
    
    # Create output directory
    render_dir = 'rendered_views'
    os.makedirs(render_dir, exist_ok=True)
    
    # Load test transforms
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    transforms_path = os.path.join(workspace_root, 'nerf_synthetic', 'chair', 'transforms_test.json')
    frames, camera_angle_x = load_test_poses(transforms_path)
    
    # Calculate focal length from camera angle
    H = W = 400  # Match training resolution
    focal = W / (2 * np.tan(camera_angle_x / 2))
    
    print(f"Rendering {len(frames)} test views...")
    for idx, frame in enumerate(frames):
        print(f"Rendering view {idx + 1}/{len(frames)}")
        
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)
        
        # Render using the correct focal length
        rgb_map = render_novel_view(model, c2w, H=H, W=W, focal=focal, device=device)
        
        rgb_map = rgb_map.cpu()
        output_path = os.path.join(render_dir, f'view_{idx:03d}.png')
        save_rendered_image(rgb_map, rgb_map.shape[1], rgb_map.shape[0], output_path)
    
    print("Creating GIF from rendered views...")
    create_gif(render_dir, 'nerf_test_views.gif', duration=0.1)
    print("Done! Check nerf_test_views.gif for the final animation.")

if __name__ == '__main__':
    main() 