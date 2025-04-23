from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F


# ChatGPT Generated Code
def save_rendered_image(rendered_colors, image_width, image_height, output_path):
    """
    Save the rendered image to a file.

    Args:
        rendered_colors (torch.Tensor): Rendered colors (N_rays, 3).
        image_width (int): Width of the output image.
        image_height (int): Height of the output image.
        output_path (str): Path to save the output image.
    """
    # Reshape the rendered colors to match the image dimensions
    image = rendered_colors.reshape(image_height, image_width, 3).cpu().numpy()
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit color

    # Save the image using Pillow
    img = Image.fromarray(image)
    img.save(output_path)
    print(f"Rendered image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Dummy NeRF model
    class DummyNeRF(torch.nn.Module):
        def forward(self, x):
            return torch.cat([torch.sigmoid(x[:, :3]), torch.ones(x.shape[0], 1, device=x.device)], dim=-1)

    # Initialize model and inputs
    model = DummyNeRF().to("cuda")
    image_width, image_height = 32, 32
    ray_origins = torch.rand(image_width * image_height, 3, device="cuda")
    ray_directions = torch.rand(image_width * image_height, 3, device="cuda").normalize(dim=-1)
    near, far = 0.1, 4.0
    num_samples = 64

    # Render rays
    rendered_image = render_rays(ray_origins, ray_directions, model, near, far, num_samples, device="cuda")

    # Save the rendered image
    save_rendered_image(rendered_image, image_width, image_height, "output.png")
