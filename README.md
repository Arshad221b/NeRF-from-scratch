# NeRF from Scratch

A PyTorch implementation of Neural Radiance Fields (NeRF) built from scratch. This implementation focuses on learning and understanding the core concepts of NeRF while maintaining good performance and code readability.

## Overview

Neural Radiance Fields (NeRF) represent scenes as continuous 5D functions that output the radiance emitted in each direction (θ, φ) at each point (x, y, z) in space. This implementation includes:

- Custom NeRF model with positional encoding
- Volume rendering pipeline
- Training on synthetic datasets
- Inference with novel view synthesis

## Requirements

```bash
torch>=1.8.0
numpy>=1.19.2
Pillow>=8.0.0
tqdm>=4.50.0
```

## Project Structure

```
NeRF-from-scratch/
├── source/
│   ├── model.py         # NeRF model architecture
│   ├── dataloaders.py   # Data loading and processing
│   ├── train.py         # Training pipeline
│   └── inference.py     # Novel view synthesis
├── nerf_synthetic/      # Dataset directory
└── README.md
```

## Training

The training pipeline includes:
- Positional encoding for coordinates and view directions
- Hierarchical sampling along rays
- Efficient batch processing with chunked rays
- Learning rate scheduling
- Gradient clipping for stability
- Automatic checkpointing

To train the model:

```bash
python source/train.py
```

Training parameters:
- Learning rate: 1e-4 with exponential decay (gamma=0.995)
- Batch size: 1 (processes multiple rays per batch)
- Number of epochs: 200
- Positional encoding frequencies: 10 (positions), 4 (directions)
- Network size: 256 hidden units

## Inference

For rendering novel views:

```bash
python source/inference.py
```

The inference script:
- Loads trained model weights
- Generates novel camera trajectories
- Renders images from new viewpoints
- Creates a GIF of the rendered views

## Dataset

This implementation uses the synthetic NeRF dataset (specifically the 'chair' scene). The dataset should be organized as:

```
nerf_synthetic/
└── chair/
    ├── train/
    │   └── *.png
    ├── test/
    │   └── *.png
    ├── transforms_train.json
    └── transforms_test.json
```

## Implementation Details

### Model Architecture
- MLP-based architecture with skip connections
- Separate branches for density and color prediction
- Positional encoding for better high-frequency detail

### Volume Rendering
- Ray sampling with stratified sampling
- Efficient chunk-based processing
- Alpha compositing for final color computation

### Training Features
- Memory-efficient batch processing
- Automatic mixed precision for faster training
- Gradient clipping for training stability
- Regular checkpointing for experiment tracking

## Tips for Best Results

1. **Memory Management**:
   - Adjust chunk size based on your GPU memory
   - Clear unused tensors during training
   - Use gradient checkpointing for larger scenes

2. **Training Stability**:
   - Start with a lower learning rate (1e-4)
   - Use gradient clipping (max_norm=0.1)
   - Monitor the running average loss

3. **Quality Improvements**:
   - Increase sampling points for better quality
   - Adjust near/far bounds based on scene
   - Fine-tune positional encoding frequencies

## Acknowledgments

This implementation is based on the original NeRF paper:
"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
ECCV 2020

## License

MIT License