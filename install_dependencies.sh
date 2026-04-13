#!/bin/bash

# Create venv if it doesn't exist
if [ ! -d "blender_venv" ]; then
    echo "Creating virtual environment 'blender_venv'..."
    python3 -m venv blender_venv
fi

# Activate venv
source blender_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Waymo and basic dependencies
echo "Installing Waymo and basic dependencies..."
pip install tensorflow==2.16.1 protobuf==3.20.3 matplotlib opencv-python numpy==1.26.4 open3d scikit-image pyarrow dask[dataframe] dacite immutabledict 
pip install waymo-open-dataset-tf-2-12-0==1.6.7 --no-deps

# Install Mask2Former dependencies
echo "Installing Mask2Former dependencies (Transformers, Torch, JAX)..."
# JAX 0.4.29 is compatible with NumPy 1.26.4
pip install transformers torch torchvision timm "jax<0.4.30" "jaxlib<0.4.30"
pip install huggingface-hub # For checkpoint download

# Clone and Install DepthPro
if [ ! -d "ml-depth-pro" ]; then
    echo "Cloning Apple DepthPro..."
    git clone https://github.com/apple/ml-depth-pro.git
fi

echo "Installing DepthPro..."
cd ml-depth-pro
pip install -e .
cd ..

# Download DepthPro checkpoint if not exists
if [ ! -f "ml-depth-pro/checkpoints/depth_pro.pt" ]; then
    echo "Downloading DepthPro checkpoint via Hugging Face Hub..."
    mkdir -p ml-depth-pro/checkpoints
    # Using hf (the system tool) as it's more reliable in this environment
    hf download apple/DepthPro depth_pro.pt --local-dir ml-depth-pro/checkpoints
fi

echo "Installation complete."
