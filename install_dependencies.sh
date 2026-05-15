#!/bin/bash

sudo apt install libfontconfig1 libegl1 libgl1 libxkbcommon0 libxcb-cursor0 libxkbcommon-x11-0 libwayland-client0 libwayland-cursor0

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


pip install PyQt6
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121



# Clone and Install GaussianSplatting

# 1. 시스템 필수 도구 설치
# 가장 먼저 컴파일러와 빌드 가속기(ninja)를 설치합니다.
sudo apt install gcc-12 g++-12 ninja-build git -y

# 2. CUDA 헤더 패치 (한 번만 수행)
# Ubuntu 최신 버전과 CUDA 12.1 간의 충돌을 방지하기 위해 에러가 났던 라인들을 주석 처리합니다.
sudo sed -i '5439s/^/\/\//' /usr/local/cuda/include/crt/math_functions.h
sudo sed -i '5499s/^/\/\//' /usr/local/cuda/include/crt/math_functions.h
sudo sed -i '5551s/^/\/\//' /usr/local/cuda/include/crt/math_functions.h
sudo sed -i '5603s/^/\/\//' /usr/local/cuda/include/crt/math_functions.h

# 5. 컴파일러 환경 변수 설정
# 현재 터미널 세션이 gcc-12를 사용하도록 강제합니다.
export CC=gcc-12
export CXX=g++-12

if [ ! -d "gaussian-splatting" ]; then
    echo "Cloning gaussian-splatting..."
    git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
fi
echo "Installing Gaussian Splatting..."
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim joblib plyfile --no-build-isolation









# Install Mask2Former dependencies
echo "Installing Mask2Former dependencies (Transformers, Torch, JAX)..."
# JAX 0.4.29 is compatible with NumPy 1.26.4
pip install transformers timm "jax<0.4.30" "jaxlib<0.4.30"
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
