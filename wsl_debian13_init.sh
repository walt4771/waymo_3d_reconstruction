sudo apt update
sudo apt install -y wget libxml2 build-essential python3 python3-pip python3-venv git
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run --silent --toolkit --override
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
nvcc --version

mkdir StudioProjects;cd StudioProjects;mkdir test;cd test;
python3 -m venv test
