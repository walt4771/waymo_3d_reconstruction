sudo apt update

sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz;tar -xf Python-3.12.0.tgz;cd Python-3.12.0
./configure --enable-optimizations
sudo make install
sudo rm -rf Python-3.12.0.tgz
sudo rm -rf Python-3.12.0

sudo apt install -y wget libxml2 libx11-6 libgl1 git
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run --silent --toolkit --override
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
nvcc --version

mkdir StudioProjects;cd StudioProjects;mkdir test;cd test;
python3 -m venv test
