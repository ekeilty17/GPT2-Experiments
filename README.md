# GPT2 Experiments

This repository is used to send code to a cloud server in order to do very labor intensive experiments in generating reflections using GPT2

## To Run

go into `main.py` and comment/uncomment the experiment you would like to run. Then in the commandline you can specify which gpt2 model you would like to run. For example
```
$ python3 main.py -model gpt2-xl
```

## Creating Cloud Server
This task was very tedious and annoying, so I have outlined exactly what I did so it can be easily replicated by someone in the future (possibly myself) to save humanity from the time I lost figuring it out.

### Creating Google Cloud Server

Go to create a new VM instance on the google cloud console. I will list everything you should change. If not mentioned it means don't do anything.

- Name: whatever you would like
- Region: `us-west1 (Oregon)` (this region tends to have more free GPUs)
- Machine Type: `n1-standard-8` - `8vCPU, 30 GB memory` (need this much RAM for the large GPT2 model)
- CPU platform and GPU
    - Add GPU
- Boot disk
    - Operating System: `Ubuntu`
    - Version: `Ubuntu 16.04 LTS`
    - Size (GB): `50` (neeed this much persistent memory because we need to save very large models)
- Access scopes: `Allow full access to all Cloud APIs`
- Firewall: `Allow HTTP traffic` and `Allow HTTPS traffic`


### Configuring Ubuntu server

Initialize server
```
sudo apt-get update
sudo apt-get upgrade
```

Install python3 and pip
```
sudo apt install python3-dev python3-pip
```

Download deep learning packages and other needed python packages
```
pip3 install --upgrade pip
pip3 install --upgrade tensorflow-gpu
pip3 install --upgrade torch
pip3 install --upgrade transformers
pip3 install pandas
```

#### Installing Drivers

To use the GPU, we need to install/update our drivers. In particular, NVIDIA, CUDA, and cuDNN. This needs to be done in a VERY specific order because of dependency issues

First, we need an NVIDIA driver
```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install nvidia-driver-450
```

Check to see everything worked
```
sudo reboot now
sudo apt install nvidia-cuda-toolkit
nvidia-smi
nvcc --version
```

That last command probably shows we have CUDA 9.1. Tensorflow 2.x requires at least CUDA 10.1, so we need to update our CUDA
```
# Removing NVIDIA
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*

# Setup the correct CUDA PPA on your system
sudo apt update
sudo add-apt-repository ppa:graphics-drivers
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

# Install CUDA 10.1 packages
sudo apt update
sudo apt install cuda-10-1
sudo apt install libcudnn7
```

We need to specify PATH to CUDA in `.profile` file
```
sudo vi ~/.profile
```
Add this to the bottom of that file
```
# set PATH for cuda 10.1 installation
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

Check to see everything worked
```
sudo reboot now
nvidia-smi
nvcc --version
```

Check that PyTorch and Tensorflow can access te GPUs with the test script in the repo
```
$ python3 gpu_test.py
```