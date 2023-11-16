# Autonomous_Navigation
A repository to hold my Master's Project on autnomous navigation with GA3C and multiple parallel instances of turtlebot3

# Setup

There are two options for the base OS: Windows using WSL2 or Ubuntu 22.04

  ## WSL2 ##
* Setup WSL2: https://learn.microsoft.com/en-us/windows/wsl/install (basic instructions included below)

  `wsl --install `

* Setup CUDA for WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/index.html (basic instructions included below)

    *Get GPU driver for Windows: https://www.nvidia.com/download/index.aspx

    *Get WSL-Ubuntu CUDA toolkit: [WSL_CUDA_TOOLKIT](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) (basic instructions below)
  
      wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  
      sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  
      wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.1-1_amd64.deb
  
      sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.1-1_amd64.deb
  
      sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
  
      sudo apt-get update
  
      sudo apt-get -y install cuda-toolkit-12-3
  
* Setup Tensorflow 2.14.0: https://www.tensorflow.org/install/pip#windows-wsl2 (basic instructions included below)
  
    `pip install --upgrade pip`
  
    `pip install tensorflow[and-cuda]`

    Verify the install with: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
  
## Ubuntu ##
