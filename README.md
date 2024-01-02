# Autonomous_Navigation
A repository to hold my Master's Project on autonomous navigation with GA3C and multiple parallel instances of turtlebot3

# Setup

There are two options for the base OS: Windows using WSL2 or Ubuntu 22.04

* ## Setup Tensorflow and CUDA ##

  <details>
  <summary>WSL2</summary>
      <br>
      
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
 
      Check for that nvidia drivers are working: `nvidia-smi`
    
      `pip install --upgrade pip`
    
      `pip install tensorflow[and-cuda]`
  
      Verify the install with: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
  
    </details>

    <details>
      <summary>Ubuntu</summary>
      <br>
 
      * Get GPU drivers for Ubuntu
      * Setup Tensorflow 2.14.0: https://www.tensorflow.org/install/pip#linux (basic instructions included below)
 
        Check that nvidia drivers are working: `nvidia-smi`
 
        `pip install --upgrade pip`
 
        `pip install tensorflow[and-cuda]`
 
        Verify the install with: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

        If there are issues with initializing inference during training. Install the cuda toolkit: `sudo apt install nvidia-cuda-toolkit`
      
    </details>
    
  * ## Install ROS2 Humble ##
  
    * General Install Instructions: https://docs.ros.org/en/humble/Installation.html
    
    * Instructions for Ubuntu 22.04 (Also works with WSL2 running Ubuntu 22.04): https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html (basic instructions included below)
    
    <details>
      <summary>Install ROS2 Humble for Ubuntu/WLS2(Ubuntu)</summary>
      <br>
  
      ```
      locale  # check for UTF-8
  
      sudo apt update && sudo apt install locales
      sudo locale-gen en_US en_US.UTF-8
      sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
      export LANG=en_US.UTF-8
      
      locale  # verify settings
      
      sudo apt install software-properties-common
      sudo add-apt-repository universe
      sudo apt update && sudo apt install curl -y
      sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
      sudo apt update
      sudo apt upgrade
      sudo apt install ros-humble-desktop
      sudo apt install ros-dev-tools
      source /opt/ros/humble/setup.bash # Replace ".bash" with your shell if you're not using bash. Possible values are: setup.bash, setup.sh, setup.zsh
      
      #Add source to shell startup script: 
      echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
      #Install Gazebo: 
      sudo apt install ros-humble-gazebo-*
      ```
  
    </details>

  * ## Clone this repository ##
  * ## Install various dependencies ##
      * Gym:
            `pip install gym`
      * Gym-gazebo2:
  
        `pip3 install transforms3d billiard psutil`
   
        change to Gym-gazebo2 directory `cd gym-gazebo2/`
   
        `pip3 install -e .`
      * Build Turtlebot3:
        
        change to Turtlebot3_ws directory `cd turtlebot3_ws/`
        
        `colcon build --symlink-install`
        
        Add source to shell startup script:
        
        `echo "source ~/Autonomous_Navigation/turtlebot3_ws/install/setup.bash" >> ~/.bashrc`
        
        `echo “export TURTLEBOT3_MODEL=waffle_depth” >> ~/.bashrc`

# Usage

The interaction is done through the shell scripts in GA3C/ga3c directory.

The following instructions are from the GA3C readme

## How to Train a model from Scratch? ##

Run `sh _clean.sh` first, and then `sh _train.sh`. The script `_clean.sh` cleans the checkpoints folder, which contains the network models saved during the training process, as well as removing `results.txt`, which is a log of scores achieved during training.

`sh _train.sh` launches the training procedure, following the parameters in `Config.py`. You can modify the training parameters directly in `Config.py`, or pass them as arguements to `_train.sh`. E.g., launching `sh _train.sh AGENTS=10 TRAINERS=2 PREDICTORS=2` overwrites the starting values of the number of agents, trainers, and predictors in `Config.py` with the ones passed as arguments.

To stop the training procedure, adjust `EPISODES` in `Config.py` properly, or simply use ctrl + c.

## How to continue training a model? ##

If you want to continue training a model, set `LOAD_CHECKPOINTS=True` in `Config.py`, and set `LOAD_EPISODE` to the episode number you want to load. Be sure that the corresponding model has been saved in the checkpoints folder.

E.g. `sh _train.sh LOAD_CHECKPOINTS=True LOAD_EPISODE=300` would load the model checkpoint at episode 300 and continue training.

## How to Play a game with a trained agent? ##

Run `_play.sh LOAD_CHECKPOINT=True LOAD_EPISODE=300` to load the model checkpoint at episode 300 to play with the environment.

## Graph rewards per episode ##

Use `python3 print_results.py` to plot the rewards per episode logged in `results.txt` from the training.
