#!/usr/bin/env bash

#========================================================================================
# Copyright (C) 2020, Robotic Systems Lab, ETH Zurich
# All rights reserved.
# http://www.rsl.ethz.ch
# https://bitbucket.org/leggedrobotics/noesis
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#========================================================================================
# Authors: Vassilios Tsounis, tsounsiv@ethz.ch
#========================================================================================

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set repository root path
NOESIS_ROOT="$( realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"/../../ )"
echo "[install nvidia]: Noesis root at: ${NOESIS_ROOT}"

# Default configurations
TENSORRT=false
DOCKER=false

# Iterate over arguments list to configure the installation.
for i in "$@"
do
case $i in
  --tensorrt)
    TENSORRT=true
    shift # past argument with no value
    ;;
  --docker)
    DOCKER=true
    shift # past argument with no value
    ;;
  *)
    echo "[install nvidia]: Error: Unknown arguments: ${i#*=}"
    exit 1
    ;;
esac
done

#==
# APT dependencies
#==

# NOTE: The following instructions were taken from the official TensorFlow [documentation](https://www.tensorflow.org/install/gpu).

# Ubuntu 20.04 package sources
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-ubuntu2004-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update

# Ubuntu 18.04 package sources
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-ubuntu1804-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt update

# Install base NVIDIA driver dependencies
sudo apt install --no-install-recommends nvidia-driver-460

# Install base CUDA dependencies
sudo apt install --no-install-recommends cuda-10-0 cuda-11-0

# Install the NVIDIA machine-learning package repo for 20.04
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt update

# Append package repository for 18.04 cuDNN packages
sudo bash -c "echo 'deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /' >> /etc/apt/sources.list.d/nvidia-machine-learning.list"
sudo apt update

# Install cuDNN dependencies
sudo apt install --no-install-recommends libcudnn7-dev libcudnn8-dev

# (Optionally) Install TensorRT support
if [[ ${TENSORRT} == true ]]
then
  # Install TensorRT. Requires that libcudnn7 is installed above.
  sudo apt-get install -y --no-install-recommends \
    libnvinfer6 \
    libnvinfer-dev \
    libnvinfer-plugin6
fi

# (Optionally) Install NVIDIA Docker support
if [[ ${DOCKER} == true ]]
then
  # Setup the stable repository and the GPG key
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  # Install the nvidia-docker2 package (and dependencies) after updating the package listing
  sudo apt update && sudo install -y nvidia-docker2
  # Restart the Docker daemon to complete the installation after setting the default runtime
  sudo systemctl restart docker
  # Check installation
  docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
fi

# EOF
