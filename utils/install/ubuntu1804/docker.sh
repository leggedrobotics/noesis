#!/usr/bin/env bash

#========================================================================================
# Copyright (C) 2021, Robotic Systems Lab, ETH Zurich
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

#==
# APT dependencies
#==

# Remove current installation
sudo apt remove docker docker-engine docker.io containerd runc

# Install system APT dependencies
sudo apt update && sudo apt install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release

# Add the official docker APT repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install docker and configure the necessary user group
sudo apt update && sudo apt install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io

# Create the `docker` group and add current user to that group
sudo groupadd docker
sudo usermod -aG docker "$USER"
newgrp docker

# Check installation
docker run hello-world

# EOF
