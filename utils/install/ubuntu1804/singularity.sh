#!/usr/bin/env bash

#========================================================================================
# Copyright (C) 2023, Robotic Systems Lab, ETH Zurich
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

# Install system APT dependencies
sudo apt update && sudo apt install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    pkg-config

#==
# Go
#==

# Download and install GO
PREFIX=
GO_INSTALL_DIR=$HOME/.local
export VERSION=1.11 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    $PREFIX rm -rf "$GO_INSTALL_DIR"/go &&
    $PREFIX tar -C "$GO_INSTALL_DIR" -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz

# Configure binary paths
echo "export GOPATH=$HOME/.go" >> ~/.bashrc && \
    echo "export PATH=$GO_INSTALL_DIR/go/bin:$PATH:$GOPATH/bin" >> ~/.bashrc && \
    source ~/.bashrc

# Check installation
go version

#==
# Singularity
#==

#
UBUNTU=focal
sudo wget -O- http://neuro.debian.net/lists/$UBUNTU.us-ca.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list && \
    sudo apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9 && \
    sudo apt update

#
sudo apt install -y singularity-container

# Check installation
singularity --version

# EOF
