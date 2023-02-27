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

# Instructions based on https://github.com/hpcng/singularity/blob/master/INSTALL.md

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
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup-bin

#==
# Go
#==

# Download and install GO
PREFIX=
GO_INSTALL_DIR=$HOME/.local
export VERSION=1.15.8 OS=linux ARCH=amd64 && \
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

# Installs golangci-lint
curl -sfL https://install.goreleaser.com/github.com/golangci/golangci-lint.sh | sh -s -- -b $(go env GOPATH)/bin v1.21.0

# Retrieves the Singularity source files
rm -rf ${GOPATH}/src/github.com/sylabs && \
mkdir -p ${GOPATH}/src/github.com/sylabs && \
  cd ${GOPATH}/src/github.com/sylabs && \
  git clone https://github.com/sylabs/singularity.git && \
  cd singularity && \
  git checkout v3.7.3

# Builds and installs Singularity
cd ${GOPATH}/src/github.com/sylabs/singularity && \
  ./mconfig -p $HOME/.local --without-suid && \
  cd ./builddir && \
  make && make install

# Check installation
singularity exec library://alpine cat /etc/alpine-release

# EOF
