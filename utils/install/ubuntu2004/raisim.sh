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

# Default configurations
SOURCE_DIR=$HOME/.raisim/src
INSTALL_DIR=$HOME/.local
BUILD=false

# Iterate over arguments list to configure the installation.
for i in "$@"
do
case $i in
    --git=*)
    SOURCE_DIR="${i#*=}"
    shift # past argument=value
    ;;
    --install=*)
    INSTALL_DIR="${i#*=}"
    shift # past argument=value
    ;;
    --build)
    BUILD=true
    shift # past argument with no value
    ;;
    *)
    echo "[install raisim]: Error: Unknown arguments: ${i#*=}"
    exit 1
    ;;
esac
done

#==
# APT dependencies
#==

# System Dependencies
sudo apt update && sudo apt install -y \
  libgles2-mesa-dev \
  libxt-dev \
  libxaw7-dev \
  libsdl2-dev \
  libzzip-dev \
  libfreeimage-dev \
  libfreetype6-dev \
  libpugixml-dev

#==
# Source dependencies
#==

# Clone all source repositories into the project's source directory
git clone --branch raisimOgre https://github.com/leggedrobotics/ogre.git ${SOURCE_DIR}/ogre
git clone --branch feature/in-source-cmake-build https://github.com/vastsoun/raisimLib.git ${SOURCE_DIR}/raisimLib
git clone --branch feature/update-raisim-version https://github.com/vastsoun/raisimOgre.git ${SOURCE_DIR}/raisimOgre

#==
# Build and install
#==

# Install OGRE
mkdir ${SOURCE_DIR}/ogre/build
cd ${SOURCE_DIR}/ogre/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DOGRE_BUILD_COMPONENT_BITES=ON \
  -DOGRE_BUILD_COMPONENT_JAVA=OFF \
  -DOGRE_BUILD_DEPENDENCIES=OFF \
  -DOGRE_BUILD_SAMPLES=False
make install -j4

# (Optionally) Install raisim
if [[ ${BUILD} == true ]]
then
  # Install raisimLib
  mkdir ${SOURCE_DIR}/raisimLib/build
  cd ${SOURCE_DIR}/raisimLib/build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_PREFIX_PATH=${INSTALL_DIR}
  make install -j4

  # Install raisimOgre
  mkdir ${SOURCE_DIR}/raisimOgre/build
  cd ${SOURCE_DIR}/raisimOgre/build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_PREFIX_PATH=${INSTALL_DIR}
  make install -j4
fi

# EOF
