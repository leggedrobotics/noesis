#!/usr/bin/env bash

#========================================================================================
#      ___           ___           ___           ___                       ___
#     /__/\         /  /\         /  /\         /  /\        ___          /  /\
#     \  \:\       /  /::\       /  /:/_       /  /:/_      /  /\        /  /:/_
#      \  \:\     /  /:/\:\     /  /:/ /\     /  /:/ /\    /  /:/       /  /:/ /\
#  _____\__\:\   /  /:/  \:\   /  /:/ /:/_   /  /:/ /::\  /__/::\      /  /:/ /::\
# /__/::::::::\ /__/:/ \__\:\ /__/:/ /:/ /\ /__/:/ /:/\:\ \__\/\:\__  /__/:/ /:/\:\
# \  \:\~~\~~\/ \  \:\ /  /:/ \  \:\/:/ /:/ \  \:\/:/~/:/    \  \:\/\ \  \:\/:/~/:/
#  \  \:\  ~~~   \  \:\  /:/   \  \::/ /:/   \  \::/ /:/      \__\::/  \  \::/ /:/
#   \  \:\        \  \:\/:/     \  \:\/:/     \__\/ /:/       /__/:/    \__\/ /:/
#    \  \:\        \  \::/       \  \::/        /__/:/        \__\/       /__/:/
#     \__\/         \__\/         \__\/         \__\/                     \__\/
#
#
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

# Set repository root path
NOESIS_ROOT="$( realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"/../../ )"
echo "[install noesis]: Noesis root at: ${NOESIS_ROOT}"

# Default configurations
SOURCE_DIR=$HOME/.noesis/src
INSTALL_DIR=$HOME/.local
GPU=false
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
  --gpu)
  GPU=true
  shift # past argument with no value
  ;;
  --build)
  BUILD=true
  shift # past argument with no value
  ;;
  *)
  echo "[install noesis]: Error: Unknown arguments: ${i#*=}"
  exit 1
  ;;
esac
done

#==
# Dependencies
#==

# System dependencies
sudo apt update && sudo apt install -y \
  software-properties-common git git-lfs build-essential pkg-config cmake libboost-all-dev libtool \
  python3 python3-pip python3-virtualenv \
  libgtest-dev googletest google-mock google-perftools \
  libsfml-dev libyaml-cpp-dev libtinyxml-dev

# Clone all source repositories into the project's source directory
git clone https://github.com/leggedrobotics/tensorflow-cpp.git ${SOURCE_DIR}/tensorflow-cpp

# Install Eigen
${SOURCE_DIR}/tensorflow-cpp/eigen/install.sh --run-cmake "${INSTALL_DIR}"

# (Optionally) Install TensorFlow-Cpp
if [[ ${BUILD} == true ]]
then
  if [ ${GPU} == true ]; then TF_USE_GPU=ON; else TF_USE_GPU=OFF; fi
  mkdir ${SOURCE_DIR}/tensorflow-cpp/tensorflow/build
  cd ${SOURCE_DIR}/tensorflow-cpp/tensorflow/build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DTF_USE_GPU=${TF_USE_GPU}
  make install -j
fi

# EOF
