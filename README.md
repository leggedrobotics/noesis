![logo](utils/readthedocs/images/noesis-logo.png)

-----------------

| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](http://docs.leggedrobotics.com/noesis/) |
| [![Build Status](https://ci.leggedrobotics.com/buildStatus/icon?job=bitbucket_leggedrobotics/noesis/master)](https://ci.leggedrobotics.com/job/bitbucket_leggedrobotics/job/noesis/job/master/) |

## Overview

Software for robotic artificial intelligence. Noesis provides a suite of C++ and python libraries,
 mostly targeting applications of Reinforcement Learning (RL) in robotics.

The software currently provides support (i.e. developed and tested) for Ubuntu 18.04 LTS.

The core of the software suite consists of two components:  

1. **`noesis`:** A CMake package providing the C++ library for the following:   
    * `framework`: Runtime implementations of the main infrastructure   
    * `gym`: Wrappers for physics engines, and implementations of RL environments   
    * `rl`: Runtime implementations of all RL algorithms and relevant components    
2. **`noesis_py`:** A `pip` package providing the Python back-end for building and generating computation-graphs using TensorFlow.

Noesis currently uses `C++14` and Python `3.6.9` by default.

Lastly, all C++ components are built using CMake, but also support [`catkin`](https://catkin-tools.readthedocs.io/en/latest/). 
The latter is typically the tool of choice in the robotics community due to the ubiquity of [ROS](http://www.ros.org/). For 
python, we use `virtualenv` and `virtualenvwrapper` for encapsulating `pip` package management on a per-user basis.

**Maintainer:** Vassilios Tsounis  
**Affiliation:** Robotic Systems Lab, ETH Zurich  
**Contact:** tsounisv@ethz.ch

## Install

Please see [these](./utils/install/README.md) instructions on how to install Noesis and all relevant dependencies.

## Build

### CMake

Building can be performed anywhere in the users' home directory using CMake. We provide an example in 
the form of a CMake project template in the `utils/workspace` directory. We recommend to use this 
for getting started. See this [directory](./utils/workspace) for details.

### Catkin

For building with Catkin, please refer to [this](https://catkin-tools.readthedocs.io/en/latest/) 
resource on how to install, configure and use catkin.
```commandline
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release 
catkin build noesis_examples
```

Now we can test the build by executing any of the examples. Lets try the `framework_hello` 
example:
```commandline
user@ubuntu:~$ ./build/noesis/noesis_examples/framework_hello
```

## Dependencies

Noesis has `apt`, `pip` and source dependencies. 

### APT
All `apt` dependencies are installed by the `install.sh` script (see installation instructions below):  

* **[GCC 7.5](https://launchpad.net/~ubuntu-toolchain):** GNU C/C++ Compiler (GCC) v7 provided by the advanced Ubuntu toolchain repository.  
* **[OpenMP 4.5](https://www.openmp.org/uncategorized/openmp-45-specs-released/):** OpenMP 4.5 provided by GCC 7.  
* **[Boost](https://www.boost.org/):** Free peer-reviewed portable C++ source libraries.  
* **[CMake](https://cmake.org):** CMake is an open-source, cross-platform family of tools designed to build, test and package software.  
* **[Python 3.6](https://www.python.org/downloads/release/python-352/):** Current default version of Python 3.5 provided in Ubuntu 16.04 LTS.  
* **[SDL2](https://www.libsdl.org/download-2.0.php):** SDL is a cross-platform development library providing abstractions based on OpenGL.  
* **[SFML](https://www.sfml-dev.org/):** Simple and Fast Multimedia Library used for drawing 2D graphics.  
* **[TinyXML](https://sourceforge.net/projects/tinyxml/):** TinyXML is a simple, small, minimal, C++ XML parser that can be easily integrating into other programs.  
* **[STB](https://github.com/nothings/stb):** STB is a set of single-file public domain libraries for C/C++. Only the parts for image processing are used.  
  
**Note**: STB is already provided in-source in `noesis/include/stb` and no action needs to be taken for it's installation.
  
### PIP
All `pip` dependencies are automatically installed when by the `noesis_py` package via the `install.sh` script:  

* **[TensorFlow 1.15](https://github.com/tensorflow/tensorflow):** Computation using data flow graphs for scalable machine learning.  
  
### Source
Noesis has a single source dependency on TensorFlow-Cpp, a CMake-based re-packaging of the C/C++ interface to TensorFlow, and can be retrieved from the respective 
`tensorflow-cpp` GitHub repository. The aforementioned carries with it the following packages:  

* **[Eigen3](https://bitbucket.org/eigen/eigen):** Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.  
* **[TensorFlow-Cpp](https://github.com/tensorflow/tensorflow):** CMake package providing the headers and libraries for the C/C++ API of TensorFlow.  

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://bitbucket.org/leggedrobotics/noesis/issues?status=new&status=open).

## License

[Apache License 2.0](LICENSE.md)

----
