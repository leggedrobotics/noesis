# CMake Workspace

This provides a template for setting up a CMake workspace for Noesis projects.

## Overview

Using the provided workspace template allows us to contain all our work within a
few user-local directories. Thus removing a workspace and everything installed simply 
amounts to deleting the following directories:  
* The workspace directory itself.
* `~/.noesis` in the user's home directory.
* `~/.virtualenvs/noesis` in the user's home directory.

This prevents polluting the global filesystem. 

## Getting Started

First we should decide where we want to organize project directories. As an example
we can assume we want our workspace at `~/cmake_ws` and we will clone our git repos
at `~/git` and symlink them to `~/cmake_ws/src`. This will give us flexibility we 
want to also work with `catkin` workspaces.

First we must install all `apt` and `git` dependencies:
```commandline
mkdir -p ~/git
git clone git@bitbucket.org:leggedrobotics/noesis.git ~/git/noesis
cd ~/git/noesis/utils/install
cp -r ../workspace ~/cmake_ws
./nvidia.sh
./docker.sh
./noesis.sh --git=${HOME}/git --install=${HOME}/cmake_ws/lib --gpu
```

Then we can install `pip` python dependencies using a `virtualenv`:
```commandline
pip install -U pip setuptools virtualenvwrapper
python -m virtualenv --python=/usr/bin/python3 ${HOME}/.virtualenvs/noesis
source ${HOME}/.local/bin/virtualenvwrapper.sh
workon noesis
pip install -U pip setuptools
pip install -e ${HOME}/git/noesis/noesis_py[gpu]
```

Then we can install special dependencies like support for physics engines.

For RaiSim:
```commandline
./raisim.sh --git=${HOME}/git --install=${HOME}/cmake_ws/lib
```

For MuJoCo:
```commandline
./mujoco.sh --git=${HOME}/git --install=${HOME}/cmake_ws/lib
```

## Environment

In order for the workspace to fully work out all paths, the provided `setup.sh` script must
either by sourced via `source /path/to/workspace/setup.bash`, once every terminal or login
session. Alternatively we can add the aforementioned command the user's `~/.bashrc` to occur
automatically every time we start a new terminal session.

## Packages

CMake packages can be added to the workspace using the convenience marco 
`project_add_module(<RELATIVE-PACKAGE-PATH>)`. This marco is making use of the standard 
`add_subdirectory()` CMake command plus additional operations that ensure all contents 
of the package are being added to the master CMake project, e.g. for code indexing etc.

## Options

The workspace can be configured using the CMake options of each package added to the build. To do
so we use the `project_enable_option(<OPTION>)` macro to enable individual options. The table below
lists the options relevant to configuring `tensroflow-cpp` and `noesis`:  

| __Option Name__       | __Role__                                                               |
|-----------------------|------------------------------------------------------------------------|
| TF_USE_GPU            | Enables using the pre-built GPU version of TensorFlow.                 |
| TF_USE_IN_SOURCE      | Disables installation targets and uses headers and libraries in-place. |
| Noesis_USE_MUJOCO     | Enables support for the MuJoCo physics engine and gym environments.    |
| Noesis_USE_RAISIM     | Enables support for the RaiSim physics engine and gym environments.    |
| Noesis_USE_CXX17      | Enables the use of C++17 as the required standard.                     |
| Noesis_USE_SIMD       | Enables the use of CPU SIMD commands such as SSE, AVX and FMA.         |
| Noesis_USE_TCMALLOC   | Enables use of TCMALLOC for memory allocations and sanitizing.         |
| Noesis_USE_SANITIZERS | Enables the use of GCC memory and thread sanitizers.                   |
| Noesis_BUILD_TESTS    | Adds unit tests to the project and build.                              |

See the [CMakeLists.txt](./CMakeLists.txt) of the template as an example.

----
