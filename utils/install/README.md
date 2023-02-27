# Installing Noesis

**NOTE**: The following sections should be followed *in order*. 

## Quickstart

Please see [these](../workspace/README.md) instructions for getting started.

## Step-by-Step

### NVIDIA

If intending to run computations on GPU, then first follow/use the `nvidia.sh` bash 
script. If TensorRT is to be used then the complete installation can be performed 
by running `./nvidia.sh --tensorrt`.

### Noesis

Core dependencies can be installed using the `noesis.sh` bash script. This by default will 
clone the `tensorflow-cpp` repository using `git` at `~/.noesis/src` and build that and it's 
Eigen3 dependency respectively and install both at in `~/.local` by default. 

Moreover we can specify the destination directories for cloning and installing by adding 
the `--git=<DESTINATION-DIR>` and `--install=<INSTALL-DIR>` arguments respectively. If in
addition we want `tensorflow-cpp` to be built and installed we can add the `--build` argument.

### Simulators

#### RaiSim

RaiSim and its OGRE-based visualization front-end can be installed using the `raism.sh`
bash script. All relevant packages are cloned using `git` in the `~/.noesis/src` directory
and can then be built from source.

If the `raisimLib` and `raisimOgre` are not to be modified by your project, then it is
recommended to install them using `./raisim.sh --build`, otherwise we can create symbolic
links from `~/.noesis/src` to wherever we intend to modify them from.

Similar to above, we can specify the destination directories for cloning and installing 
all repositories by adding the `--git=<DESTINATION-DIR>` and `--install=<INSTALL-DIR>` 
arguments respectively. If in addition we want all RaiSim packages to be built and installed 
we can add the `--build` argument.

#### MuJoCo

**Coming Soon**

### Noesis Python

In addition to the aforementioned steps, we must also create a python virtualenv environment 
and install basic dependency packages:
```commandline
# Update+install common python utilities   
pip install -U pip setuptools virtualenvwrapper

# Creates a new virtualenv
python -m virtualenv --python=/usr/bin/python3 ${HOME}/.virtualenvs/noesis

# Enables virtualenvwrapper --> skip if this is already present in your .bashrc
source /usr/local/bin/virtualenvwrapper.sh

# Enables the new virtualenv
workon noesis # Enable the virtualenv
```

Next we need to install `noesis` and its `tensorflow` dependencies based on the device supported 
by the host machine. Let's assume the workspace has been created at a location specified by a `
NOESIS_WS` environment variable.  

For NVIDIA GPU-enabled setups:
```commandline
pip install -e <GIT-DIR>/noesis/noesis_py[gpu]
```

For CPU-only setups:
```commandline
pip install -e <GIT-DIR>/noesis/noesis_py[cpu]
```

If the aforementioned steps succeeded then by running `tfdevices` in an active virtualenv we should 
see something like this:
```commandline
(noesis) user@ubuntu:~$ tfdevices 
ALL:  ['/device:CPU:0', '/device:GPU:0']
```

----
