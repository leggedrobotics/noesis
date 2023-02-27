==================================
Welcome to Noesis's documentation!
==================================

.. image:: ../images/noesis-logo.png
  :align: center

A software for robotic artificial intelligence. Noesis provides a suite of C++ and python libraries, mostly targeting applications of
Reinforcement Learning (RL) in robotics.

The core of the software suite consists of two components:

1. **``noesis``:** A CMake package providing the primary C++ library for the implementations of main infrastructure.
2. **``noesis_py``:** A ``pip`` package providing the Python back-end for building and generating computation-graphs using TensorFlow.

In addition, the following packages are also provided:

3. **``noesis_environments``:** A CMake package providing C++ wrappers for physics engines, and implementations of RL environments.
4. **``noesis_agents``:** A CMake package providing C++ library for the runtime implementations of all RL algorithms and relevant components.
   algorithms.

Noesis currently uses ``C++14`` and Python ``3.5.2``.

Lastly, all C++ components are built using CMake, but also support `catkin <https://catkin-tools.readthedocs.io/en/latest/>`__.
The latter is typically the tool of choice in the robotics community due to the ubiquity of `ROS <http://www.ros.org/>`__. For python, we use
``virtualenv`` and ``virtualenvwrapper`` for encapsulating ``pip`` package management on a per-user basis.

.. toctree::
  :maxdepth: 3
  :caption: Table of Contents
  :glob:

  pages/*

  api/library_root
