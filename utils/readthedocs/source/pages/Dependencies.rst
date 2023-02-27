============
Dependencies
============

Noesis has ``apt``, ``pip`` and source dependencies.

APT
~~~

All ``apt`` dependencies are installed by the ``install.sh`` script (see
installation instructions below):

- `GCC 7.3 <https://launchpad.net/~ubuntu-toolchain>`__: GNU C/C++
   Compiler (GCC) v7 provided by the advanced Ubuntu toolchain
   repository.
- `OpenMP 4.5 <https://www.openmp.org/uncategorized/openmp-45-specs-released/>`__:
   OpenMP 4.5 provided by GCC 7.
- `Boost <https://www.boost.org/>`__: Free peer-reviewed portable
   C++ source libraries.
- `CMake <https://cmake.org>`__: CMake is an open-source,
   cross-platform family of tools designed to build, test and package
   software.
- `Python 3.5 <https://www.python.org/downloads/release/python-352/>`__:
   Current default version of Python 3.5 provided in Ubuntu 16.04 LTS.
- `SDL2 <https://www.libsdl.org/download-2.0.php>`__: SDL is a
   cross-platform development library providing abstractions based on
   OpenGL.
- `SFML <https://www.sfml-dev.org/>`__: Simple and Fast Multimedia
   Library used for drawing 2D graphics.

PIP
~~~

All ``pip`` dependencies are automatically installed when by the
``noesis_py`` package via the ``install.sh`` script:

- `TensorFlow Python <https://github.com/tensorflow/tensorflow>`__:
   Computation using data flow graphs for scalable machine learning.

Source
~~~~~~

All source dependencies are provided with the source repository in
``thirdparty/`` directory and are integrated via respective CMake
packages:

- `Eigen3 <https://bitbucket.org/eigen/eigen>`__: Eigen is a C++
   template library for linear algebra: matrices, vectors, numerical
   solvers, and related algorithms.
- `TensorFlow C/C++ <https://github.com/tensorflow/tensorflow>`__:
   CMake package providing the headers and libraries for the C/C++ API
   of TensorFlow.
- `TinyXML <https://sourceforge.net/projects/tinyxml/>`__: TinyXML
   is a simple, small, minimal, C++ XML parser that can be easily
   integrating into other programs.
- `STB <https://github.com/nothings/stb>`__: STB is a set of
   single-file public domain libraries for C/C++. Only the parts for
   image processing are used.