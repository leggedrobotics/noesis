============
Installation
============

Preparation
~~~~~~~~~~~

First the source repository must be cloned (in a user-defined location):

.. code:: bash

    git clone git@bitbucket.org:leggedrobotics/noesis.git

The ``install.sh`` script is provided for installing all dependencies
and preparing the development environment for Noesis. Specifically, it
will execute the following steps:

1. Install all ``apt`` dependencies.
2. Install ``virtualenv`` and ``virtualenvwrapper`` in the global
   ``pip`` environment.
3. Create a ``virtualenv`` environment named ``noesis``.
4. Install the ``noesis_py`` package via ``pip`` within the ``noesis``
   environment . This also installs all ``pip`` dependencies.

Before proceeding to run the ``install.sh`` script, consider which
version of TensorFlow is to be used: CPU-only or with GPU support.

For CPU-only, proceed with the following commands:

.. code:: bash

    cd noesis/
    ./install.sh --cpu

For GPU support, first follow the `instructions for GPU
support <https://www.tensorflow.org/install/gpu>`__ from the TensorFlow
website. Then proceed with the following commands:

.. code:: bash

    cd noesis/
    ./install.sh --gpu

We can test whether the TensorFlow and Noesis installation succeeded buy
running the TensorFlow device discovery utility:

.. code:: bash

    workon noesis
    tfdevices

If this succeeds then you should see something like this:

.. code:: bash

    (noesis) user@ubuntu:~$ tfdevices
    ALL:  ['/device:CPU:0', '/device:GPU:0']

Building
~~~~~~~~

GCC Version
^^^^^^^^^^^

Noesis is supported for GCC v7.3 and higher versions. It is installed by
``install.sh`` script, but CMake still needs to be directed to use this
version instead of the system default. There are two options recommended
for achieving this: 1. Adding
``export CC=/usr/bin/gcc-7; export CXX=/usr/bin/g++-7`` to the user's
``~/.bashrc``. 2. Adding
``-DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7``
to the command line call to cmake to generate the build configuration.

CMake
^^^^^

Building can be performed anywhere in the users' home directory using
CMake. We provide an example in the form of a CMake project template in
the ``utils/cmake_project/`` directory. We recommend to use this for
getting started. Lets assume we cloned the Noesis repository in
``~/git/noesis`` and we want to setup the development and build
directory in ``~/Noesis``:

.. code:: bash

    cp -r noesis/framework/utils/cmake_project ~/Noesis
    cd ~/Noesis
    ln -s ~/git/noesis ~/Noesis/src/noesis
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 ..

Now we can build individual build targets (libraries and executables),
e.g. ``framework_hello``:

.. code:: bash

    make framework_hello

Catkin
^^^^^^

For building with Catkin, please refer to
`this <https://catkin-tools.readthedocs.io/en/latest/>`__ resource on
how to install, configure and use catkin.

.. code:: bash

    mkdir -p catkin_ws/src
    cd catkin_ws
    catkin init
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_EXPORTED_LIBS=ON
    catkin build noesis_examples

Now we can test the build by executing any of the examples. Lets try the
``framework_hello`` example:

.. code:: bash

    user@ubuntu:~$ ./build/noesis/noesis_examples/framework_hello
