# Copyright 2023 The Noesis Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Installation script for the 'noesis' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # TODO: what else is need here?
]

# Dependencies required during setup
SETUP_REQUIRES = [
    # TODO: what else is need here?
]

# Required for package testing
TESTS_REQUIRE = [
    'pytest'
]

# Allows the package to install appropriate version of TensorFlow depending on the
# computing resources available
EXTRAS_REQUIRE = {
    'cpu': ['tensorflow==1.15.2', 'tensorflow-probability==0.8'],
    'gpu': ['tensorflow-gpu==1.15.2', 'tensorflow-probability==0.8'],
    # TODO: add simulation packages
}

# Helper scripts provided by this package
SCRIPTS = [
    'bin/tfdevices'
]

# Installation operation
setup(name='noesis-py',
      version='0.2.0',
      description='Robotic AI using TensorFlow',
      url='https://github.com/leggedrobotics/noesis',
      author='Robotic Systems Lab, ETH Zurich',
      author_email='tsounisv@ethz.ch',
      license='Apache 2.0',
      keywords=["robotics", "reinforcement learning", "machine learning", "tensorflow"],
      packages=[package for package in find_packages() if package.startswith('noesis')],
      scripts=SCRIPTS,
      install_requires=INSTALL_REQUIRES,
      setup_requires=SETUP_REQUIRES,
      tests_require=TESTS_REQUIRE,
      extras_require=EXTRAS_REQUIRE,
      zip_safe=False)

# EOF
