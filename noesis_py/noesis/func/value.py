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

"""Graph-level implementations of common policy function-approximators.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp

from tensorflow.contrib.layers import fully_connected, xavier_initializer
from noesis.core.tensor import variables_accessors


# TODO: Define base class for Policy
class StateValueFunction(tf.Module):
    """
    TODO
    """
    def __init__(self, latent, name="StateValueFunction"):
        super(StateValueFunction, self).__init__(name=name)
        assert isinstance(latent, tf.Module)
        # TODO
        self.__latent = latent.output
        self.__dtype = latent.output.dtype
        self.__theta = tf1.get_collection(scope=latent.name_scope.name, key=tf1.GraphKeys.TRAINABLE_VARIABLES)
        # TODO
        self.__value_op = None
        # TODO
        with tf.name_scope(self.name_scope.name):
            with tf.name_scope("latent"):
                # TODO: Make configurable
                vfn = None
                latent = fully_connected(
                    inputs=self.__latent,
                    num_outputs=int(1),
                    activation_fn=vfn,
                    weights_initializer=tf1.initializers.glorot_uniform(),
                    biases_initializer=tf1.initializers.zeros(),
                    scope=self.name_scope.name + "latent",
                    trainable=True,
                )
            # Wrap the output value tensor
            with tf.name_scope("value_distribution"):
                self.__value_op = tf.identity(latent, name="mean")
            # Store the list of trainable variables
            self.__theta += tf1.get_collection(scope=self.name_scope.name, key=tf1.GraphKeys.TRAINABLE_VARIABLES)
            # Define the action distribution
            with tf.name_scope("parameters"):
                var_ops = variables_accessors(var_list=self.__theta, dtype=self.__dtype)
                self.__theta_size_op = var_ops[0]
                self.__theta_in = var_ops[1]
                self.__theta_in_set_op = var_ops[2]
                self.__theta_in_get_op = var_ops[3]

    @property
    def value(self):
        return self.__value_op

    @property
    def parameters(self):
        return self.__theta

# EOF
