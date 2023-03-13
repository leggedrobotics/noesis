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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from tensorflow.contrib.layers import fully_connected, xavier_initializer
from noesis.core.tensor import flatten_and_concatenate


class MultiLayerPerceptron(tf.Module):
    """
    TODO
    """
    def __init__(self, inputs, units, activations, name="MultiLayerPerceptron"):
        super(MultiLayerPerceptron, self).__init__(name=name)
        # TODO
        assert isinstance(inputs, list)
        assert isinstance(units, list)
        assert isinstance(activations, list)
        assert len(units) == len(units)
        # TODO
        self.__inputs = inputs
        self.__layers = []
        self.__output = None
        # TODO
        with tf.name_scope(self.name_scope.name):
            with tf.name_scope('input'):
                top = flatten_and_concatenate(self.__inputs, name="flat")
                self.__layers.append(top)
            # TODO: Make configurable ???
            b0 = tf.cast(x=float(1e-3), dtype=top.dtype)
            # TODO: Use Keras Dense
            for i in range(0, len(units)):
                name = 'hidden_' + repr(i)
                with tf.name_scope(name):
                    top = fully_connected(
                        inputs=top,
                        num_outputs=units[i],
                        activation_fn=activations[i],
                        weights_initializer=xavier_initializer(),
                        biases_initializer=tf.random_uniform_initializer(-b0, b0),
                        scope=self.name_scope.name + name,
                        trainable=True)
                    self.__layers.append(top)
            self.__output = tf.identity(top, name="output")

    @property
    def inputs(self):
        return self.__inputs

    @property
    def layers(self):
        return self.__layers

    @property
    def output(self):
        return self.__output

# EOF
