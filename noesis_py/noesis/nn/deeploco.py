# Copyright 2020 The Noesis Authors. All Rights Reserved.
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

from tensorflow.contrib.layers import flatten, fully_connected, xavier_initializer, convolution2d, max_pool2d
from noesis.core.tensor import flatten_and_concatenate, split_and_fold


class DeepLocoNetwork(tf.Module):
    """
    TODO
    """
    def __init__(
            self,
            perception,
            states,
            cnn_kernels=[3, 3, 3, 3, 3],
            cnn_strides=[2, 1, 2, 1, 1],
            cnn_outputs=[16, 16, 32, 32, 32],
            cnn_formats=["NHWC", None, None, None, None],
            cnn_activations=[tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu],
            cnn_out_units=128,
            cnn_out_activations=tf.nn.leaky_relu,
            mlp_units=[512, 256],
            mlp_activations=[tf.nn.leaky_relu, tf.nn.leaky_relu],
            name="DeepLocoNetwork"):
        super(DeepLocoNetwork, self).__init__(name=name)
        # Check arguments
        assert isinstance(perception, list)
        assert isinstance(perception[0], tf.Tensor)
        assert isinstance(states, list)
        assert isinstance(states[0], tf.Tensor)
        # Initialize internal members
        self.__states = states
        self.__perception = perception
        self.__inputs = [states, perception]
        self.__layers = []
        self.__output = None

        # Define network architecture
        with tf.name_scope(self.name_scope.name):

            # Inputs
            with tf.name_scope('input'):
                state = flatten_and_concatenate(self.__states, name="state/flat")
                # print("state: ", state)
                elevation_dims = perception[0].shape
                elevation_dims = (-1, elevation_dims[2], elevation_dims[1], 1)
                elevation = tf.reshape(perception[0], shape=elevation_dims, name="perception/elevation")
                # print("elevation: ", elevation)
                self.__layers.append(state)
                self.__layers.append(elevation)

            # High-dimensional channel
            with tf.name_scope('Perception'):
                # Input is the elevation tensor
                top = elevation
                # Stack of convolutional layers
                for i in range(0, len(cnn_kernels)):
                    name = 'cnn_' + repr(i)
                    with tf.name_scope(name):
                        top = convolution2d(
                            inputs=top,
                            padding='VALID',
                            kernel_size=cnn_kernels[i],
                            stride=cnn_strides[i],
                            data_format=cnn_formats[i],
                            num_outputs=cnn_outputs[i],
                            activation_fn=cnn_activations[i],
                            normalizer_fn=None,
                            weights_initializer=xavier_initializer(),
                            trainable=True,
                            scope=self.name_scope.name + name)
                        self.__layers.append(top)
                        # print("cnn: ", top)
                # Single fully-connected layer
                top = flatten(top)
                self.__layers.append(top)
                # print("flattened: ", top)
                with tf.name_scope("cnn_out"):
                    top = fully_connected(
                        inputs=top,
                        num_outputs=cnn_out_units,
                        activation_fn=cnn_out_activations,
                        normalizer_fn=None,
                        weights_initializer=xavier_initializer(),
                        trainable=True,
                        scope=self.name_scope.name + "cnn_out")
                    self.__layers.append(top)
                    # print("cnn_out: ", top)

            # Low-dimensional channel: Fully-connected layers
            with tf.name_scope("MLP"):
                # Input is the cnn latent output and state inputs
                top = tf.concat([top, state], 1)
                self.__layers.append(top)
                # print("concat: ", top)
                # TODO: Make configurable ???
                b0 = tf.cast(x=float(1e-3), dtype=top.dtype)
                # Stack of fully-connected layers
                for i in range(0, len(mlp_units)):
                    name = 'mlp_' + repr(i)
                    with tf.name_scope(name):
                        top = fully_connected(
                            inputs=top,
                            num_outputs=mlp_units[i],
                            activation_fn=mlp_activations[i],
                            normalizer_fn=None,
                            weights_initializer=xavier_initializer(),
                            biases_initializer=tf.random_uniform_initializer(-b0, b0),
                            scope=self.name_scope.name + name,
                            trainable=True)
                        self.__layers.append(top)
                        # print("mlp: ", top)

            # Final output is top of MLP layers
            self.__output = tf.identity(top, name="output")

    @property
    def states(self):
        return self.__states

    @property
    def perception(self):
        return self.__perception

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
