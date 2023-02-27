# Copyright 2019 The Noesis Authors. All Rights Reserved.
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

"""An implementations of convolutional (feed-forward) neural-network architecture for function
approximation."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.contrib.layers import fully_connected, xavier_initializer, convolution2d, \
    flatten, max_pool2d, dropout, batch_norm
from tensorflow.contrib.rnn import static_rnn
from tensorflow.nn.rnn_cell import LSTMCell, GRUCell

from noesis.common.configuration import *
from noesis.core.base import Architecture
from noesis.core.registration import register
from noesis.common.tensor import flatten_and_concatenate, split_and_fold, flatten_dimensions


class ConvolutionalNeuralNetwork(Architecture):
    """
    CNN architecture with RNN support.
    """
    def __init__(self, namescope, inputs_spec, outputs_spec, arch_config, dtype, device, verbose):
        super(ConvolutionalNeuralNetwork, self).__init__(namescope, inputs_spec, outputs_spec, arch_config, dtype, device,
                                                         verbose)
        # Retrieve the layer activations list
        self._layer_activations, self._layer_units, self._max_initial_value = get_architecture_from_xml_config(arch_config)

    def add_subgraph(self, graph, verbose=False):
        # Retrieve arch input-output specifications
        inputs = self.get_input_nodes()
        inputs_dims = self.get_input_dimensions()
        outputs_names = self.get_output_names()
        outputs_dims = self.get_output_dimensions()

        # Ensure input dimensions are flat:
        assert (len(inputs_dims[0]) <= 3), "First input tensor must be at most rank 3 (i.e. image)."
        for dims in inputs_dims[1:]:
            assert (len(dims) <= 2), "Input tensors must be at most rank 2 (i.e. matrices)."
        for dims in outputs_dims:
            assert (len(dims) <= 2), "Output tensors must be at most rank 2 (i.e. matrices)."

        # Compute flattened output dimensions
        flat_output_dims = flatten_dimensions(outputs_dims)

        # Append all operations to the specified tensorflow graph
        with graph.as_default():
            with tf.device(self.device):

                with tf.name_scope(self.operations_namescope):
                    # Construct sub-graph inputs
                    with tf.name_scope('inputs'):
                        # Fully-connected input layer
                        new_shape = (-1, inputs_dims[0][0], inputs_dims[0][0], 1)
                        input_layer = tf.reshape(inputs[0], new_shape, name="image")

                    # Construct sub-graph operations
                    with tf.name_scope("CNN"):
                        top = input_layer
                        param_namescope = self.parameters_namescope + "/CNN"
                        top = convolution2d(inputs=top,
                                            kernel_size=7,
                                            stride=1,
                                            num_outputs=64,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_0')

                        top = convolution2d(inputs=top,
                                            kernel_size=5,
                                            stride=1,
                                            num_outputs=64,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_1')

                        top = max_pool2d(inputs=top,
                                         kernel_size=2,
                                         scope=param_namescope+'/max_pool_0')

                        top = convolution2d(inputs=top,
                                            kernel_size=3,
                                            stride=1,
                                            num_outputs=64,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_2')

                        top = convolution2d(inputs=top,
                                            kernel_size=3,
                                            stride=1,
                                            num_outputs=64,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_3')

                        top = max_pool2d(inputs=top,
                                         kernel_size=2,
                                         scope=param_namescope+'/max_pool_1')

                    with tf.name_scope("additional_inputs"):
                        param_namescope = self.parameters_namescope + "/add_inputs"
                        if len(inputs) > 1:
                            add_layer_in = flatten_and_concatenate(inputs[1:], inputs_dims[1:])
                            add_layer = fully_connected(inputs=add_layer_in,
                                                        num_outputs=128,
                                                        activation_fn=tf.nn.relu,
                                                        normalizer_fn=batch_norm,
                                                        weights_initializer=xavier_initializer(),
                                                        trainable=True,
                                                        scope=param_namescope + "/FC")
                            add_layer = tf.expand_dims(tf.expand_dims(add_layer, 1), 1)
                            add_layer = tf.tile(add_layer, [1, tf.shape(top)[1], tf.shape(top)[2], 1])
                            top = tf.concat([top, add_layer], 3)

                            top = convolution2d(inputs=top,
                                                kernel_size=2,
                                                stride=1,
                                                num_outputs=64,
                                                activation_fn=tf.nn.relu,
                                                normalizer_fn=batch_norm,
                                                weights_initializer=xavier_initializer(),
                                                trainable=True,
                                                scope=param_namescope + '/CNN')

                            top = max_pool2d(inputs=top,
                                             kernel_size=2,
                                             scope=param_namescope + '/max_pool')

                    top = flatten(top)

                    with tf.name_scope("fully_connected"):
                        param_namescope = self.parameters_namescope + "/fully_connected"
                        top = fully_connected(inputs=top,
                                              num_outputs=256,
                                              activation_fn=tf.nn.relu,
                                              normalizer_fn=batch_norm,
                                              weights_initializer=xavier_initializer(),
                                              trainable=True,
                                              scope=param_namescope + "/FC_1")

                    with tf.name_scope("RNN"):
                        param_namescope = self.parameters_namescope + "/rnn"
                        rnn_cell = GRUCell(256)
                        # rnn_cell = LSTMCell(256)
                        top, _ = static_rnn(rnn_cell, [top], dtype=tf.float32, scope=param_namescope)
                        top = tf.squeeze(top, 0)

                    with tf.name_scope("fully_connected"):
                        param_namescope = self.parameters_namescope + "/fully_connected"
                        top = fully_connected(inputs=top,
                                              num_outputs=128,
                                              activation_fn=tf.nn.relu,
                                              normalizer_fn=batch_norm,
                                              weights_initializer=xavier_initializer(),
                                              trainable=True,
                                              scope=param_namescope + "/FC_2")

                    with tf.name_scope("dropout"):
                        top = dropout(top, 0.9)

                    # Fully-connected output layer
                    with tf.name_scope('outputs'):
                        # Configure the output layer
                        output_layer = fully_connected(inputs=top,
                                                       num_outputs=sum(flat_output_dims),
                                                       activation_fn=self._layer_activations[-1],
                                                       weights_initializer=xavier_initializer(),
                                                       trainable=True,
                                                       scope=self.parameters_namescope + "/output")
                        # Check for multiple outputs and split if true
                        outputs = split_and_fold(output_layer, flat_output_dims, outputs_dims, outputs_names)

        # Store inputs
        self.add_input_node(inputs)
        # Store outputs
        self.add_output_node(outputs)


# Register architecture implementation
register("ARCH", ConvolutionalNeuralNetwork)

# EOF
