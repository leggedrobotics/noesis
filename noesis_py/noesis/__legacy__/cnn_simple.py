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

from noesis.common.configuration import *
from noesis.core.base import Architecture
from noesis.core.registration import register
from noesis.common.tensor import flatten_and_concatenate, split_and_fold, flatten_dimensions


class ConvolutionalNeuralNetwork(Architecture):
    """
    Generic CNN architecture.
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

        # Get image and remove it from the lists
        new_inputs = inputs
        new_inputs_dims = inputs_dims
        image = self.get_input_node_by_name("image")
        image_dim = new_inputs_dims.pop(new_inputs.index(image))
        new_inputs.remove(image)

        # Ensure input dimensions are flat:
        assert (len(image_dim) in [2, 3]), "Image must be at most rank 3."
        for dims in new_inputs_dims:
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
                        if len(image_dim) == 2:
                            image_input = tf.expand_dims(image, -1)
                        else:
                            image_input = image
                        # Channels first for better GPU optimization
                        image_input = tf.transpose(image_input, [0, 3, 1, 2])

                    # Construct sub-graph operations
                    with tf.name_scope("CNN"):
                        top = image_input
                        param_namescope = self.parameters_namescope + "/CNN"
                        top = convolution2d(inputs=top,
                                            kernel_size=3,
                                            stride=1,
                                            data_format="NCHW",
                                            num_outputs=16,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_0')

                        top = convolution2d(inputs=top,
                                            kernel_size=3,
                                            stride=1,
                                            data_format="NCHW",
                                            num_outputs=16,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_1')

                        top = max_pool2d(inputs=top,
                                         kernel_size=2,
                                         data_format="NCHW",
                                         scope=param_namescope+'/max_pool_0')

                        top = convolution2d(inputs=top,
                                            kernel_size=3,
                                            stride=1,
                                            data_format="NCHW",
                                            num_outputs=32,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_2')

                        top = convolution2d(inputs=top,
                                            kernel_size=3,
                                            stride=1,
                                            data_format="NCHW",
                                            num_outputs=32,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=xavier_initializer(),
                                            trainable=True,
                                            scope=param_namescope+'/layer_3')

                        top = max_pool2d(inputs=top,
                                         kernel_size=2,
                                         data_format="NCHW",
                                         scope=param_namescope+'/max_pool_1')

                        top = flatten(top)

                        top = fully_connected(inputs=top,
                                              num_outputs=256,
                                              activation_fn=tf.nn.relu,
                                              normalizer_fn=batch_norm,
                                              weights_initializer=xavier_initializer(),
                                              trainable=True,
                                              scope=param_namescope + "/FC")

                    with tf.name_scope("additional_inputs"):
                        param_namescope = self.parameters_namescope + "/add_inputs"
                        add_layer_in = flatten_and_concatenate(new_inputs, new_inputs_dims)
                        add_layer = fully_connected(inputs=add_layer_in,
                                                    num_outputs=64,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=batch_norm,
                                                    weights_initializer=xavier_initializer(),
                                                    trainable=True,
                                                    scope=param_namescope + "/FC")
                        top = tf.concat([top, add_layer], 1)

                    with tf.name_scope("fully_connected"):
                        param_namescope = self.parameters_namescope + "/fully_connected"
                        top = fully_connected(inputs=top,
                                              num_outputs=128,
                                              activation_fn=tf.nn.relu,
                                              normalizer_fn=batch_norm,
                                              weights_initializer=xavier_initializer(),
                                              trainable=True,
                                              scope=param_namescope + "/FC_1")

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
