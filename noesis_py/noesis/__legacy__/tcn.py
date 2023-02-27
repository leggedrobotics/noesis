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

"""An implementations of multi-layer perceptron (feed-forward) neural-network architecture for function
approximation."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from noesis.common.configuration import *
from noesis.core.base import Architecture
from noesis.core.registration import register
from noesis.common.tensor import flatten_and_concatenate, split_and_fold, flatten_dimensions


class TemporalConvolutionalNetwork(Architecture):
    """
    Temporal Convolutional Network architecture.
    """
    def __init__(self, namescope, inputs_spec, outputs_spec, arch_config, dtype, device, verbose):
        super(TemporalConvolutionalNetwork, self).__init__(namescope, inputs_spec, outputs_spec, arch_config, dtype, device,
                                                           verbose)
        # Retrieve the layer activations list
        self._layer_activations=['relu', 'relu']
        self._layer_filters = [64,64]
        self._kernel_size = 4
        self._dilated = False
        self._dropout = 0.0
        self._layer_activations, self._layer_filters, self._kernel_size,self._dropout, self._dilated = self.read_xml_config(arch_config)
        assert len(self._layer_activations)==len(self._layer_filters), "The number of activations must match the number of blocks"
        self.training=True

    def read_xml_config(self, config_element):
        layer_activations = []
        for activation_name in config_element.iter("activations").__next__().iter("element"):
            layer_activations.append(activation_name.attrib["value"])
        layer_filters = []
        # number of filters (units) in each layer
        for filters in config_element.iter("filters").__next__().iter("element"):
            layer_filters.append(int(filters.attrib["value"]))
        # kernel size
        kernel_size= int(config_element.iter("kernel_size").__next__().attrib["value"])
        # dropout
        dropout = float(config_element.iter("dropout").__next__().attrib["value"])
        dilated = (config_element.iter("dilated").__next__().attrib["value"] in ["true","True"])
        return layer_activations, layer_filters, kernel_size, dropout, dilated

    def add_subgraph(self, graph, verbose=False):
        # Retrieve arch input-output specifications
        inputs = self.get_input_nodes()
        outputs_name = self.get_output_names()


        # Append all operations to the specified tensorflow graph
        with graph.as_default():
            with tf.device(self.device):

                # Construct sub-graph inputs
                with tf.name_scope(self.operations_namescope) as opscope:
                    with tf.name_scope('inputs'):
                        # input layer
                        outputs = tf.identity(inputs[0], name='TCN_inputs')# Needed?

                # Construct sub-graph operations
                with tf.variable_scope(self.parameters_namescope):
                    # The model contains "num_levels" TemporalBlock
                    num_levels = len(self._layer_filters)
                    dilation_rate=1
                    for i in range(num_levels):
                        if self._dilated:
                            dilation_rate = 2 ** i                  # exponential growth
                        outputs = TemporalBlock(self._layer_activations[i],self._layer_filters[i], dilation_rate, self._kernel_size,
                                                padding='causal', dropout_rate=self._dropout)(outputs, self.training)
                # output scope
                with tf.name_scope(opscope):
                    with tf.name_scope('outputs'):
                        outputs = tf.identity(outputs, name='TCN_outputs')


        # Store inputs
        self.add_input_node(inputs)
        # Store outputs
        self.add_output_node(outputs)

layers = tf.keras.layers

class TemporalBlock(tf.keras.Model):
    def __init__(self, activation, nb_filters, dilation_rate, kernel_size,
                 padding, dropout_rate=0.0):
        super(TemporalBlock, self).__init__()
        init = tf.keras.initializers.glorot_normal()
        assert padding in ['causal', 'same']

        # block1
        self.conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, padding=padding)
        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation(activation)
        self.drop1 = layers.Dropout(rate=dropout_rate)

        # block2
        self.conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, padding=padding)
        self.batch2 = layers.BatchNormalization(axis=-1)
        self.ac2 = layers.Activation(activation)
        self.drop2 = layers.Dropout(rate=dropout_rate)

        # skip
        self.downsample = layers.Conv1D(filters=nb_filters, kernel_size=1,
                                        padding='same', kernel_initializer=init)
        self.ac3 = layers.Activation("linear")


    def call(self, x, training):
        prev_x = x
        x = self.conv1(x)
        #x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if training else x

        x = self.conv2(x)
        #x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if training else x

        if prev_x.shape[-1] != x.shape[-1]:    # match the dimensions
            prev_x = self.downsample(prev_x)

        return self.ac3(prev_x + x)

# Register architecture implementation
register("ARCH", TemporalConvolutionalNetwork)

# EOF
